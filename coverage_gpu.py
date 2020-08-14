# %%
from subprocess import Popen, PIPE

import os
import cudf
import time
import tabix
import numpy as np
import pandas as pd
from numba import cuda

import gzip

from collections.abc import Sequence


def query_fragments(fragment_file, chrom, start, end):
    tb = tabix.open(fragment_file)
    results = tb.querys("%s:%d-%d" % (chrom, start, end))
    records = []
    for record in results:
        records.append(record)
    return records


def tabix_query(filename, chrom, start, end):
    """Call tabix and generate an array of strings for each line it returns."""
    query = '{}:{}-{}'.format(chrom, start, end)
    process = Popen(['tabix', '-f', filename, query], stdout=PIPE)
    records = []
    for line in process.stdout:
        record = line.decode('utf-8').strip().split('\t')
        records.append(record)
    return records


def get_aggregate_score(cells_in_cluster, sum_inversed_reads, normalized_total):
    # value scaled down by its cell's total coverage
    # also normalized by total number of cells in its cluster
    for i in range(cuda.threadIdx.x, len(cells_in_cluster), cuda.blockDim.x):
        normalized_total[i] = sum_inversed_reads[i] * (1.0 / cells_in_cluster[i])


def inverse(total_reads, inversed_reads):
    for i, (total_read) in enumerate(total_reads):
        inversed_reads[i] = (1 / total_read)


@cuda.jit
def expand_interval(start, end, index, end_index,
                    interval_start, interval_end, interval_index, step):
    for i in range(cuda.threadIdx.x, start.size, cuda.blockDim.x):
        # Starting position in the target frame
        first_index = end_index[i] - (end[i] - start[i])
        chrom_start = start[i]
        for j in range(first_index, end_index[i], step):
            interval_start[j] = chrom_start
            chrom_start += 1
            interval_end[j] = chrom_start
            interval_index[j] = index[i]


def create_or_read_grouping_file(fragment_file, cluster_file, group_file):

    if os.path.exists(group_file):
        gdf = cudf.read_csv(group_file)
    else:
        fdf = cudf.read_csv(
            gzip.open(fragment_file), sep='\t', 
            names=['chrom', 'start', 'end', 'cell', 'duplicate'])

        cell_cnt_series = fdf.groupby(['cell'])['cell'].count()
        cell_gdf = cudf.DataFrame(
            {'cell':cell_cnt_series.index, 
            'total_reads':cell_cnt_series.values})

        cdf = cudf.read_csv(cluster_file)
        cdf.rename({'Barcode': 'cell', 'Cluster': 'group'}, inplace=True)

        gdf = cell_gdf.merge(cdf, on=['cell'])
        gdf.to_csv(group_file, index=False)
    return gdf


def read_data(chrom, start, end, fragment_file, cluster_file, group_file):
    t0 = time.time()
    #Create a DF from the output of tablix query
    reads = cudf.DataFrame(
        data=tabix_query(fragment_file, chrom, start, end),
        columns=['chrom', 'start', 'end', 'cell', 'duplicate'])
    reads['row_num'] = reads.index
    print("Tabix query took %.2fs" % (time.time() - t0))

    t0 = time.time()
    # Read grouping file
    grouping = create_or_read_grouping_file(fragment_file, cluster_file, group_file)
    print("Reading group file took %.2fs" % (time.time() - t0))

    t0 = time.time()
    reads = reads[reads['cell'].isin(grouping['cell'].tolist())]
    reads = reads.astype({"start": np.int32, "end": np.int32})
    # print(reads.shape)
    print("Filtering reads took and type conversion %.2fs" % (time.time() - t0))

    return reads, grouping


def get_coverages(chrom, start, end,
                  fragment_file, cluster_file, group_file, resolution=1):
    reads, grouping = read_data(
        chrom, start, end, fragment_file, cluster_file, group_file)

    # Add column with count of cells in cluster
    t0 = time.time()
    print('Computing cell count in group/cluster...')
    grp_cnt_series = grouping.groupby(['group'], as_index=False)['group'].count()
    grp_cnt = cudf.DataFrame({
        'group': grp_cnt_series.index, 
        'cells_in_cluster': grp_cnt_series.values})

    # grp_cnt has counts of cell and total_reads.
    # Drop Total_reads and rename 'cell' label to something relavent(cells_in_cluster)
    grouping = grouping.merge(grp_cnt, on=['group'])
    del grp_cnt
    print("GroupBy cluster took %.2fs" % (time.time() - t0))

    # Add total reads per cell as a column for each read in reads df
    t0 = time.time()
    reads = reads.merge(grouping, on=['cell'])
    del grouping
    # print(reads.head())
    print("Merging groups with reads took %.2fs" % (time.time() - t0))

    # Expand reads to preferred resolution
    t0 = time.time()
    reads['diff'] = reads['end'] - reads['start']
    sumdf = reads['diff'].cumsum()

    interval_size = reads['diff'].sum()
    print('Size of interval frame: ', interval_size)

    intervals = cudf.DataFrame()
    intervals['start'] = np.zeros(interval_size, dtype=np.int32)
    intervals['end'] = np.zeros(interval_size, dtype=np.int32)
    intervals['row_num'] = np.zeros(interval_size, dtype=np.int32)

    expand_interval.forall(reads.shape[0])(
        reads['start'],
        reads['end'],
        reads['row_num'],
        sumdf,
        intervals['start'],
        intervals['end'],
        intervals['row_num'],
        1
    )

    # Drop duplicate columns
    reads.drop(['start', 'end', 'diff'], inplace=True)
    intervals = intervals.merge(reads, on='row_num')
    intervals.drop(['row_num'], inplace=True)
    del reads

    # Create an column to store inverse of total_reads.
    intervals = intervals.apply_rows(inverse, 
        incols=['total_reads'], 
        outcols={'inversed_reads': np.float}, kwargs={})
    intervals.drop(['total_reads'], inplace=True)
    
    # print(intervals.head())
    print("expanding reads to bp resolution took %.2fs" % (time.time() - t0))

    t0 = time.time()

    # Sum inversed_reads by grouping 'chrom', 'start', 'end', 'group'
    # TODO: Is there a way to sum() just one column and merge it back to the df?
    # print(intervals.head())
    reads_sum_group = intervals.groupby(
        ['chrom', 'start', 'end', 'group'], as_index=False).sum()
    reads_sum_group.rename({'inversed_reads': 'sum_inversed_reads'}, inplace=True)
    reads_sum_group.drop(['cells_in_cluster'], inplace=True)
    
    intervals = intervals.merge(reads_sum_group, on=['chrom', 'start', 'end', 'group'])
    grouped = intervals.groupby(['chrom', 'start', 'end', 'group'], as_index=False)
    
    intervals = grouped.apply_grouped(
        get_aggregate_score,
        incols=['cells_in_cluster', 'sum_inversed_reads'],
        outcols={'normalized_total': np.float})

    intervals.drop(
        ['cell', 'cells_in_cluster', 'inversed_reads', 'sum_inversed_reads'], 
        inplace=True)

    print("getting scores per bp took %.2fs" % (time.time() - t0))
    return intervals


if __name__ == "__main__":
    t0 = time.time()

    FRAGMENTS_FILE = './coverage/data/500//example.bed.gz'
    GROUPING_FILE = './coverage_from_fragments/grouping_table.txt'

    COLUMN_NAMES = ['chrom', 'start', 'end', 'cell']

    intervals = get_coverages('chr11', 118202327, 118226182,
                              FRAGMENTS_FILE, GROUPING_FILE)
    print("Coverage took %.2fs" % (time.time() - t0))

# %%