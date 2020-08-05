# %%
from subprocess import Popen, PIPE
import cudf
import pandas as pd
import numpy as np
from numba import cuda
import time

FRAGMENTS_FILE = './coverage_from_fragments/example.bed'
GROUPING_FILE = './coverage_from_fragments/grouping_table.txt'

COLUMN_NAMES = ['chrom', 'start', 'end', 'cell']


def query_fragments(df, chrom_map, chrom, start, end):
    """Call tabix and generate an array of strings for each line it returns."""

    chrHash = chrom_map[chrom]
    fdf = df.query('chrom_hashed == @chrHash and start >= ' +
                   str(start) + ' and end <= ' + str(end))
    return fdf


def get_aggregate_score(cells_in_cluster, sum_inversed_reads, normalized_total):
    # value scaled down by its cell's total coverage
    # also normalized by total number of cells in its cluster
    for i in range(cuda.threadIdx.x, len(cells_in_cluster), cuda.blockDim.x):
        normalized_total[i] = sum_inversed_reads[i] * (1.0 / cells_in_cluster[i])
    
    # values = group['value']
    # total_cell_reads = group['total_reads']
    # cells_in_cluster = group['cells_in_cluster']
    # return np.sum(1 / total_cell_reads) * (1.0 / cells_in_cluster[0])


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


def get_coverages(df, chrom_map, chrom, start, end,
                  group_file, resolution=1):
    t0 = time.time()
    reads = query_fragments(df, chrom_map, chrom, start, end)

    print("Tabix query took %.2fs" % (time.time() - t0))
    grouping = cudf.read_csv(group_file, sep='\t')
    # print(grouping.head())

    t_start = time.time()

    # Add column with count of cells in cluster
    t0 = time.time()
    grp_cnt = grouping.groupby(['group'], as_index=False).count()
    # grp_cnt has counts of cell and total_reads. 
    # Drop Total_reads and rename 'cell' label to something relavent(cells_in_cluster)
    print('Computing cell count in group/cluster...')
    grp_cnt.drop(['total_reads'], inplace=True)
    grp_cnt.rename({'cell': 'cells_in_cluster'}, inplace=True)

    grouping = grouping.merge(grp_cnt, on=['group'])
    grouping.fillna({'cells_in_cluster': 0}, inplace=True)

    print("groupBy cluster took %.2fs" % (time.time() - t0))
    
    # Filter df to only include reads for cells in groups file
    t0 = time.time()
    reads = reads[reads['cell'].isin(grouping['cell'])]
    print("filtering reads took %.2fs" % (time.time() - t0))

    # Add total reads per cell as a column for each read in reads df
    t0 = time.time()

    reads = reads.merge(grouping, on=['cell'])
    # print(reads.head())

    print("merging groups with reads took %.2fs" % (time.time() - t0))

    # Assign each read a value of 1 (placeholder in case we would like to weight reads later)
    reads['value'] = 1

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
    reads.drop(['start', 'end'], inplace=True)
    intervals = intervals.merge(reads, on='row_num')
    intervals.drop(['row_num'], inplace=True)
    # print(intervals.head())

    intervals = intervals.apply_rows(inverse, 
        incols=['total_reads'], 
        outcols={'inversed_reads': np.float}, kwargs={})
    # print(intervals.head())

    print("expanding reads to bp resolution took %.2fs" % (time.time() - t0))

    # Combine reads at shared positions for a given cluster
    t0 = time.time()

    intervals_repl = intervals.copy()
    intervals_repl.drop(
        ['chrom_hashed', 'total_reads', 'cells_in_cluster', 'value', 'diff'],
        inplace=True)
    reads_sum_group = intervals_repl.groupby(
        ['chrom', 'start', 'end', 'group'], as_index=False).sum()
    reads_sum_group.rename({'inversed_reads': 'sum_inversed_reads'}, inplace=True)

    intervals = intervals.merge(reads_sum_group, on=['chrom', 'start', 'end', 'group'])
    # print(intervals)

    grouped = intervals.groupby(
        ['chrom', 'start', 'end', 'group'], as_index=False)

    intervals = grouped.apply_grouped(
        get_aggregate_score,
        incols=['cells_in_cluster', 'sum_inversed_reads'],
        outcols={'normalized_total': np.float})
    # print(intervals.head())

    intervals.reset_index(inplace=True)
    print("getting scores per bp took %.2fs" % (time.time() - t0))
    print("Total processing took %.2fs" % (time.time() - t_start))

    return intervals


def initialize(fragments_file):

    df = cudf.read_csv(fragments_file, sep='\t',
                       header=None, names=COLUMN_NAMES)

    df['chrom_hashed'] = df.hash_columns(['chrom'])
    df['row_num'] = df.index  # Used to relate bed info to interval info

    chrom_map = {}
    for name, group in df.groupby(['chrom', 'chrom_hashed']):
        chrom_map[name[0]] = name[1]

    return df, chrom_map


if __name__ == "__main__":
    t0 = time.time()
    df, chrom_map = initialize(FRAGMENTS_FILE)
    print("Initialization took %.2fs" % (time.time() - t0))

    intervals = get_coverages(df, chrom_map,
                              'chr11', 118202327, 118226182,
                              GROUPING_FILE)
    # print(intervals.head())

# %%
