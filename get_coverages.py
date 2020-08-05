from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
import time


def expand_interval(interval, step=1):
    """
    Adapted from AtacWorks-0.3 repository.
    Expand an interval to single-base resolution and add scores.

    Args:
        interval : dict or DataFrame containing chrom, start, end and
        optionally scores.
        step : size of the sub-intervals that are being created

    Returns:
        expanded: pandas DataFrame containing a row for every base in the
        interval

    """
    # print(interval.keys())
    expanded = pd.DataFrame(columns=interval.keys())
    expanded['start'] = range(interval['start'], interval['end'], step)
    expanded['end'] = expanded['start'] + step
    if (interval['end'] - interval['start']) % step != 0:
        expanded['end'][-1] = interval['end']

    for col in expanded:
        if col == 'start' or col == 'end':
            continue
        expanded[col] = interval[col]
    return expanded


def tabix_query(filename, chrom, start, end):
    """Call tabix and generate an array of strings for each line it returns."""
    query = '{}:{}-{}'.format(chrom, start, end)
    process = Popen(['tabix', '-f', filename, query], stdout=PIPE)
    for line in process.stdout:
        yield line.strip().split()


def get_aggregate_score(group):
    # value scaled down by its cell's total coverage
    # also normalized by total number of cells in its cluster
    values = group['value']
    total_cell_reads = group['total_reads']
    cells_in_cluster = group['cells_in_cluster']
    return np.sum(1 / total_cell_reads) * (1.0 / cells_in_cluster[0])


def get_coverages(chrom, start, end,
                  fragment_file,
                  group_file,
                  resolution=1):
    t0 = time.time()
    reads = pd.DataFrame(tabix_query(fragment_file, chrom, start, end),
                         columns = ['chrom', 'start', 'end', 'cell'],
                         dtype = str
                         )
    print("Tabix query took %.2fs" % (time.time() - t0))
    reads['start'] = pd.to_numeric(reads['start'])
    reads['end'] = pd.to_numeric(reads['end'])

    grouping = pd.read_csv(group_file, sep='\t')

    t_start = time.time()

    # Add column with count of cells in cluster
    t0 = time.time()
    grouping['cells_in_cluster'] = grouping.groupby('group')['group'].transform('count')
    print("groupBy cluster took %.2fs" % (time.time() - t0))


    # Filter df to only include reads for cells in groups file
    t0 = time.time()
    reads = reads[ reads['cell'].isin(grouping['cell']) ]
    print("filtering reads took %.2fs" % (time.time() - t0))

    # Add total reads per cell as a column for each read in reads df
    t0 = time.time()
    reads = reads.merge(grouping[['cell', 'group', 'total_reads', 'cells_in_cluster']], on='cell')
    print("merging groups with reads took %.2fs" % (time.time() - t0))

    # Assign each read a value of 1 (placeholder in case we would like to weight reads later)
    reads['value'] = 1

    # Expand reads to preferred resolution
    t0 = time.time()
    intervals = pd.concat(list(reads.apply(expand_interval, axis=1)))
    print("expanding reads to bp resolution took %.2fs" % (time.time() - t0))

    # Combine reads at shared positions for a given cluster
    t0 = time.time()
    grouped = intervals.groupby(['chrom', 'start', 'end', 'group'], as_index=False)
    intervals = grouped.apply(get_aggregate_score)
    intervals = intervals.reset_index()
    intervals.rename(columns={0: 'normalized_total'}, inplace=True)
    print("getting scores per bp took %.2fs" % (time.time() - t0))

    print("Total processing took %.2fs" % (time.time() - t_start))

    return intervals

if __name__ == "__main__":
    intervals = get_coverages('chr11', 118202327, 118206327, 'example.bed.gz', 'grouping_table.txt')
    print(intervals.head())


