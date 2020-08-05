Compute cluster-specific ATAC coverage from a fragments file and clusters table.
See 5k_pbmc_coverages.ipynb for an example applied to a 5000 immune cell dataset.

Currently, bug in cuDF causing incomplete reading of fragments file. Will port initial fragment filtering step to tabix to circumvent this issue.
