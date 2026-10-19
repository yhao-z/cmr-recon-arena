"""Dataset preprocessing for cmr-recon-arena.

First layer = CLI entry points. Run these to build a dataset:

    python preprocessing/run_ocmr.py         --raw_dir ... --target_dir ...
    python preprocessing/run_cmrxrecon23.py  --raw_dir ... --target_dir ...

Layout:
    run_ocmr.py / run_cmrxrecon23.py   argparse wrappers: flags, defaults
                                       and output naming (thin -- read these
                                       first, they are the run commands)
    core/                              per-dataset implementations
        ocmr.py                        OCMRPreprocessing: reads ISMRMRD .h5
        cmrxrecon23.py                 CMRxRecon23Preprocessing: reads Challenge .mat
        scan_cmrxrecon_mat_csv.py      metadata CSV builder for CMRxRecon23
    data_attributes/                   metadata CSVs consumed by the pipelines

Both pipelines write one h5 per slice plus coil-sensitivity maps. Everything
generic (MRI math, losses/metrics, I/O, mask generation, raw-data readers)
lives in `helpers/` and knows nothing about specific datasets.
"""
