"""Entry point: build the processed OCMR h5 dataset from raw OCMR files.
Only `--data_source ocmr` is supported here; it drives `helpers.OCMRPreprocessing`,
which retrospectively undersamples every selected acquisition and writes one h5
sample per slice, plus coil sensitivity maps (and, with `--cs_dir`, CS
reference reconstructions).

Output naming (preprocessing/core/ocmr.py: `partition`/`out_file` in
`save_data`, `output_name` in `process_sample`):
    <target_dir>/<data_source>_<partition>/<file stem>_sliceNN.h5
    partition = '<split>_<mask>_<acc:02d>' for test (e.g. `ocmr_test_gro_06`)
    and '<split>' for train/val; NN is the zero-padded 2-digit slice index.
    Each h5 stores k-space, mask, reference reconstructions and attributes.
Coil sensitivities:
    <coil_sens_dir>/<data_source>/<file stem>_sliceNN/coil_sens_avg.h5
CS reconstructions (only with `--cs_dir`):
    <cs_dir>/<data_source>_<mask>_<acc:02d>/<file stem>_sliceNN/reconstruction.h5
Metadata CSV: `preprocessing/data_attributes/<data_source>.csv`, unless
`--attributes_csv` points somewhere else.

Example:
    python preprocessing/run_ocmr.py \\
        --raw_dir data/OCMR/orig --target_dir data/OCMR/processed_ocmr \\
        --accelerations 6 --mask_types gro
"""

import argparse
import logging
import sys
from pathlib import Path

# make `helpers` importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.core.ocmr import OCMRPreprocessing

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s|%(message)s',
    datefmt='%Y-%m-%d %H-%M-%S',
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments
    parser.add_argument('--data_source', type=str,
                        default='ocmr',
                        help='Type of data that will be processed (only `ocmr` allowed for now)')
    parser.add_argument('--raw_dir', type=Path,
                        default=Path('data/OCMR/orig'),
                        help='Dir containing raw input files')
    parser.add_argument('--target_dir', type=Path,
                        default=Path('data/OCMR/processed_ocmr'),
                        help='Dir where processed data samples are stored')
    parser.add_argument('--coil_sens_dir', type=Path,
                        default=Path('data/OCMR/processed_ocmr/csm'),
                        help='Dir where coil sensitivities are located or will be stored, if computed')
    parser.add_argument('--cs_dir', nargs='?', type=Path,
                        help='Dir where CS reconstructions are located or will be stored, if computed')
    parser.add_argument('--csv_query', nargs='*', type=str, default=['smp=="fs"'],
                        help='Pandas query to filter data set')
    parser.add_argument('--csv_idx_range', nargs='*', type=int, default=[0, None],
                        help='Indices of data samples to process. (start) | (start, stop)')
    parser.add_argument('--accelerations', '--acceleration', nargs='+', type=int, default=[6],
                        help='Acceleration rates to use for retrospective undersampling')
    parser.add_argument('--mask_types', '--mask_type', nargs='+', type=str, default=['gro'],
                        choices=['random', 'random_acs', 'vista', 'gro', 'cava', 'equispaced', 'equispaced_acs'],
                        help='Types of k-space masks to use for retrospective undersampling')
    parser.add_argument('--coil_sens_backend', type=str, choices=['sigpy', 'bart'], default='sigpy',
                        help='Which backend to use to calc coil sensitivity maps')
    parser.add_argument('--n_maps', type=int, default=1,
                        help='Number of coil sensitivity maps to estimate')
    parser.add_argument('--attributes_csv', type=Path, default=None,
                        help='Path to the data-attributes CSV holding per-file metadata '
                             '(split, orientation, bbox, ...). Defaults to '
                             'preprocessing/data_attributes/<data_source>.csv')
    parser.add_argument('--no_cuda', dest='try_cuda', action='store_false',
                        help='Disable cuda even if available')
    args = parser.parse_args()

    # process args
    if args.csv_query:
        args.csv_query = ' and '.join(args.csv_query)

    if len(args.csv_idx_range) == 0:
        args.csv_idx_range = [0, None]
    elif len(args.csv_idx_range) == 1:
        args.csv_idx_range = [args.csv_idx_range[0], None]
    elif len(args.csv_idx_range) != 2:
        raise ValueError(f'Invalid csv_idx_range: {args.csv_idx_range}')

    # print values of all arguments
    logging.info(f'Arguments: ')
    for k, v in vars(args).items():
        logging.info(f'{k}: {v}')

    # run preprocessing
    if args.attributes_csv is None:
        args.attributes_csv = Path(__file__).resolve().parent / 'data_attributes' / f'{args.data_source}.csv'

    preprocessing = OCMRPreprocessing(
        args.attributes_csv,
        args.data_source,
        args.raw_dir,
        args.target_dir,
        args.coil_sens_dir,
        cs_dir=args.cs_dir,
        csv_query=args.csv_query,
        csv_idx_range=args.csv_idx_range,
        accelerations=args.accelerations,
        mask_types=args.mask_types,
        coil_sens_backend=args.coil_sens_backend,
        n_maps=args.n_maps,
        try_cuda=False,  # literal False: the parsed --no_cuda/--try_cuda flag is ignored here
                         # (contrast preprocessing/run_cmrxrecon23.py, which passes try_cuda=args.try_cuda)
    )
    preprocessing.run()
