import argparse

parser = argparse.ArgumentParser(description='Generate single-shot training data.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='''\
Samples without errors are skipped in the generation process.
A report of the performance of MWPM decoding is given.

The created file contains the following arrays:
(Zstab_x_train, Zstab_y_train, Xstab_x_train, Xstab_y_train,
 Zstab_x_test,  Zstab_y_test,  Xstab_x_test,  Xstab_y_test)''')
parser.add_argument('dist', type=int,
                    help='the distance of the code')
parser.add_argument('out', type=str,
                    help='the name of the output file')
parser.add_argument('--ntrain', type=int, default=2000000,
                    help='how many datapoints to generate in the training set (default: %(default)s)')
parser.add_argument('--nval', type=int, default=100000,
                    help='how many datapoints to generate in the validation set (default: %(default)s)')
parser.add_argument('--prob', type=float, default=0.9,
                    help='the probability of no error on the physical qubit (default: %(default)s)')

args = parser.parse_args()

from codes import generate_training_data
import numpy as np

res, _ = generate_training_data(l=args.dist,
                                p=args.prob,
                                train_size=args.ntrain,
                                test_size=args.nval,
                               )

np.savez_compressed(args.out, *res)
