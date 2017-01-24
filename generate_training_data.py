import argparse

parser = argparse.ArgumentParser(description='Generate single-shot training data.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='''\
Samples without errors are skipped in the generation process.
A report of the performance of MWPM decoding is given.

The created file contains the following arrays:
(Zstab_x_train, Zstab_y_train, Xstab_x_train, Xstab_y_train,
 Zstab_x_test,  Zstab_y_test,  Xstab_x_test,  Xstab_y_test)''')
parser.add_argument('ntrain', type=int,
                    help='how many datapoints to generate in the training set')
parser.add_argument('nval', type=int,
                    help='how many datapoints to generate in the validation set')
parser.add_argument('prob', type=float,
                    help='the single physical qubit X/Z error probability')
parser.add_argument('dist', type=int,
                    help='the distance of the code')
parser.add_argument('out', type=str,
                    help='the name of the output file')

args = parser.parse_args()

from codes import generate_training_data
import numpy as np

res = generate_training_data(l=args.dist,
                             p=args.prob,
                             train_size=args.ntrain,
                             test_size=args.nval,
                            )

np.savez_compressed(args.out, *res)