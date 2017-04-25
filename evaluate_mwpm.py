import argparse

parser = argparse.ArgumentParser(description='Evaluate MWPM.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='''\
A report of the performance of MWPM decoding is given.
''')
parser.add_argument('dist', type=int,
                    help='the distance of the code')
parser.add_argument('out', type=str,
                    help='the name of the output file')
parser.add_argument('--neval', type=int, default=100000,
                    help='how many datapoints to generate in the evaluation set (default: %(default)s)')
parser.add_argument('--prob', type=float, default=0.9,
                    help='the probability of no error on the physical qubit (default: %(default)s)')

args = parser.parse_args()

from codes import generate_training_data
import numpy as np

_, fractions =  generate_training_data(l=args.dist,
                                       p=args.prob,
                                       train_size=0,
                                       test_size=args.neval,
                                      )

np.savetxt(args.out, fractions)
