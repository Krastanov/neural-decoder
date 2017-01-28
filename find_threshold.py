import argparse

parser = argparse.ArgumentParser(description='Find the threshold of a code with binary search.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='''\
The output is in the form
`(
 np.vstack([error probabilities,
            steps to failure for the small code,
            std errors on the above estimate,
            steps to failure for the large code,
            std errors on the above estimate]),
 steps to intersection,
 error probability intersection i.e. threshold
 )`
''')
parser.add_argument('dist1', type=int,
                    help='the distance of the smaller code')
parser.add_argument('dist2', type=int,
                    help='the distance of the bigger code')
parser.add_argument('out', type=str,
                    help='the name of the output file')
parser.add_argument('--samples', type=int, default=100000,
                    help='the number of samples to per point (default: %(default)s)')
parser.add_argument('--plow', type=float, default=0.82,
                    help='the lower bound of the search interval (default: %(default)s)')
parser.add_argument('--phigh', type=float, default=0.87,
                    help='the higher bound of he search interval (default: %(default)s)')

args = parser.parse_args()
print(args)

from codes import find_threshold

find_threshold(Lsmall=args.dist1, Llarge=args.dist2,
               p=(args.plow+args.phigh)/2, high=args.phigh, low=args.plow,
               samples=args.samples, logfile=args.out)