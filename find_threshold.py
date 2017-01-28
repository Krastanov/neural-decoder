import argparse

parser = argparse.ArgumentParser(description='Find the threshold of a code.',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='''\
Either binary search between two different size
codes is used:

    In which case the output is in the form
    `(
     np.vstack([error probabilities,
                steps to failure for the small code,
                std errors on the above estimate (2 rows),
                steps to failure for the large code,
                std errors on the above estimate (2 rows)]),
     'steps to' intersection,
     'error probability' intersection i.e. threshold
     )`

or for a single code its "average steps to failure" are
computed:

    In which case the output is in the form
    `error probabilities,
     steps to failure (nb. of samples rows)])`
''')
parser.add_argument('dist', type=int,
                    help='the distance of the smaller code (could be the only code)')
parser.add_argument('out', type=str,
                    help='the name of the output file')
parser.add_argument('--dist2', type=int,
                    help='the distance of the bigger code (if present, a binary search is done)')
parser.add_argument('--samples', type=int, default=100000,
                    help='the number of samples to per point (default: %(default)s)')
parser.add_argument('--plow', type=float, default=0.82,
                    help='the lower bound of the search interval (default: %(default)s)')
parser.add_argument('--phigh', type=float, default=0.87,
                    help='the higher bound of he search interval (default: %(default)s)')
parser.add_argument('--steps', type=int, default=10,
                    help='the number of steps between phigh and plow (used only if not doing a binary search) (default: %(default)s)')

args = parser.parse_args()
print(args)

from codes import find_threshold, sample
import numpy as np
from tqdm import tqdm

if args.dist2:
    find_threshold(Lsmall=args.dist, Llarge=args.dist2,
		   p=(args.plow+args.phigh)/2, high=args.phigh, low=args.plow,
		   samples=args.samples, logfile=args.out)
else:
    ps = np.linspace(args.plow, args.phigh, args.steps+1)[:-1]
    r = []
    for p in tqdm(ps):
        r.append(sample(args.dist, p, args.samples))
        np.savetxt(args.out, np.vstack([ps[:len(r)], np.array(r).T]), fmt='%.8e')
