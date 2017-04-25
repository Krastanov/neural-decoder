'''
Tested only in interactive use with the Jupyter notebook.
Some tools might fail without it due to the use of `tnrange`
and `tqdm_notebook` from `tqdm`. Similarly `matplotlib` is
used with its interactive interface, which might cause trouble
if `ioff` is not called.'''

import itertools

import numpy as np
import scipy.linalg
import scipy.stats as stats
import scipy.optimize as optimize

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

import networkx as nx

from tqdm import tqdm, trange

try:
    from IPython import display
except ImportError:
    pass


class ToricCode:
    '''

    ::

        Lattice:
        X00--Q00--X01--Q01--X02...
         |         |         |
        Q10  Z00  Q11  Z01  Q12
         |         |         |
        X10--Q20--X11--Q21--X12...
         .         .         .
    '''
    def __init__(self, L):
        '''Toric code of ``2 L**2`` physical qubits and distance ``L``.'''
        self.L = L
        self.Xflips = np.zeros((2*L,L), dtype=np.dtype('b')) # qubits where an X error occured
        self.Zflips = np.zeros((2*L,L), dtype=np.dtype('b')) # qubits where a  Z error occured
        self._Xstab = np.empty((L,L), dtype=np.dtype('b'))
        self._Zstab = np.empty((L,L), dtype=np.dtype('b'))

    @property
    def flatXflips2Zstab(self):
        L = self.L
        _flatXflips2Zstab = np.zeros((L**2, 2*L**2), dtype=np.dtype('b'))
        for i, j in itertools.product(range(L),range(L)):
            _flatXflips2Zstab[i*L+j, (2*i  )%(2*L)*L+(j  )%L] = 1
            _flatXflips2Zstab[i*L+j, (2*i+1)%(2*L)*L+(j  )%L] = 1
            _flatXflips2Zstab[i*L+j, (2*i+2)%(2*L)*L+(j  )%L] = 1
            _flatXflips2Zstab[i*L+j, (2*i+1)%(2*L)*L+(j+1)%L] = 1
        return _flatXflips2Zstab

    @property
    def flatZflips2Xstab(self):
        L = self.L
        _flatZflips2Xstab = np.zeros((L**2, 2*L**2), dtype=np.dtype('b'))
        for i, j in itertools.product(range(L),range(L)):
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+1)%(2*L)*L+(j+1)%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+2)%(2*L)*L+(j  )%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+3)%(2*L)*L+(j+1)%L] = 1
            _flatZflips2Xstab[(i+1)%L*L+(j+1)%L, (2*i+2)%(2*L)*L+(j+1)%L] = 1
        return _flatZflips2Xstab

    @property
    def flatXflips2Zerr(self):
        L = self.L
        _flatXflips2Zerr = np.zeros((2, 2*L**2), dtype=np.dtype('b'))
        for k in range(L):
            _flatXflips2Zerr[0, (2*k+1)%(2*L)*L+(0  )%L] = 1
            _flatXflips2Zerr[1, (2*0  )%(2*L)*L+(k  )%L] = 1
        return _flatXflips2Zerr

    @property
    def flatZflips2Xerr(self):
        L = self.L
        _flatZflips2Xerr = np.zeros((2, 2*L**2), dtype=np.dtype('b'))
        for k in range(L):
            _flatZflips2Xerr[0, (2*0+1)%(2*L)*L+(k  )%L] = 1
            _flatZflips2Xerr[1, (2*k  )%(2*L)*L+(0  )%L] = 1
        return _flatZflips2Xerr

    def H(self, Z=True, X=False):
        H = []
        if Z:
            H.append(self.flatXflips2Zstab)
        if X:
            H.append(self.flatZflips2Xstab)
        H = scipy.linalg.block_diag(*H)
        return H

    def E(self, Z=True, X=False):
        E = []
        if Z:
            E.append(self.flatXflips2Zerr)
        if X:
            E.append(self.flatZflips2Xerr)
        E = scipy.linalg.block_diag(*E)
        return E

    def Zstabilizer(self):
        '''Return all measurements of the Z stabilizer with ``true`` marking non-trivial.'''
        stab = self._Zstab
        X = self.Xflips
        stab[0:-1,0:-1] = X[0:-2:2,0:-1:] ^ X[1:-1:2,0:-1:] ^ X[2::2,0:-1:] ^ X[1:-1:2,1::]
        stab[  -1,0:-1] = X[  -2  ,0:-1:] ^ X[  -1  ,0:-1:] ^ X[  0 ,0:-1:] ^ X[  -1  ,1::]
        stab[0:-1,  -1] = X[0:-2:2,  -1 ] ^ X[1:-1:2,  -1 ] ^ X[2::2,  -1 ] ^ X[1:-1:2,  0]
        stab[  -1,  -1] = X[  -2  ,  -1 ] ^ X[  -1  ,  -1 ] ^ X[  0 ,  -1 ] ^ X[  -1  ,  0]
        return stab

    def Xstabilizer(self):
        '''Return all measurements of the X stabilizer with ``true`` marking non-trivial.'''
        stab = self._Xstab
        Z = self.Zflips
        stab[1:,1:] = Z[1:-2:2,1:] ^ Z[2:-1:2,0:-1] ^ Z[3::2,1:] ^ Z[2:-1:2,1:]
        stab[0 ,1:] = Z[  -1  ,1:] ^ Z[   0  ,0:-1] ^ Z[  1 ,1:] ^ Z[   0  ,1:]
        stab[1:,0 ] = Z[1:-2:2,0 ] ^ Z[2:-1:2,  -1] ^ Z[3::2,0 ] ^ Z[2:-1:2,0 ]
        stab[0 ,0 ] = Z[  -1  ,0 ] ^ Z[   0  ,  -1] ^ Z[  1 ,0 ] ^ Z[   0  ,0 ]
        return stab

    def _plot_flips(self, s, flips_yx, label):
        '''Given an array of yx coordiante plot qubit flips on subplot ``s``.'''
        if not len(flips_yx): return
        y, x = flips_yx
        x = x.astype(float)
        x[y%2==0] += 0.5
        x = np.concatenate([x, x-self.L, x])
        y = np.concatenate([y/2., y/2., y/2.-self.L])
        s.plot(x, y,'o', ms=50/self.L, label=label)

    def plot(self, legend=True, stabs=True):
        '''Plot the state of the system (including stabilizers).'''
        f = plt.figure(figsize=(5,5))
        s = f.add_subplot(1,1,1)
        self._plot_legend = legend

        self._plot_flips(s, self.Xflips.nonzero(), label='X')
        self._plot_flips(s, self.Zflips.nonzero(), label='Z')
        self._plot_flips(s, (self.Xflips & self.Zflips).nonzero(), label='Y')

        if stabs:
            y, x = self.Zstabilizer().nonzero()
            x = np.concatenate([x+0.5, x+0.5-self.L, x+0.5, x+0.5-self.L])
            y = np.concatenate([y+0.5, y+0.5, y+0.5-self.L, y+0.5-self.L])
            s.plot(x,y,'s', mew=0, ms=190/self.L, label='plaq')

            y, x = self.Xstabilizer().nonzero()
            s.plot(x, y, '+', mew=100/self.L, ms=200/self.L, label='star')

        s.set_xticks(range(0,self.L))
        s.set_yticks(range(0,self.L))
        s.set_xlim(-0.6,self.L-0.4)
        s.set_ylim(-0.6,self.L-0.4)
        s.invert_yaxis()
        for tic in s.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        for tic in s.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
        s.grid()
        if legend: s.legend()
        return f, s

    def _wgraph(self, operator):
        g = nx.Graph()
        if operator == 'Z':
            nodes = zip(*self.Zstabilizer().nonzero())
        elif operator == 'X':
            nodes = zip(*self.Xstabilizer().nonzero())
        def dist(node1, node2):
            dy = abs(node1[0]-node2[0])
            dy = min(self.L-dy, dy)
            dx = abs(node1[1]-node2[1])
            dx = min(self.L-dx, dx)
            return dx+dy
        g.add_weighted_edges_from((node1, node2, -dist(node1, node2))
            for node1, node2 in itertools.combinations(nodes, 2))
        return g

    def Zwgraph(self):
        '''The distance graph for non-trivial Z stabilizer.'''
        return self._wgraph('Z')

    def Xwgraph(self):
        '''The distance graph for non-trivial X stabilizer.'''
        return self._wgraph('X')

    def Zcorrections(self):
        '''Qubits on which to apply Z operator to fix the X stabilizer.'''
        L = self.L
        graph = self.Xwgraph()
        matches = {tuple(sorted(_)) for _ in
                   nx.max_weight_matching(graph, maxcardinality=True).items()}
        qubits = set()
        for (y1, x1), (y2, x2) in matches:
            ym, yM = 2*min(y1, y2), 2*max(y1, y2)
            if yM-ym > L:
                ym, yM = yM, ym+2*L
                horizontal = yM if (x2-x1)*(y2-y1)<0 else ym
            else:
                horizontal = ym if (x2-x1)*(y2-y1)<0 else yM
            xm, xM = min(x1, x2), max(x1, x2)
            if xM-xm > L/2:
                xm, xM = xM, xm+L
                vertical = xM
            else:
                vertical = xm
            qubits.update((horizontal%(2*L), _%L) for _ in range(xm, xM))
            qubits.update(((_+1)%(2*L), vertical%L) for _ in range(ym, yM, 2))
        return matches, qubits

    def Xcorrections(self):
        '''Qubits on which to apply X operator to fix the Z stabilizer.'''
        L = self.L
        graph = self.Zwgraph()
        matches = {tuple(sorted(_)) for _ in
                   nx.max_weight_matching(graph, maxcardinality=True).items()}
        qubits = set()
        for (y1, x1), (y2, x2) in matches:
            ym, yM = 2*min(y1, y2), 2*max(y1, y2)
            if yM-ym > L:
                ym, yM = yM, ym+2*L
                horizontal = yM if (x2-x1)*(y2-y1)<0 else ym
            else:
                horizontal = ym if (x2-x1)*(y2-y1)<0 else yM
            xm, xM = min(x1, x2), max(x1, x2)
            if xM-xm > L/2:
                xm, xM = xM, xm+L
                vertical = xM
            else:
                vertical = xm
            qubits.update(((horizontal+1)%(2*L), (_+1)%L) for _ in range(xm, xM))
            qubits.update(((_+2)%(2*L), vertical%L) for _ in range(ym, yM, 2))
        return matches, qubits

    def plot_corrections(self, s, plot_matches=False):
        '''Add to subplot ``s`` the corrections that have to be performed according to min. weight matching.'''
        def stitch_torus(y1, y2):
            if abs(y1-y2)>L/2:
                return (y1+L, y2-L) if y1<y2 else (y1-L, y2+L)
            return y1, y2
        def shorten(y1,y2):
            if y1==y2:
                return y1, y2
            return (y1+0.15, y2-0.15) if y1<y2 else (y1-0.15, y2+0.15)
        S = shorten
        matches, qubits = self.Xcorrections()
        L = self.L
        if matches:
            if plot_matches:
                for ((y1,x1),(y2,x2)) in np.array(list(matches))+0.5:
                    Y1, Y2 = stitch_torus(y1,y2)
                    X1, X2 = stitch_torus(x1,x2)
                    s.plot(S(x1,X2), S(y1,Y2), 'k-', lw=20/self.L)
                    s.plot(S(X1,x2), S(Y1,y2), 'k-', lw=20/self.L)
            y, x = np.array(list(qubits)).T
            cX = np.array([y,x])
        else:
            cX = np.array([[],[]])

        matches, qubits = self.Zcorrections()
        if matches:
            if plot_matches:
                matches = np.array(list(matches))
                for ((y1,x1),(y2,x2)) in matches:
                    Y1, Y2 = stitch_torus(y1,y2)
                    X1, X2 = stitch_torus(x1,x2)
                    s.plot(S(x1,X2), S(y1,Y2), 'k-', lw=20/self.L)
                    s.plot(S(X1,x2), S(Y1,y2), 'k-', lw=20/self.L)
            y, x = np.array(list(qubits)).T
            cZ = np.array([y,x])
        else:
            cZ = np.array([[],[]])
        self._plot_flips(s, cX, label='cX')
        self._plot_flips(s, cZ, label='cZ')
        cY = np.array(list(set(zip(*cZ)).intersection(set(zip(*cX))))).T
        self._plot_flips(s, cY, label='cY')
        if self._plot_legend: s.legend()

    def add_errors(self, p): #TODO probably faster with numba
        '''Add X, Y, Z errors at rate ``(1-p)/3`` each, e.g. depolarization at ``1-p``.'''
        rand = np.random.rand(self.L*2, self.L)
        q = (1-p)/3
        x_flips =                rand<  q
        z_flips =   (q<=rand) & (rand<2*q)
        both    = (2*q<=rand) & (rand<3*q)
        self.Xflips ^= x_flips
        self.Xflips ^= both
        self.Zflips ^= z_flips
        self.Zflips ^= both

    def perform_perfect_correction(self):
        self.Xflips[list(zip(*self.Xcorrections()[1]))] ^= True
        self.Zflips[list(zip(*self.Zcorrections()[1]))] ^= True

    def logical_errors(self):
        z1 = np.logical_xor.reduce(self.Xflips[1::2,0])
        z2 = np.logical_xor.reduce(self.Xflips[0,:])
        x1 = np.logical_xor.reduce(self.Zflips[1,:])
        x2 = np.logical_xor.reduce(self.Zflips[0::2,0])
        return z1, z2, x1, x2

    def step_error_and_perfect_correction(self, p):
        self.add_errors(p)
        self.perform_perfect_correction()
        return not any(self.logical_errors())

    @staticmethod
    def assert_correctness():
        '''A bunch of functionality is implemented in multiple ways - here we assert they are equivalent.'''
        c = 0
        while c<1000:
            t = ToricCode(10)
            t.add_errors(0.750)
            # Computing stabilizers and measurements with linear algebra and with explicit elementwise ops.
            stabz = t.Zstabilizer().ravel()
            stabzm = t.flatXflips2Zstab.dot(t.Xflips.ravel()) % 2
            stabx = t.Xstabilizer().ravel()
            stabxm = t.flatZflips2Xstab.dot(t.Zflips.ravel()) % 2
            errz = t.logical_errors()[0:2]
            errx = t.logical_errors()[2:4]
            errzm = t.flatXflips2Zerr.dot(t.Xflips.ravel()) % 2
            errxm = t.flatZflips2Xerr.dot(t.Zflips.ravel()) % 2
            assert np.all(stabz==stabzm)
            assert np.all(stabx==stabxm)
            assert np.all(errz==errzm)
            assert np.all(errx==errxm)
            c += 1
            if not c%100:
                print('\r',c,end='',flush=True)


def sample(L, p, samples=1000, cutoff=200):
    '''Repeated single shot corrections for the toric code with perfect measurements.

    Return an array of nb of cycles until failure for a given L and p.'''
    results = []
    for _ in trange(samples, desc='%d; %.2f'%(L,p), leave=False):
        code = ToricCode(L)
        i = 1
        while code.step_error_and_perfect_correction(p) and i<cutoff:
            i+=1
        results.append(i)
    return np.array(results, dtype=int)

def stat_estimator(samples, cutoff=200, confidence=0.99):
    '''Max Likelihood Estimator for censored exponential distribution.

    See "Estimation of Parameters of Truncated or Censored Exponential Distributions",
    Walter L. Deemer and David F. Votaw'''
    samples = samples.astype(float)
    n = (samples<cutoff).sum()
    N = len(samples)
    estimate = n/samples.sum()
    y_conf = stats.norm.ppf((1+confidence)/2)
    y = lambda c: N**0.5*(estimate/c-1)*(1-np.exp(-c*cutoff))**0.5
    low  = optimize.root(lambda c: y(c)-y_conf, estimate)
    high = optimize.root(lambda c: y(c)+y_conf, estimate)
    if not (low.success and high.success):
        raise RuntimeError('Could not find confidence interval for the given samples!')
    return np.array([1/estimate, 1/high.x, 1/low.x])

def find_threshold(Lsmall=3, Llarge=5, p=0.8, high=1, low=0.79, samples=1000, logfile=None):
    '''Use binary search (between two sizes of codes) to find the threshold for the toric code.'''
    ps = []
    samples_small = []
    samples_large = []
    def step(p):
        ps.append(p)
        samples_small.append(stat_estimator(sample(Lsmall, p, samples=samples)))
        samples_large.append(stat_estimator(sample(Llarge, p, samples=samples)))
    def intersection(xs, y1s, y2s, log=True):
        d = np.linalg.det
        if log:
            y1s, y2s = np.log([y1s,y2s])
        ones = np.array([1.,1.])
        dx  = d([xs , ones])
        dy1 = d([y1s, ones])
        dy2 = d([y2s, ones])
        x = (d([xs, y1s])-d([xs, y2s])) / (dy2-dy1)
        y = (d([xs, y1s])*dy2 - d([xs, y2s])*dy1) / dx / (dy2-dy1)
        if log:
            y = np.exp(y)
        return x, y
    step(p)
    if logfile:
        with open(logfile, 'w') as f:
            ss = samples_small[0]
            sl = samples_large[0]
            f.write(str((np.vstack([ps, [ss[0]], [ss[1]-ss[0]], [ss[2]-ss[0]], [sl[0]], [sl[1]-sl[0]], [sl[2]-sl[0]]]), (ss[0]+sl[0])/2, ps[0])))
    else:
        f = plt.figure()
        s = f.add_subplot(1,1,1)
    while not (samples_large[-1][1]<samples_small[-1][0]<samples_large[-1][2]
            or samples_small[-1][1]<samples_large[-1][0]<samples_small[-1][2]):
        if samples_small[-1][0]<samples_large[-1][0]:
            p, high = low+(ps[-1]-low)/2, p
        else:
            p, low = ps[-1]+(high-ps[-1])/2, p
        step(p)
        _argsort = np.argsort(ps)
        _ps = np.array(ps)[_argsort]
        _ss = np.array(samples_small)
        _small = _ss[_argsort,0]
        _small_err = np.abs(_ss[_argsort,1:].T - _small)
        _sl = np.array(samples_large)
        _large = _sl[_argsort,0]
        _large_err = np.abs(_sl[_argsort,1:].T - _large)
        ix, iy = intersection(ps[-2:],[_[0] for _ in samples_small[-2:]],[_[0] for _ in samples_large[-2:]])
        if logfile:
            with open(logfile, 'w') as f:
                f.write(str((np.vstack([_ps, _small, _small_err, _large, _large_err]), iy, ix)))
        else:
            s.clear()
            s.errorbar(_ps,_small,yerr=_small_err,alpha=0.6,label=str(Lsmall))
            s.errorbar(_ps,_large,yerr=_large_err,alpha=0.6,label=str(Llarge))
            s.plot([ix],[iy],'ro',alpha=0.5)
            s.set_title('intersection at p = %f'%ix)
            s.set_yscale('log')
            display.clear_output(wait=True)
            display.display(f)

    return ps, samples_small, samples_large

def generate_training_data(l=3, p=0.9, train_size=2000000, test_size=100000): # TODO duplicated code with data_generator in neural.py
    '''Generate errors and corresponding stabilizers at a given `p` for the toric code.

    The samples with no errors are skipped.
    It counts and prints out how many of the errors are fixed by MWPM.

    returns: (Zstab_x_train, Zstab_y_train, Xstab_x_train, Xstab_y_train,
              Zstab_x_test,  Zstab_y_test,  Xstab_x_test,  Xstab_y_test)'''
    Zstab_x_train = np.zeros((train_size, l**2))
    Zstab_y_train = np.zeros((train_size, 2*l**2))
    Xstab_x_train = np.zeros((train_size, l**2))
    Xstab_y_train = np.zeros((train_size, 2*l**2))
    for i in trange(train_size):
        t = ToricCode(l)
        t.add_errors(p)
        while not (np.any(t.Xflips) or np.any(t.Zflips)):
            t = ToricCode(l)
            t.add_errors(p)
        Zstab_x_train[i,:] = t.Zstabilizer().ravel()
        Zstab_y_train[i,:] = t.Xflips.ravel()
        Xstab_x_train[i,:] = t.Xstabilizer().ravel()
        Xstab_y_train[i,:] = t.Zflips.ravel()
    Zstab_x_test = np.zeros((test_size, l**2))
    Zstab_y_test = np.zeros((test_size, 2*l**2))
    Xstab_x_test = np.zeros((test_size, l**2))
    Xstab_y_test = np.zeros((test_size, 2*l**2))
    errors = xstab_errors = zstab_errors = 0
    for i in trange(test_size):
        t = ToricCode(l)
        t.add_errors(p)
        while not (np.any(t.Xflips) or np.any(t.Zflips)):
            t = ToricCode(l)
            t.add_errors(p)
        Zstab_x_test[i,:] = t.Zstabilizer().ravel()
        Zstab_y_test[i,:] = t.Xflips.ravel()
        Xstab_x_test[i,:] = t.Xstabilizer().ravel()
        Xstab_y_test[i,:] = t.Zflips.ravel()
        t.perform_perfect_correction()
        errors += any(t.logical_errors())
        xstab_errors += any(t.logical_errors()[0:2])
        zstab_errors += any(t.logical_errors()[2:4])
    decoded_fraction = 1 - errors/test_size
    xstab_decoded_fraction = 1 - xstab_errors/test_size
    zstab_decoded_fraction = 1 - zstab_errors/test_size
    print('decoded_fraction, zstab_decoded_fraction, xstab_decoded_fraction =')
    print(decoded_fraction, zstab_decoded_fraction, xstab_decoded_fraction)
    return ((Zstab_x_train, Zstab_y_train, Xstab_x_train, Xstab_y_train,
             Zstab_x_test,  Zstab_y_test,  Xstab_x_test,  Xstab_y_test),
            (decoded_fraction, zstab_decoded_fraction, xstab_decoded_fraction))
