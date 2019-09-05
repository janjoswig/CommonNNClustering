"""
"""

import numpy as np
from scipy.linalg import eig
import warnings
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
from core.cnn import CNN

class CMSM():
    """Core set MSM class
    """
    
    def __init__(self, dtrajs=None, T=None, unit="ps", step=1):
        self.dtrajs = dtrajs
        self.T = T
        self.__tau = None
        self.M = None
        self.__Qminus = None
        self.__Qplus = None
        self.__F = None
        self.__B = None
        self.__eigenvectors_right = None
        self.__eigenvectors = self.__eigenvectors_right
        self.__eigenvectors_left = None
        self.__eigenvalues = None
        self.__unit = unit
        self.__step = step
        self.__its = None
        self.__Its = SortedDict()
        self.__mode = None
    
    @staticmethod
    def dtrajhandler(dtraj):
        if isinstance(dtraj, list):
            # TODO: format
            return dtraj, "memory"
        
        if isinstance(dtraj, CNN):
            raise NotImplementedError()

            return dtraj, "memory"
        


    @property
    def eigenvalues(self):
        if self.__eigenvalues is None:
            self.get_eigvalues()

        return self.__eigenvalues

    @property
    def eigenvectors_right(self):
        if self.__eigenvectors_right is None:
            _, self.eigenvectors_right = eig(self.__T, left=False, right=True)
        return self.__eigenvectors_right

    @property
    def eigenvectors_left(self):
        if self.__eigenvectors_left is None:
            _, self.eigenvectors_left = eig(self.__T, left=True, right=False)
        return self.__eigenvectors_left

    @property
    def tau(self):
        return self.__tau

    @property
    def mode(self):
        return self.__mode

    @property
    def dtrajs(self):
        return self.__dtrajs

    @dtrajs.setter
    def dtrajs(self, _dtrajs):
        # TODO modify setter, so dtraj can be passed correctly in
        # different formats: cobj, list of traces, path to file ...
        # Assume for now a list of np.ndarrays
        self.__dtrajs, self.__mode = self.dtrajhandler(_dtrajs)

    @property
    def its(self):
        if self.__its is None:
            self.get_its()
        return self.__its

    @property
    def Its(self):
        return self.__Its

    @property
    def unit(self):
        return self.__unit

    @property
    def step(self):
        return self.__step

    @property
    def T(self):
        return self.__T

    @T.setter
    def T(self, _T):
        self.__T = _T

    @property
    def M(self):
        return self.__M

    @M.setter
    def M(self, _M):
        self.__M = _M

    @property
    def Qminus(self):
        return self.__Qminus
    
    @property
    def Qplus(self):
        return self.__Qplus

    @property
    def F(self):
        return self.__F
    
    @property
    def B(self):
        return self.__B
    
    def get_milestoning(self, dtraj):
        """Neglect transitions to noise as state transition
        """

        qminus = np.copy(dtraj) # where do we come from?
        qplus = np.copy(dtraj) # where do we go next?

        for c, i in enumerate(dtraj[1:], 1):
            if i == 0:
                qminus[c] = qminus[c-1]

        for c, i in zip(range(len(dtraj[:-2]), -1, -1), dtraj[::-1][1:]):
            if i == 0:
                qplus[c] = qplus[c+1]

        return qminus, qplus

    def get_characteristicf(self, qminus, qplus, n_clusters):
        
        assert len(qminus) == len(qplus)
        
        chi_f = np.zeros((len(qminus), n_clusters), dtype=int)
        chi_b = np.zeros((len(qplus), n_clusters), dtype=int)
        
        for c, i in enumerate(range(n_clusters), 1):
            chi_f[qminus == c, i] = 1
            chi_b[qplus == c, i] = 1

        return chi_f, chi_b
    
    def get_connected_sets(self, T, largest='cluster_count', v=True):
        """
        largest: None, 'cluster_count', 'point_count'
        """

        rowsum = np.sum(T, axis=1)
        nonzerorows = np.nonzero(rowsum)[0]

        original_set = set(range(len(T)))
        connected_sets = []
        while original_set:
            try:
                root = next(x for x in original_set if x in nonzerorows)
            except StopIteration:
                break
            current_set = set([root])
            added_set = set([root])
            original_set.remove(root)
            while added_set:
                added_set_support = set()
                for cluster in added_set:
                    for connected_cluster in np.nonzero(T[cluster])[0]:
                        if connected_cluster not in current_set:
                            current_set.add(connected_cluster)
                            added_set_support.add(connected_cluster)
                            original_set.remove(connected_cluster)
                added_set = added_set_support.copy()
            connected_sets.append(current_set)
        
        if largest is None:
            return connected_sets
        elif largest == 'cluster_count':
            set_size = [len(x) for x in connected_sets]
            return connected_sets[np.argmax(set_size)]
        elif largest == 'point_count':
            raise NotImplementedError()
        else:
            raise ValueError(
f"""Invalid value {largest} for keyword argument 'largest'. Must be one of 
None, 'cluster_count', 'point_count'."""
                )
    
    @staticmethod
    def rownorm(T): 
        rowsum = np.sum(T, axis=1)
        return np.divide(T, rowsum[:, None])

    def correctnumerics(self, T=None, tol=1e-8, rownorm=True):
        # TODO: Maybe confusing -> separte instance from staticmethod
        if T is None:
        
            self.__T[abs(self.__T < tol)] = 0
            if rownorm:
                self.__T = self.rownorm(self.__T)

        else:
            T[abs(T < tol)] = 0
            if rownorm:
                T = rownorm(T)

    def get_transitionmatrix(self, F, B, lag=0, n_clusters=None, rownorm=True):
        lag = int(lag)
        self.__tau = lag

        if n_clusters == None:
           n_clusters = len(F[0][0])

        T = np.zeros((n_clusters, n_clusters))
        for c, chi_f in enumerate(F):
            T += np.dot(chi_f[:len(chi_f)-lag].T, B[c][lag:])
        
        # force symmetry
        T += T.T
        if rownorm:
            T = self.rownorm(T)
        
        return T
    
    @staticmethod
    def trimzeros(dtraj):
        nonzero = np.nonzero(dtraj)[0]
        try:
            first, last = nonzero[0], nonzero[-1]
        except IndexError:
            dtraj = np.array([])
        return dtraj[first:last+1]

    def cmsm(self, dtrajs=None, lag=1, minlenfactor=10, v=True, correct=False):
        """Estimate coreset markov model from characteristic functions
        at a given lagtime
        """
        if v:
            print(
f"\n*********************************************************\n" +
f"---------------------------------------------------------\n" +
f"Computing coreset MSM at lagtime {lag*self.__step} {self.__unit}\n" +
f"---------------------------------------------------------\n"
            )

        if dtrajs is None:
            dtrajs = self.__dtrajs

        # Remove leading and trailing 0s in trajectory
        # for dtraj in dtrajs:
        dtrajs = [
            self.trimzeros(x)
            for x in dtrajs
            ]

        # Check if the provided trajectories are long enough
        length = [len(x) for x in dtrajs]
        threshold = minlenfactor*lag
        empty = []
        tooshort = []
        for c, i in enumerate(length):
            if i == 0:
                # warnings.warn(, UserWarning)
                empty.append(c)
            elif i < threshold:
                # warnings.warn(, UserWarning)
                tooshort.append(c)

        dtrajs = [x for c, x in enumerate(dtrajs) if length[c] >= threshold]

        n_clusters = max(np.max(x) for x in dtrajs)

        if v:
            if tooshort:
                print(
                    f"Trajectories {tooshort}\n" +
                    f"are shorter then step threshold (lag*minlenfactor = {threshold})\n" +
                    f"and will not be used to compute the MSM.\n"
                    )

            if empty:
                print(
                    f"Trajectories {empty}\n" + 
                    f"are empty and will not be used to to compute the MSM.\n"
                    )

            print(
                f"Using {len(dtrajs)} trajectories with {len(np.concatenate(dtrajs))} " +
                f"steps over {n_clusters} coresets\n" +
                f"---------------------------------------------------------\n" +
                f"*********************************************************\n"
                )

        # calculate milestone processes
        self.__Qminus = []
        self.__Qplus = []
        for dtraj in dtrajs:
            qminus, qplus = self.get_milestoning(dtraj)
            self.__Qminus.append(qminus)
            self.__Qplus.append(qplus)

        # calculate committer
        self.__F = []
        self.__B = []
        for qminus, qplus in zip(self.__Qminus, self.__Qplus):
            f, b = self.get_characteristicf(qminus, qplus, n_clusters)
            self.__F.append(f)
            self.__B.append(b)

        # calculate mass matrix
        self.__M = self.get_transitionmatrix(
            self.__F, self.__B, lag=0, n_clusters=n_clusters
            )

        self.__M[np.isnan(self.__M)] = 0

        # calculate transition matrix
        self.__T = self.get_transitionmatrix(
            self.__F, self.__B, lag=lag, n_clusters=n_clusters
            )

        self.__T[np.isnan(self.__M)] = 0
        
        # get largest connected set
        lcs = list(self.get_connected_sets(self.__T))
        self.__T = self.__T[tuple(np.meshgrid(lcs, lcs))].T

        # Weight T with the inverse M
        self.__T = np.dot(self.__T, np.linalg.inv(self.__M))

        if correct:
            self.correctnumerics()

    def diagonalise(self, T=None, **kwargs):
        if T is None:
            T = self.__T

        self.__eigenvalues, self.__eigenvalues_left, self.__eigenvalues_right = eig(
            T, left=True, right=True
        )

        self.__eigenvalues = np.sort(abs(self.__eigenvalues.real))
        self.__eigenvalues = self.__eigenvalues[::-1]

    def get_eigvalues(self, T=None, **kwargs):
        if T is None:
            T = self.__T

            self.__eigenvalues = eig(
                T, left=False, right=False
                )

            self.__eigenvalues = np.sort(abs(self.__eigenvalues.real))
            self.__eigenvalues = self.__eigenvalues[::-1]
        
        else:
            # TODO: Maybe this is confusing behaviour
            _eigenvalues = eig(
                T, left=False, right=False
                )

            _eigenvalues = np.sort(abs(_eigenvalues.real))
            _eigenvalues = _eigenvalues[::-1]

            return _eigenvalues


    def get_its(self, processes=None, purge=True):
        if purge:
            self.get_eigvalues()

        if processes is None:
            processes = len(self.__eigenvalues)

        self.__its = (-self.__tau / np.log(self.__eigenvalues[1:]))[:processes]
        self.__Its[self.tau] = self.__its


    def plot_its(self, its=None, ax=None, processes=None, ax_props=None, 
                 line_props=None, ref_props=None):
        if its is None:
            its = self.__Its

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if processes is None:
            processes = min([len(v) for v in its.values()])

        line_props_defaults = {
            }

        if line_props is not None:
            line_props_defaults.update(line_props)

        ref_props_defaults = {
            'linestyle': '--',
            'color': 'k',
            }

        if ref_props is not None:
            ref_props_defaults.update(ref_props)

        time = np.array([k for k in its]) * self.__step
        timescales = np.vstack([v for v in its.values()]) * self.__step
        lines = ax.plot(time, timescales, **line_props_defaults)
        ref = ax.plot(time, time, **ref_props_defaults)

       
        ax_props_defaults = {
            'xlabel': r"$\tau$ " + f" / {self.__unit}",
            'ylabel': f"its / {self.__unit}",
            'xlim': (time[0], time[-1]),
            }

        if ax_props is not None:
            ax_props_defaults.update(ax_props)

        ax.set(**ax_props_defaults)
        
        return (fig, ax, lines, ref)