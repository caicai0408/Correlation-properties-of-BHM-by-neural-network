# -*- coding: utf-8 -*
import numpy as np
from scipy.special import binom 
import scipy.sparse as sparse


class Model:
    # omegas : Onsite energies
    # links :  Each hopping is of the form [from, to, amplitude].
    # U : Interaction strength
    def __init__(self, omegas, links, U):
        self.omegas = np.array(omegas) - U/2
        self.links = links
        self.U = U
        self.n = len(omegas)

    @property
    def hopping(self):
        # The single particle hopping Hamiltonian
        H0 = np.zeros([self.n]*2)  

        for link in self.links:
            H0[link[0], link[1]] = link[2] if len(link) > 2 else -1

        return H0 + H0.T  

    def numbersector(self, nb):
        # Returns a specific particle number sector object based on this model.
        return NumberSector(self.n, nb, model=self)


class NumberSector:
    def __init__(self, N, nb, model=None):
        self.N = N
        self.nb = nb
        self.basis = Basis(N, nb)
        if model is not None:
            self.model = model
    @property
    def hamiltonian(self):
        if not hasattr(self, '_hamiltonian') or self._hamiltonian == False:
            self._hamiltonian = NumberSector.generate_hamiltonian(self.model, self.basis)

        return self._hamiltonian
    @staticmethod
    def generate_hamiltonian(m, basis):
        #Generates the (sparse) Hamiltonian
        nbas = basis.len

        HDi = np.arange(nbas)
        HD = NumberSector.onsite_hamiltonian(m.omegas, basis.vs) \
            + NumberSector.interaction_hamiltonian(m.U, basis.vs)
        Hki, Hkj, Hkv = NumberSector.hopping_hamiltonian(basis, m.hopping, basis.vs)

        # return sparse.coo_matrix((Hkv, (Hki, Hkj)), shape=(nbas, nbas)).tocsr() \
        #     + sparse.coo_matrix((HD, (HDi, HDi)), shape=(nbas, nbas)).tocsr()
        # "\" mean line continue
        return sparse.coo_matrix((Hkv, (Hki, Hkj)), shape=(nbas, nbas)).toarray() \
            + sparse.coo_matrix((HD, (HDi, HDi)), shape=(nbas, nbas)).toarray()
    @staticmethod
    def onsite_hamiltonian(omegas, states):
        #Onsite hamiltonian
        return states.dot(omegas)
    @staticmethod
    def hopping_hamiltonian(basis, H0, states):
        #Hopping Hamiltonian expressed in the many-particle basis
        H1s, H2s, Hvs = [], [], []

        for i in range(H0.shape[0]):
            js = np.nonzero(states[:, i])[0]  # affected states
            nj = len(js)

            ls = np.nonzero(H0[i, :])[0]  # relevant hoppings
            nl = len(ls)

            ks = np.zeros((nj*nl,))  # storing result states
            vs = np.zeros((nj*nl,))  # storing result states

            for k, l in enumerate(ls):  ##enumerate
                nstates = states[js, :]
                nstates[:, i] -= 1  # remove one element
                nstates[:, l] += 1  # add it here

                ks[k*nj:(k+1)*nj] = basis.index(nstates)  # the new states
                vs[k*nj:(k+1)*nj] = H0[i, l]*np.sqrt(states[js, i]*(states[js, l] + 1))

            H1s += np.tile(js, nl).tolist()
            H2s += ks.tolist()
            Hvs += vs.tolist()

        return H1s, H2s, Hvs
    @staticmethod
    def interaction_hamiltonian(U, states):
        #interaction Hamiltonian
        return np.sum(U/2*states**2, axis=1)


class Basis:
    # Many-body basis of specific <nb> charge state on a lattice with <N> sites.
    def __init__(self, N, nb):
        self.N = N  # number of sites
        self.nb = nb  # number of bosons

        self.len = Basis.size(N, nb)
        self.vs = Basis.generate(N, nb)
        self.hashes = Basis.hash(self.vs)
        self.sorter = Basis.argsort(self.hashes)

    def index(self, state):
        return Basis.stateindex(state, self.hashes, self.sorter)
    @staticmethod
    def size(N, nb):
        return int(binom(nb+N-1, nb))
    @staticmethod
    def stateindex(state, hashes, sorter):
        key = Basis.hash(state)
        return sorter[np.searchsorted(hashes, key, sorter=sorter)]
    @staticmethod
    def generate(N, nb):
        states = np.zeros((Basis.size(N, nb), N), dtype=int)

        states[0, 0] = nb
        ni = 0  # init
        for i in range(1, states.shape[0]):

            states[i, :N-1] = states[i-1, :N-1]
            states[i, ni] -= 1
            states[i, ni+1] += 1 + states[i-1, N-1]

            if ni >= N-2:
                if np.any(states[i, :N-1]):
                    ni = np.nonzero(states[i, :N-1])[0][-1]
            else:
                ni += 1

        return states
    @staticmethod
    def hash(states):
        n = states.shape[1] if len(states.shape) > 1 else len(states)
        ps = np.sqrt(lowest_primes(n))
        return states.dot(ps)
    @staticmethod
    def argsort(hashes):
        return np.argsort(hashes, 0, 'quicksort')


def lowest_primes(n):
    return primes(n**2)[:n]


def primes(upto):
    primes = np.arange(3, upto+1, 2)

    isprime = np.ones((upto-1)/2, dtype=bool)

    for factor in primes[:int(np.sqrt(upto))]:

        if isprime[(factor-2)/2]:
            isprime[(factor*3-2)/2::factor] = 0

    return np.insert(primes[isprime], 0, 2)