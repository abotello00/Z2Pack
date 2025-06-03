#!/usr/bin/env python
"""
This module contains a class for creating Systems which are described by a Hamiltonian matrix (hm), such as k•p models.
"""

import itertools
import numbers

import numpy as np
import scipy.linalg as la

from .system import EigenstateSystem

__all__ = ["System"]


class System(EigenstateSystem):
    r"""
    This class is used when the system can be explicitly described as a matrix Hamiltonian :math:`\mathcal{H}(\mathbf{k})`.

    :param hamilton: A function taking the wavevector ``k`` (``list`` of length 3) as an input and returning the matrix Hamiltonian.
    :type hamilton: collections.abc.Callable

    :param dim:     Dimension of the system.
    :type dim:      int

    :param pos: Positions of the orbitals w.r.t the reduced unit cell.
        Per default, all orbitals are put at the origin.
    :type pos: list

    :param bands: Specifies either the number of occupied bands (if it is an integer) or which bands should be taken into consideration (if it is a list of indices). If no value is given, half the given bands are considered.
    :type bands: :py:class:`int` or :py:class:`list`

    :param hermitian_tol:   Maximum absolute value in the difference between the Hamiltonian and its hermitian conjugate. Use ``hermitian_tol=None`` to deactivate the test entirely.
    :type hermitian_tol:    float

    :param basis_overlap: A function taking the wavevector ``k`` (``list`` of length 3) as an input and returning the overlap matrix between the basis vectors w.r.t which the Hamiltonian is defined. If no value is given, the basis is assumed to be orthonormal.
    :type basis_overlap: collections.abc.Callable

    :param convention: The convention used for the Hamiltonian, following the `pythtb formalism <http://www.physics.rutgers.edu/pythtb/_downloads/pythtb-formalism.pdf>`_. Convention 1 means that the eigenvalues of :math:`\mathcal{H}(\mathbf{k})` are wave vectors :math:`\left|\psi_{n\mathbf{k}}\right>`. With convention 2, they are the cell-periodic Bloch functions :math:`\left|u_{n\mathbf{k}}\right>`.
    :type convention: int

    :param check_periodic: Evaluate the Hamiltonian at :math:`\{0, 1\}^d` as a simple check if it is periodic. Note that this does not work if the Hamiltonian is written such that the eigenstates acquire a phase when being translated by a lattice vector.
    :type check_periodic: bool
    """

    def __init__(
        self,
        hamilton,
        *,
        dim=3,
        pos=None,
        bands=None,
        hermitian_tol=1e-6,
        basis_overlap=None,
        convention=2,
        check_periodic=False,
    ):
        self._hamilton = hamilton
        self._hermitian_tol = hermitian_tol
        self._basis_overlap = basis_overlap
        self._convention = int(convention)
        if self._convention not in {1, 2}:
            raise ValueError(
                "Invalid value '{}' for 'convention', must be either 1 or 2.".format(
                    self._convention
                )
            )
        self._hamilton_orthogonal = basis_overlap is None

        if check_periodic:
            k_values = itertools.product([0, 1], repeat=dim)
            k_first = next(k_values)
            ham_first = self._hamilton(k_first)
            for k in k_values:
                if not np.allclose(ham_first, self._hamilton(k)):
                    raise ValueError(
                        "The given Hamiltonian is not periodic: H(k={}) != H(k={})".format(
                            k_first, k
                        )
                    )

        size = len(self._hamilton([0] * dim))  # assuming to be square...
        if not self._hamilton_orthogonal:
            size_S = len(  # pylint: disable=invalid-name
                self._basis_overlap([0] * dim)
            )  # assuming to be square...
            if size_S != size:
                raise ValueError(
                    "The dimensions of overlap matrix ({}) and Hamilontonian ({}) do not match.".format(
                        size_S, size
                    )
                )

        # add one atom for each orbital in the hamiltonian
        if pos is None:
            self._pos = [np.zeros(dim) for _ in range(size)]
        else:
            if len(pos) != size:
                raise ValueError(
                    "The number of positions ({}) does not match the size of the Hamiltonian ({}).".format(
                        len(pos), size
                    )
                )
            self._pos = [np.array(p) for p in pos]
        if bands is None:
            bands = size // 2
        if isinstance(bands, numbers.Integral):
            self._bands = list(range(bands))
        else:
            self._bands = bands

    def get_eig_serial(self, kpt):
        __doc__ = (  # pylint: disable=unused-variable,redefined-builtin
            super().__doc__  # pylint: disable=no-member
        )
        # create k-points for string
        k_points = kpt[:-1]

        # get eigenvectors corr. to the chosen bands
        eigs = []
        for i, k in enumerate(k_points):
            ham = self._hamilton(k)
            if self._hermitian_tol is not None:
                diff = la.norm(ham - ham.conjugate().transpose(), ord=np.inf)
                if diff > self._hermitian_tol:
                    raise ValueError(
                        "The Hamiltonian you used is not hermitian, with the maximum difference between the Hamiltonian and its adjoint being {}. Use the ``hamilton_tol`` input parameter (in the ``tb.Hamilton`` constructor; currently {}) to set the sensitivity of this test or turn it off completely (``hamilton_tol=None``).".format(
                            diff, self._hermitian_tol
                        )
                    )
            if not self._hamilton_orthogonal:
                ovl = self._basis_overlap(k)
                if self._hermitian_tol is not None:
                    diff = la.norm(ovl - ovl.conjugate().transpose(), ord=np.inf)
                    if diff > self._hermitian_tol:
                        raise ValueError(
                            "The overlap you used is not hermitian, with the maximum difference between the overlap matrix and its adjoint being {}. Use the ``hermitian_tol`` input parameter (currently {}) to set the sensitivity of this test or turn it off completely (``hermitian_tol=None``).".format(
                                diff, self._hermitian_tol
                            )
                        )
                ovl2 = la.inv(la.sqrtm(ovl))
                ortho_ham = np.dot(ovl2, ham.dot(ovl2))
                ham = ortho_ham

            eigval, eigvec = la.eigh(ham)

            phase = np.angle(eigvec[0, np.arange(eigvec.shape[1])])
            phase_adjustment = np.exp(-1j * phase)
            eigvec *= phase_adjustment


            eigval = np.real(eigval)
            idx = eigval.argsort()

            idx = idx[self._bands]
            idx.sort()
            # take only the lower - energy eigenstates
            eigvec = eigvec[:, idx]
            if  i == 0 :
                print(eigval[-5:])
                #print(norms[-5:])
            # cast to complex explicitly to avoid casting error when the phase
            # is complex but the eigenvector itself is not.
            eigs.append(np.array(eigvec, dtype=complex))

        for i, k in enumerate(kpt[:-1]):
            if self._convention == 2:
                # normalize phases to get u instead of phi
                eigs[i] *= np.exp(-2j * np.pi * np.dot(self._pos, k))[:, None]
            eigs[i] = list(eigs[i].T)

        # The last bloch state is the same as the first up to a phase factor
        eigs.append(
            list(eigs[0] * np.exp(-2j * np.pi * np.dot(self._pos, kpt[-1] - kpt[0]))[None, :])
        )
        print(np.array(eigs[0][:3])[:,-5:])
        return eigs

    def get_eig_mpi(self, kpt):
        __doc__ = (  # pylint: disable=unused-variable,redefined-builtin
            super().__doc__  # pylint: disable=no-member
        )
        from elsipy import elsi
        from mpi4py import MPI
        # ==== MPI Setup ====
        comm=MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # ==== ELSI Setup ====
        n_basis = len(self._hamilton([0,0,0]))
        n_electrons = n_basis
        elsi_h = elsi.elsi(parallel_mode=0,matrix_format=0,  n_basis=n_basis,n_electron=n_electrons,n_state=n_basis)
        elsi_h.elsi_set_mpi(comm)
        # create k-points for string
        k_points = kpt[:-1]


        # split to local k-points
        k_points_local = np.array_split(k_points, size)[rank]
        n_kpoints_local = len(k_points_local)
        eigs_local = [] #np.empty((n_basis,n_basis,1), dtype = 'complex')

        for i, k in enumerate(k_points_local):
            ham = np.array(self._hamilton(k), dtype='complex128')
            if self._hermitian_tol is not None:
                diff = la.norm(ham - ham.conjugate().transpose(), ord=np.inf)
                if diff > self._hermitian_tol:
                    raise ValueError(
                        "The Hamiltonian you used is not hermitian, with the maximum difference between the Hamiltonian and its adjoint being {}. Use the ``hamilton_tol`` input parameter (in the ``tb.Hamilton`` constructor; currently {}) to set the sensitivity of this test or turn it off completely (``hamilton_tol=None``).".format(
                            diff, self._hermitian_tol
                        )
                    )
            ovl = np.identity(n_basis, dtype='complex')
            elsi_h.elsi_set_unit_ovl(1)
            if not self._hamilton_orthogonal:
                ovl = np.array(self._basis_overlap(k), dtype='complex128')
            #    if self._hermitian_tol is not None:
            #        diff = la.norm(ovl - ovl.conjugate().transpose(), ord=np.inf)
            #        if diff > self._hermitian_tol:
            #            raise ValueError(
            #                "The overlap you used is not hermitian, with the maximum difference between the overlap matrix and its adjoint being {}. Use the ``hermitian_tol`` input parameter (currently {}) to set the sensitivity of this test or turn it off completely (``hermitian_tol=None``).".format(
            #                    diff, self._hermitian_tol
            #                )
            #            )
                ovl2 = la.inv(la.sqrtm(ovl))
                ortho_ham = np.dot(ovl2, ham.dot(ovl2))
                ham = ortho_ham
                ovl =  np.identity(n_basis, dtype='complex')
                elsi_h.elsi_set_unit_ovl(1)

            eigval, eigvec =  elsi_h.elsi_ev_complex(ham,ovl)
            eigvec = eigvec.T # fortran to python column arrangement
            phase = np.angle(eigvec[0, np.arange(eigvec.shape[1])])
            phase_adjustment = np.exp(-1j * phase)
            eigvec *= phase_adjustment

            eigval = np.real(eigval)
            idx = eigval.argsort()




            idx = idx[self._bands]
            idx.sort()
            # take only the lower - energy eigenstates

            eigvec = eigvec[:, idx]
            #if rank ==0 and  i == 0 :
                #print(eigval[-5:])
                #print(norms[-5:])

            # cast to complex explicitly to avoid casting error when the phase
            # is complex but the eigenvector itself is not.
            eigs_local.append(np.array(eigvec, dtype='complex128'))


        eigs=[]

        # gather:
        ## Send the results to the root process
        if rank == 0 :
            eigs = eigs_local
            # Receive the results on the root process
            for i in range(1, size):
                #start = i * n_kpoints_local
                #end = start + n_kpoints_local
                loc = comm.recv(source=i, tag=11)
                for l in loc:
                    eigs.append(l)
        else:
            comm.send(eigs_local, dest=0, tag=11)

        comm.Barrier()

        eigs = comm.bcast(eigs, root=0)

        for i, k in enumerate(kpt[:-1]):
            if self._convention == 2:
                # normalize phases to get u instead of phi
                eigs[i] *= np.exp(-2j * np.pi * np.dot(self._pos, k))[:, None]
            eigs[i]  = list(eigs[i].T)

        # The last bloch state is the same as the first up to a phase factor
        eigs.append(
            list(eigs[0] * np.exp(-2j * np.pi * np.dot(self._pos, kpt[-1] - kpt[0]))[None, :]
        ))

        #if rank == 0 :
            #print(np.array(eigs[0][:3])[:,-5:])
        return eigs

    def get_eig_mpi_par_orb(self, kpt):
        __doc__ = (  # pylint: disable=unused-variable,redefined-builtin
            super().__doc__  # pylint: disable=no-member
        )
        from elsipy import elsi,blacs
        from mpi4py import MPI
        # ==== MPI Setup ====

        comm=MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        npcol = 8
        nprow = size//npcol
        if nprow == 0:
            nprow = 1
        blacsctxt=blacs.sys2blacs_handle(comm)
        blacs.gridinit(blacsctxt,nprow,npcol)


        # ==== ELSI Setup ====
        n_basis = len(self._hamilton([0,0,0]))
        n_electrons = n_basis

        elsi_h = elsi.elsi(parallel_mode=1,matrix_format=0,  n_basis=n_basis,n_electron=n_electrons,n_state=n_basis)
        elsi_h.elsi_set_mpi(comm)
        elsi_h.elsi_set_blacs(blacsctxt,8)
        #elsi_h.elsi_set_mpi_global(comm)

        # create k-points for string
        k_points = kpt[:-1]


        # split to local k-points
        #k_points_local = np.array_split(k_points, size)[rank]
        #n_kpoints_local = len(k_points_local)
        eigs= [] #np.empty((n_basis,n_basis,1), dtype = 'complex')

        for i, k in enumerate(k_points):
            ham = np.array(self._hamilton(k), dtype='complex128')
            if self._hermitian_tol is not None:
                diff = la.norm(ham - ham.conjugate().transpose(), ord=np.inf)
                if diff > self._hermitian_tol:
                    raise ValueError(
                        "The Hamiltonian you used is not hermitian, with the maximum difference between the Hamiltonian and its adjoint being {}. Use the ``hamilton_tol`` input parameter (in the ``tb.Hamilton`` constructor; currently {}) to set the sensitivity of this test or turn it off completely (``hamilton_tol=None``).".format(
                            diff, self._hermitian_tol
                        )
                    )
            ovl = np.identity(n_basis, dtype='complex')
            elsi_h.elsi_set_unit_ovl(1)
            if not self._hamilton_orthogonal:
                ovl = np.array(self._basis_overlap(k), dtype='complex128')
            #    if self._hermitian_tol is not None:
            #        diff = la.norm(ovl - ovl.conjugate().transpose(), ord=np.inf)
            #        if diff > self._hermitian_tol:
            #            raise ValueError(
            #                "The overlap you used is not hermitian, with the maximum difference between the overlap matrix and its adjoint being {}. Use the ``hermitian_tol`` input parameter (currently {}) to set the sensitivity of this test or turn it off completely (``hermitian_tol=None``).".format(
            #                    diff, self._hermitian_tol
            #                )
            #            )
                ovl2 = la.inv(la.sqrtm(ovl))
                ortho_ham = np.dot(ovl2, ham.dot(ovl2))
                ham = ortho_ham
                ovl =  np.identity(n_basis, dtype='complex')
                elsi_h.elsi_set_unit_ovl(1)

            eigval, eigvec =  elsi_h.elsi_ev_complex(ham,ovl)
            comm.barrier()
            eigvec = eigvec.T # fortran to python column arrangement
            phase = np.angle(eigvec[0, np.arange(eigvec.shape[1])])
            phase_adjustment = np.exp(-1j * phase)
            eigvec *= phase_adjustment

            eigval = np.real(eigval)
            idx = eigval.argsort()

            idx = idx[self._bands]
            idx.sort()
            # take only the lower - energy eigenstates

            eigvec = eigvec[:, idx]
            if rank ==0 and  i == 0 :
                print(eigval[-5:])
                #print(norms[-5:])

            # cast to complex explicitly to avoid casting error when the phase
            # is complex but the eigenvector itself is not.
            eigs.append(np.array(eigvec, dtype='complex128'))




        for i, k in enumerate(kpt[:-1]):
            if self._convention == 2:
                # normalize phases to get u instead of phi
                eigs[i] *= np.exp(-2j * np.pi * np.dot(self._pos, k))[:, None]
            eigs[i]  = list(eigs[i].T)

        # The last bloch state is the same as the first up to a phase factor
        eigs.append(
            list(eigs[0] * np.exp(-2j * np.pi * np.dot(self._pos, kpt[-1] - kpt[0]))[None, :]
        ))

        if rank == 0 :
            print(np.array(eigs[0][:3])[:,-5:])
        return eigs

    def get_eig(self, kpt, mode='serial'):
        if mode=='serial':
            return self.get_eig_serial(kpt)
        elif mode == 'mpi':
            return self.get_eig_mpi(kpt)
        elif mode == 'mpi_par_orb':
            return self.get_eig_mpi_par_orb(kpt)
        else:
            print('Invalid mode option, defaulting to serial mode')
            return self.get_eig_serial(kpt)
