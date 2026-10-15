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

    :param parallel: Set to ``"mpi"`` to distribute the k-points of each line over the processes of ``comm``. Every process must run the same Z2Pack script; all of them receive the full result. Requires ``mpi4py``. See :ref:`z2pack_tutorial_mpi`.
    :type parallel: str

    :param comm: MPI communicator used when ``parallel="mpi"``. Defaults to ``MPI.COMM_WORLD``.
    :type comm: mpi4py.MPI.Comm

    :param eigensolver: Library used to diagonalize the Hamiltonian at each k-point: ``"lapack"`` (``scipy.linalg.eigh``) or ``"elsi"`` (ELSI through ``elsipy``, single-process mode).
    :type eigensolver: str
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
        parallel=None,
        comm=None,
        eigensolver="lapack",
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

        if parallel not in {None, "mpi"}:
            raise ValueError(
                "Invalid value '{}' for 'parallel', must be either None or 'mpi'.".format(parallel)
            )
        self._parallel = parallel
        if parallel == "mpi" and comm is None:
            from mpi4py import MPI  # pylint: disable=import-outside-toplevel

            comm = MPI.COMM_WORLD
        self._comm = comm

        if eigensolver not in {"lapack", "elsi"}:
            raise ValueError(
                "Invalid value '{}' for 'eigensolver', must be either 'lapack' or 'elsi'.".format(
                    eigensolver
                )
            )
        self._eigensolver = eigensolver
        self._size = size
        self._elsi_handle = None
        self._elsi_unit_ovlp = None

    @property
    def comm(self):
        """
        MPI communicator over which the k-points of a line are distributed,
        or ``None`` if the system is not parallelized.
        """
        return self._comm if self._parallel == "mpi" else None

    @comm.setter
    def comm(self, value):
        self._comm = value

    def get_eig(self, kpt):
        __doc__ = (  # pylint: disable=unused-variable,redefined-builtin
            super().__doc__  # pylint: disable=no-member
        )
        # create k-points for string
        k_points = kpt[:-1]

        # get eigenvectors corr. to the chosen bands
        if self._parallel == "mpi":
            eigs = self._get_eigvecs_mpi(k_points)
        else:
            eigs = [self._get_eigvecs(k) for k in k_points]

        for i, k in enumerate(kpt[:-1]):
            if self._convention == 2:
                # normalize phases to get u instead of phi
                eigs[i] *= np.exp(-2j * np.pi * np.dot(self._pos, k))[:, None]
            eigs[i] = list(eigs[i].T)

        # The last bloch state is the same as the first up to a phase factor
        eigs.append(
            list(eigs[0] * np.exp(-2j * np.pi * np.dot(self._pos, kpt[-1] - kpt[0]))[None, :])
        )
        return eigs

    def _get_eigvecs_mpi(self, k_points):
        """
        Computes the eigenvectors for a contiguous chunk of ``k_points`` on
        each process, and gathers the full list on all processes.
        """
        rank = self._comm.Get_rank()
        size = self._comm.Get_size()
        indices = np.array_split(np.arange(len(k_points)), size)[rank]
        local_eigs = []
        error = None
        try:
            local_eigs = [self._get_eigvecs(k_points[i]) for i in indices]
        except Exception as exc:  # pylint: disable=broad-except
            # re-raised on every process below, so that no process hangs
            error = exc
        gathered = self._comm.allgather((local_eigs, error))
        for _, err in gathered:
            if err is not None:
                raise err
        return [eig for chunk, _ in gathered for eig in chunk]

    def _get_eigvecs(self, k):
        """
        Returns the eigenvectors (as columns) of the selected bands at ``k``.
        """
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

        eigval, eigvec = self._eigh(ham)
        eigval = np.real(eigval)
        idx = eigval.argsort()

        idx = idx[self._bands]
        idx.sort()
        # take only the lower - energy eigenstates
        eigvec = eigvec[:, idx]

        # cast to complex explicitly to avoid casting error when the phase
        # is complex but the eigenvector itself is not.
        return np.array(eigvec, dtype=complex)

    def _eigh(self, ham):
        if self._eigensolver == "elsi":
            return self._elsi_eigh(ham)
        return la.eigh(ham)

    def _elsi_eigh(self, ham):
        """
        Diagonalizes ``ham`` with ELSI in single-process mode. The overlap is
        always the identity, since non-orthogonal bases are already
        orthogonalized before this is called.
        """
        if self._elsi_handle is None:
            from elsipy import elsi  # pylint: disable=import-outside-toplevel

            self._elsi_handle = elsi.elsi(
                parallel_mode=0,
                matrix_format=0,
                n_basis=self._size,
                n_electron=float(self._size),
                n_state=self._size,
            )
            self._elsi_handle.elsi_set_unit_ovlp(1)
            self._elsi_unit_ovlp = np.eye(self._size, dtype=complex, order="F")
        # ELSI is Fortran: the matrix must be column-major, and it may be
        # overwritten, so always pass a fresh copy.
        ham = np.array(ham, dtype=complex, order="F", copy=True)
        return self._elsi_handle.elsi_ev_complex(ham, self._elsi_unit_ovlp)
