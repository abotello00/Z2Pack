.. _z2pack_tutorial_mpi :

Running in Parallel with MPI
============================

Calculations for systems described by a matrix Hamiltonian (:class:`z2pack.hm.System`) can be distributed over several MPI processes, using `mpi4py <https://mpi4py.readthedocs.io>`_ (``pip install z2pack[mpi]``). Every process runs the same Python script, and every process receives the full result. Only the root process (rank 0) writes the log and the ``save_file``.

There are two levels of parallelism, which can be combined:

* **Lines**: the lines of a surface which are added at the same time -- the initial ``num_lines`` lines, and the new lines of each refinement step -- are independent of each other. They are distributed over groups of processes.
* **k-points**: the k-points of a single line are distributed over the processes of a group. Each process diagonalizes the Hamiltonian at its own k-points.

The result does not depend on how the work is distributed: it is identical to a serial calculation.

Parallelizing over lines
------------------------

Pass the communicator to :func:`z2pack.surface.run` (or :func:`z2pack.volume.run`). By default, every process then computes lines on its own:

.. code :: python

    from mpi4py import MPI
    import z2pack

    system = z2pack.hm.System(hamilton, bands=...)
    result = z2pack.surface.run(
        system=system,
        surface=lambda t1, t2: [t1, t2, 0],
        comm=MPI.COMM_WORLD,
    )

.. code :: bash

    mpirun -n 8 python my_script.py

Lines are computed in batches: first ``num_lines`` lines, then one line per non-converged gap between neighbouring lines. A batch with fewer lines than groups leaves some groups idle, so choose ``num_lines`` as a multiple of the number of groups, for example ``num_lines=16`` with 16 processes. The later refinement batches are usually smaller, which is where k-point parallelism helps.

For :func:`z2pack.volume.run`, the surfaces are computed one after the other, and the lines within each surface are distributed as described here.

Parallelizing over k-points
---------------------------

Set ``parallel="mpi"`` when creating the system. The communicator of the system (``MPI.COMM_WORLD`` by default, or the ``comm`` argument) is then also used by the run functions, so that all processes compute each line together:

.. code :: python

    system = z2pack.hm.System(hamilton, bands=..., parallel="mpi")
    result = z2pack.surface.run(system=system, surface=lambda t1, t2: [t1, t2, 0])

A line with :math:`N` k-points (the values yielded by ``iterator``) requires :math:`N - 1` diagonalizations, because the last k-point is equivalent to the first. They are split into contiguous chunks over the :math:`P` processes of a group, and the slowest process determines the time. For an even load, :math:`N - 1` should be a multiple of :math:`P`, so the ``iterator`` should start at :math:`mP + 1` and increase in steps of :math:`P`:

.. code :: python

    P = 8
    result = z2pack.surface.run(
        system=system,
        surface=lambda t1, t2: [t1, t2, 0],
        iterator=range(P + 1, 5 * P + 2, P),  # N = 9, 17, 25, 33, 41
    )

Each step of the iterator recomputes the whole line, so the step size only controls how finely the convergence in :math:`N` is resolved. As long as :math:`N - 1 < P`, some processes are idle: with the default iterator (starting at :math:`N = 8`), k-point parallelism alone is efficient only up to about 8 processes. For more processes, combine it with the parallelization over lines.

Combining both
--------------

The ``line_groups`` argument splits the processes into groups of (almost) equal size, made of consecutive ranks. Each group computes different lines, and the processes of a group share the k-points of each line:

.. code :: python

    # 32 processes: 4 groups of 8, each computing lines with 8 processes
    system = z2pack.hm.System(hamilton, bands=..., parallel="mpi")
    result = z2pack.surface.run(
        system=system,
        surface=lambda t1, t2: [t1, t2, 0],
        line_groups=4,
        num_lines=12,
        iterator=range(9, 42, 8),
    )

The default is ``line_groups=1`` if the system is parallelized itself (``parallel="mpi"``), and one group per process otherwise.

As a rule of thumb: parallelize over lines first, since lines need no communication during the calculation. Add k-point parallelism when there are more processes than lines in a batch, or when a single line is too slow.

Eigensolver
-----------

By default, the Hamiltonian is diagonalized with LAPACK (``scipy.linalg.eigh``). With ``eigensolver="elsi"``, `ELSI <https://wordpress.elsi-interchange.org>`_ is used instead, through its Python bindings ``elsipy``. These are not on PyPI; they are built from ``test/python/elsipy`` in the ELSI source (see the README there). ELSI is used in single-process mode: each process diagonalizes the full Hamiltonian of its own k-points.

Notes
-----

* All processes must call the run function with the same arguments.
* An error on any process (for example a non-hermitian Hamiltonian) is raised on all processes, so that the calculation does not hang.
* With ``load=True``, every process reads the ``save_file``, which must therefore be on a file system shared by all processes.
* Each process keeps the full result in memory, including the eigenstates of all lines. These are also what is communicated between processes (about :math:`N \times` ``bands`` :math:`\times` the matrix size complex numbers per line), which can become significant for large Hamiltonians.
