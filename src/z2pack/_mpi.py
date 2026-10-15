"""
Helpers for running calculations on several MPI processes (via ``mpi4py``).

All processes execute the same Z2Pack script and end up with the same result.
Only the root process (rank 0) writes the log and the ``save_file``.
"""

import contextlib
import logging

__all__ = []


def _get_comm(comm, system):
    """
    Returns the communicator of a run: the explicitly given ``comm``, or the
    one the system uses for its own parallelization (``system.comm``).
    """
    if comm is None:
        comm = getattr(system, "comm", None)
    return comm


def _is_root(comm):
    return comm is None or comm.Get_rank() == 0


@contextlib.contextmanager
def _root_only_logging(comm):
    """Silences the Z2Pack log on all processes except the root."""
    if _is_root(comm):
        yield
        return
    logger = logging.getLogger("z2pack")
    old_level = logger.level
    logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        logger.setLevel(old_level)


def _default_line_groups(comm, system):
    """
    If the system distributes its own work (k-points) over ``comm``, all
    processes compute each line together. Otherwise, every process computes
    lines on its own.
    """
    if getattr(system, "comm", None) is not None:
        return 1
    return comm.Get_size()


class _LineGroups:
    """
    Splits the processes of ``comm`` into ``num_groups`` groups of (almost)
    equal size, each of which computes whole lines. Within a group, a system
    with its own MPI parallelization (``system.comm``) uses the group's
    communicator.

    :param comm: The communicator of the run, or ``None`` for a serial run.
    :param num_groups: Number of groups, between 1 and ``comm.Get_size()``.
    :param system: The system of the run.
    """

    def __init__(self, comm, num_groups, system):
        self.comm = comm
        self.system = system
        if comm is None:
            self.num_groups = 1
            self.group = 0
            self.group_comm = None
            return

        size = comm.Get_size()
        if num_groups is None:
            num_groups = _default_line_groups(comm, system)
        if not 1 <= num_groups <= size:
            raise ValueError(
                "Invalid value '{}' for 'line_groups', must be between 1 and the number of MPI processes ({}).".format(
                    num_groups, size
                )
            )
        self.num_groups = num_groups
        # contiguous blocks of ranks, so that a group tends to share a node
        self.group = comm.Get_rank() * num_groups // size
        self.group_comm = comm.Split(self.group, comm.Get_rank())

    @contextlib.contextmanager
    def system_comm(self):
        """Lets the system parallelize over the group instead of ``comm``."""
        old_comm = getattr(self.system, "comm", None)
        if old_comm is None or self.group_comm is None:
            yield
            return
        self.system.comm = self.group_comm
        try:
            yield
        finally:
            self.system.comm = old_comm

    def map(self, fct, args):
        """
        Computes ``[fct(a) for a in args]``, distributing the elements of
        ``args`` round-robin over the groups. Every process receives the full
        list. An exception raised on any process is re-raised on all of them.
        """
        args = list(args)
        if self.comm is None:
            return [fct(a) for a in args]

        local = []
        error = None
        try:
            with self.system_comm():
                for i, arg in enumerate(args):
                    if i % self.num_groups == self.group:
                        local.append((i, fct(arg)))
        except Exception as exc:  # pylint: disable=broad-except
            error = exc
        # only one process per group needs to send the results
        send = local if self.group_comm.Get_rank() == 0 else []
        gathered = self.comm.allgather((send, error))
        for _, err in gathered:
            if err is not None:
                raise err
        results = [None] * len(args)
        for chunk, _ in gathered:
            for i, res in chunk:
                results[i] = res
        return results

    def free(self):
        if self.group_comm is not None:
            self.group_comm.Free()
            self.group_comm = None
