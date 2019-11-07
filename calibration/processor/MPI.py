from itertools import groupby
import os
import re
import sys

from mpi4py import MPI


comm = MPI.COMM_WORLD


def distribute(path, comm=comm):
    """Distribute modules and sequences over multiple processes

    Jobs are first split over modules
    Minimum processes required is equal to number of modules in a run
    Further remaining processes are used to split over sequences for
    each module.
    Number of processes should be multiple of number or modules.

    Parameters
    ----------
        path: (str): Path to the run folder

    Return
    ------
        local_sequences: list of tuples (can be empty)
            Local sequeces that each process will process
            [(modno, Filesequence), ...]
    """

    index = None
    module_sequences = None
    if comm.rank == 0:
        pattern = f"(.+)AGIPD(.+)-S(.+).h5"
        try:
            sequences = [(
                re.match(pattern, f).group(2),
                os.path.join(path, f)) for f in os.listdir(path)
                if f.endswith('.h5') and re.match(pattern, f)]
        except Exception as ex:
            print(ex)
            sys.exit()

        module_sequences = []
        channels = []
        for key, group in groupby(sorted(sequences), lambda x: x[0]):
            module_sequences.append(list(group))
            channels.append(key)

        if comm.size % len(channels) != 0:
            print(f"Use multiple of {len(channels)} processes")
            sys.exit()

        index = comm.size // len(module_sequences)

    index = comm.bcast(index, root=0)
    # index is the number of processes available for each module
    module_sequences = comm.bcast(module_sequences, root=0)

    local_modno = comm.rank // index
    temp = comm.rank % index
    # index number of jobs for each module is distributed over sequences
    num_sequences = len(module_sequences[local_modno])
    sequence_cuts = [int(num_sequences * i / index) for i in range(index + 1)]

    chunks = list(zip(sequence_cuts[:-1], sequence_cuts[1:]))
    start, stop = chunks[temp][0], chunks[temp][1]

    local_sequences = module_sequences[local_modno][start:stop]

    return local_sequences


if __name__ == "__main__":

    path = "/gpfs/exfel/exp/MID/201901/p002542/raw/r0349"
    module_sequences = distribute(path)
    print("Module ", module_sequences)
