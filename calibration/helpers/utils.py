"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from functools import wraps
from glob import iglob
from itertools import chain
import os.path as osp
import psutil as ps


def timeit(original):
    import time
    @wraps(original)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        print(f"Time to evaluate {original.__name__} with args {args}"
              f" : {time.perf_counter() - t0} secs")
        return result
    return wrapper


def pulse_filter(pulse_ids, data_counts):

    pulse_ids_intrain = parse_ids(pulse_ids)

    if pulse_ids_intrain == [-1]:
        pulse_ids_intrain = list(range(data_counts.iloc[0]))
    indices = [list(map(lambda x: x + idx * n_pulse, pulse_ids_intrain))
               for idx, n_pulse in enumerate(data_counts, start=0)]

    return indices

# list(chain(*indices))

def parse_le(text):
    if ":" not in text:
        raise ValueError("Input is incomprehensible")
    elif text == ":":
        return [-1]
    else:
        indices = text.split(":")
        if len(indices) < 0 or len(indices) > 2:
            raise ValueError("Input is incomprehensible")

        try:
            start = int(indices[0].strip())
            stop = int(indices[1].strip())
        except Exception as ex:
            raise ValueError("Cannot convert inputs to integers")

        if start < 0 or stop < 0:
            raise ValueError("Indices cannot be less than 0")

        return start, stop


def parse_ids(text):

    """ Method to parse pulse ids in KaraboFAI.

    Parse a string into a list of integers.

    :param str text: the input string.

    :return list: a list of IDs which are integers.

    :raise ValueError
    """
    def parse_item(v):
        if not v:
            return []

        if v.strip() == ':':
            return [-1]

        if ':' in v:
            try:
                x = v.split(':')
                if len(x) < 2 or len(x) > 3:
                    raise ValueError("The input is incomprehensible!")

                start = int(x[0].strip())
                if start < 0:
                    raise ValueError("Pulse index cannot be negative!")
                end = int(x[1].strip())

                if len(x) == 3:
                    inc = int(x[2].strip())
                    if inc <= 0:
                        raise ValueError(
                            "Increment must be a positive integer!")
                else:
                    inc = 1

                return list(range(start, end, inc))

            except Exception as e:
                raise ValueError("Invalid input: " + repr(e))

        else:
            try:
                v = int(v)
                if v < 0:
                    raise ValueError("Pulse index cannot be negative!")
            except Exception as e:
                raise ValueError("Invalid input: " + repr(e))

        return v

    ret = set()
    # first split string by comma, then parse them separately
    for item in text.split(","):
        item = parse_item(item.strip())
        if isinstance(item, int):
            ret.add(item)
        else:
            ret.update(item)

    return sorted(ret)


def get_virtual_memory():
    virtual_memory, swap_memory = ps.virtual_memory(), ps.swap_memory()
    return virtual_memory, swap_memory


def find_proposal(proposal, run, data='raw'):
    """Access EuXFEL data on the Maxwell cluster by proposal and run number.
    Parameters
    ----------
    proposal: str, int
        A proposal number, such as 2012, '2012', 'p002012', or a path such as
        '/gpfs/exfel/exp/SPB/201701/p002012'.
    run: str, int
        A run number such as 243, '243' or 'r0243'.
    data: str
        'raw' or 'proc' (processed) to access data from one of those folders.
        The default is 'raw'.

    Return:
    -------
    proposal_path: str
    """
    DATA_ROOT_DIR = "/gpfs/exfel/exp"
    def find_dir(propno):
        """Find the proposal directory for a given proposal on Maxwell"""
        if '/' in propno:
            # Already passed a proposal directory
            return propno

        for d in iglob(osp.join(DATA_ROOT_DIR, '*/*/{}'.format(propno))):
            return d

        raise Exception("Couldn't find proposal dir for {!r}".format(propno))

    if isinstance(proposal, int):
        proposal = 'p{:06d}'.format(proposal)
    elif ('/' not in proposal) and not proposal.startswith('p'):
        proposal = 'p' + proposal.rjust(6, '0')

    prop_dir = find_dir(proposal)

    if isinstance(run, int):
        run = 'r{:04d}'.format(run)
    elif not run.startswith('r'):
        run = 'r' + run.rjust(4, '0')

    return osp.join(prop_dir, data, run)