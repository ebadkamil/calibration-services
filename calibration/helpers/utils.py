"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from functools import wraps
from glob import iglob
import h5py
from itertools import chain
import numpy as np
import os.path as osp
import psutil as ps
import xarray as xr


def timeit(name):
    def profile(original):
        import time
        @wraps(original)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = original(*args, **kwargs)
            print(f"Time to evaluate {name} "
                  f" : {time.perf_counter() - t0} secs")
            return result
        return wrapper
    return profile


def timeit_class(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


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


def dark2file(filename, dic):
    """
    Save a dark data dictionary into an h5 file.
    Parameters:
    -----------
    filename: str
        A filename where to store the data
    dic: dictionary
        A dictionary with module numbers as keys and dark data as values

    Return:
    -------

    """
    with h5py.File(filename, 'w') as f:
        g = f.create_group(f'entry_1/FXE_DET_LPD1M-1')
        for modno, data in dic.items():
            if data is not None:
                h = g.create_group(f'module_{modno:02}')
                h.create_dataset('data', data=data)


def file2dark(filename, modules):
    """
    Retrieve dark data from an h5 file as a dictionary
    Parameters:
    ----------
    filename: str
       An h5 file with dark data
    modules: list, int
       A list of modules to put data in

    Return:
    -------
    dic: dictionary
        A dictionary with module numbers as keys and dark data as values
    """
    file_ = h5py.File(filename, 'r')
    dic = {}
    for modno in modules:
        path = f'entry_1/FXE_DET_LPD1M-1/module_{modno:02}/data'

    dic[modno] = file_[path][:]

    return dic


def intensity2file(filename, dic, digitizer=False):
    """
    Save a ROI intensity dictionary into an h5 file.
    Parameters:
    -----------
    filename: str
        A filename where to store the data
    dic: dictionary
        A dictionary with module numbers as keys and intensity data
        as values in an xarray with 'trainId' as coordinates
    digitizer: dictionary
        A dictionary of digitizer data

    Return:
    -------

    """
    with h5py.File(filename, 'w') as f:
        g = f.create_group(f'entry_1/FXE_DET_LPD1M-1')
        for modno, data in dic.items():
            if data is not None and digitizer:
                h = g.create_group(f'module_{modno:02}')
                h.create_dataset('intensity', data=data.values)
                h.create_dataset('trainId', data=data.coords.get('trainId'))
                h.create_dataset('digitizer', data=digitizer)
            else:
                h = g.create_group(f'module_{modno:02}')
                h.create_dataset('intensity', data=data.values)
                h.create_dataset('trainId', data=data.coords.get('trainId'))


def file2intensity(filename, modules):
    """
    Retrieve intensity data from an h5 file as a dictionary
    Parameters:
    ----------
    filename: str
       An h5 file with intensity data
    modules: list, int
       A list of modules to put data in

    Return:
    -------
    dic: dictionary
        A dictionary with module numbers as keys and intensity data
        as values in an xarray with 'trainId' as coordinates
    """
    file_ = h5py.File(filename, 'r')
    dic = {}
    for modno in modules:
        grp = file_[f'entry_1/FXE_DET_LPD1M-1/module_{modno:02}']
        data = grp['intensity'][:]
        tid = grp['trainId'][:]
        arr = xr.DataArray(data, coords={'train': tid},
                           dims=['train', 'mem_cells'])

    dic[modno] = arr

    return dic


def delay2file(filename, dic):
    """
    Save a delay dictionary into an h5 file.
    Parameters:
    -----------
    filename: str
        A filename where to store the data
    dic: dictionary
        A dictionary with module numbers as keys and delay data
        as values in an xarray with '
        trainId' as coordinates

    Return:
    -------

    """
    with h5py.File(filename, 'w') as f:
        g = f.create_group(f'entry_1/FXE_DET_LPD1M-1')
        for modno, data in dic.items():
            if data is not None:
                h = g.create_group(f'module_{modno:02}')
                h.create_dataset(
                        'delay [ns]',
                        data=np.stack([pulse['x'] for pulse in dic[modno]]))
                h.create_dataset(
                        'ROI_intensity',
                        data=np.stack([pulse['y'] for pulse in dic[modno]]))
                h.create_dataset(
                        'error_y',
                        data=np.stack(
                            [pulse['error_y']['array'] for pulse in dic[modno]]))
                h.create_dataset(
                        'pulse',
                        data=np.stack(
                            [np.asarray(pulse['name'][-1:], dtype=np.uint8) for pulse in dic[modno]]))


def file2delay(filename, modules):
    """
    Retrieve delay data from an h5 file as a dictionary
    Parameters:
    ----------
    filename: str
       An h5 file with delayy data
    modules: list, int
       A list of modules to put data in

    Return:
    -------
    dic: dictionary
        A dictionary with module numbers as keys and delay data
        as values in an xarray with 'memory cells' and 'pulses'
        as coordinates
    """
    file_ = h5py.File(filename, 'r')
    dic = {}
    for modno in modules:
        grp = file_[f'entry_1/FXE_DET_LPD1M-1/module_{modno:02}']
        delay = grp['delay [ns]'][:]
        mean_ROI_int = grp['ROI_intensity'][:]
        error_int = grp['error_y'][:]
        pulse = grp['pulse'][:]
        arr = xr.Dataset(
            data_vars = {
                'delay_ns': (['mem_cells','point'], xr.DataArray(delay)),
                'ROI_intensity': (['mem_cells','point'], xr.DataArray(mean_ROI_int)),
                'error_int': (['mem_cells','point'], xr.DataArray(error_int))})

        dic[modno] = arr

    return dic