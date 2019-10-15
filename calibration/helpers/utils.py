"""
Calibration analysis and visualization for AGIPD Detector

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

from itertools import chain


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

