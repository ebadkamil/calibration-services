import os.path as osp
import re
from setuptools import setup, find_packages
import sys


def get_script_path():
    return osp.dirname(osp.realpath(sys.argv[0]))


def read(*parts):
    return open(osp.join(get_script_path(), *parts)).read()


def find_version(*parts):
    vers_file = read(*parts)
    match = re.search(r'^__version__ = "(\d+\.\d+\.\d+)"', vers_file, re.M)
    if match is not None:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name="calibration",
      version=find_version("calibration", "__init__.py"),
      author="European XFEL GmbH",
      author_email="ebad.kamil@xfel.eu",
      maintainer="Ebad Kamil",
      packages=find_packages(),
      entry_points={
          "console_scripts": [
              "detector_characterize = calibration.application:detector_characterize",
              "calibration_dashservice = calibration.application:run_dashservice",
          ],
      },
      install_requires=[
           'karabo-data>=0.7.0',
           'dash>=1.6.1',
           'dash-daq>=0.3.1',
           'dask>=2.7.0',
           'dask-jobqueue>=0.7.0',
           'ipywidgets>=7.5.1',
           'mpi4py>=3.0.2',
      ],
      python_requires='>=3.6',
)