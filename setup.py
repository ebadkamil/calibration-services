import os.path as osp
import re
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


def find_version():
    with open(osp.join('calibration', '__init__.py'), 'r') as f:
        match = re.search(r'^__version__ = "(\d+\.\d+\.\d+)"', f.read(), re.M)
        if match is not None:
            return match.group(1)
        raise RuntimeError("Unable to find version string.")


class PyTest(TestCommand):
    def run(self):
        import pytest
        errno = pytest.main(['--pyargs', 'calibration'])
        sys.exit(errno)


setup(name="calibration",
      version=find_version(),
      author="European XFEL GmbH",
      author_email="ebad.kamil@xfel.eu",
      maintainer="Ebad Kamil",
      packages=find_packages(),
      package_data={
        'calibration.geometries': ['*.h5']
      },
      cmdclass = {'test': PyTest},
      entry_points={
          "console_scripts": [
              "detector_characterize = calibration.application:detector_characterize",
              "calibration_dashservice = calibration.application:run_dashservice",
          ],
      },
      install_requires=[
           'extra_data>=1.1.0',
           'extra_geom>=0.9.0',
           'dash>=1.6.1',
           'dash-daq>=0.3.1',
           'ipywidgets>=7.5.1',
           'mpi4py>=3.0.2',
           'iminuit',
           'pyFAI>0.16.0'
      ],
      extras_require={
        'test': [
          'pytest',
        ]
      },
      python_requires='>=3.6',
)
