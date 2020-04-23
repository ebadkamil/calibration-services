Installing
==========

`calibration` 

Create conda environment with Python 3.5 or later:

    conda create -n {env_name} python=3.6

Activate conda environment:

    conda activate {env_name}
    conda install -c conda-forge openmpi 
    pip install -e .

On Maxwell cluster:

	module load exfel calibration-services

Jupyter kernel `calibration-services-kernel` will be created that can 
be used in jupyter notebooks.