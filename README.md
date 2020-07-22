Installing
==========

`calibration` 

Create conda environment with Python 3.5 or later:

    conda create -n {env_name} python=3.6

Activate conda environment:

    conda activate {env_name}
    conda install -c conda-forge openmpi 
    pip install -e .

How to use at XFEL
==================

Use of notebooks
----------------

- Get the repository in your workspace, either on online or Maxwell cluster:
    
        user@machine git clone https://git.xfel.eu/gitlab/dataAnalysis/calibration-services.git

    Notebooks are available in the folder `calibration-services/notebooks`

Running a jupyter server on online cluster
------------------------------------------

- Via FastX client

    Open a terminal and type the following commands

        user@max-display: ssh exflgateway
        ssh exflonc<NN>  #NN = 12 for eg.
        module load exfel calibration-services
        jupyter-notebook --port 8008 --no-browser

    This will start a jupyter server on online cluster and a link will be provided that you can copy.

    Now open another terminal and do the port forwarding

        user@max-display: ssh -L 8008:localhost:8008 exflgateway -t ssh -L 8008:localhost:8008 exflonc<NN>

    Open your regular browser and paste the link that you copied before the last step. Provided you have cloned the repository as shown in first step, the calibration-services folder will be visible in the jupyter notebook.

- Via local computer

    Open a terminal and type the following commands

        user@localmachine$ ssh user@bastion.desy.de
        ssh exflgateway
        ssh exflonc<NN>
        module load exfel calibration-services
        jupyter-notebook --port 8008 --no-browser --ip "*"

    Now open another terminal and do the port forwarding
        user@localmachine: ssh -L 8008:localhost:8008 user@bastion.desy.de -t ssh -L 8008:exflonc<NN>:8008 exflgateway
    
    Open your regular browser and paste the link that you copied before the last step.

- From the hutch control room computers

    Open a terminal and type the following commands

        ssh exflonc<NN>
        module load exfel calibration-services
        jupyter-notebook --port 8008 --no-browser
    
    Open another terminal and do the port forwarding
        ssh -L 8008:localhost:8008 exflonc<NN>

   Open your regular browser and paste the link that you copied before the last step.

Running notebooks on Maxwell cluster
------------------------------------

On Maxwell cluster:

	module load exfel calibration-services

Preferred way is to use jupyterhub instance running on https://max-jhub.desy.de/hub/login

Which jupyter kernel to use
---------------------------

Doing `module load exfel calibration-services` once, a jupyter kernel `calibration-services-kernel` will be created that can be used in jupyter notebooks.