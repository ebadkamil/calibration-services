Bootstrap: library
From: ubuntu:18.04

%labels
    Author Ebad Kamil
    Email ebad.kamil@xfel.eu

%environment

    unset PYTHONPATH

    __conda_setup="$('/usr/local/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/usr/local/etc/profile.d/conda.sh" ]; then
            . "/usr/local/etc/profile.d/conda.sh"
        else
            export PATH="/usr/local/bin:$PATH"
        fi
    fi
    unset __conda_setup

    conda activate base
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
    export PATH=$PATH:/usr/local/bin

%post
    apt-get update && apt-get -y install wget git
    apt-get -y install build-essential
    unset PYTHONPATH
    #  Install miniconda
     wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
     chmod +x /tmp/miniconda.sh
     /tmp/miniconda.sh -bfp /usr/local/
     rm -f /tmp/miniconda.sh
     . /usr/local/bin/activate

    # Install OpenMPI
     wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz -O /tmp/openmpi-4.0.2.tar.gz
     tar xvzf /tmp/openmpi-4.0.2.tar.gz  -C /tmp/
     cd /tmp/openmpi-4.0.2
     ./configure --prefix=/usr/local
     make all install
     rm -rf /tmp/openmpi-4.0.2
     rm -f /tmp/openmpi-4.0.2.tar.gz

    cd /

    export PATH=$PATH:/usr/local/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
    # Install calibration_services
    # Require username and password
     git clone https://gitlab+deploy-token-3:r1yutEFxffXpNsTWZ9K7@git.xfel.eu/gitlab/dataAnalysis/calibration-services.git /usr/local/calibration_services
     cd /usr/local/calibration_services
     git checkout dev
     pip install .

    cd /

%runscript
    export PATH=$PATH:/usr/local/bin
    calibration_dashservice

%test
    export PATH=$PATH:/usr/local/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
    python -c 'import calibration; print(f"Calibration version: {calibration.__version__}")'
    python -c 'from mpi4py import MPI; print(f"Number of cores: {MPI.COMM_WORLD.size}")'
