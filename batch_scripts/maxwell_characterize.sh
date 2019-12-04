#!/bin/bash
#SBATCH --partition=exfel
#SBATCH --time=10:00:00       # Maximum time requested
#SBATCH --nodes=1             # Number of nodes

echo "start at " `date`

cd $PWD
echo "Current Working directory is $PWD"

source $HOME/.bashrc
conda activate calibration

echo `which detector_characterize`

module=$1
proposal=900091
run=491
detector='agipd'

bin_low=-200
bin_high=400
nbins=601

pulseids="1:24:2"

detector_characterize ${detector} ${module} --proposal ${proposal} \
--run ${run} --bin_low ${bin_low} --bin_high ${bin_high} --nbins ${nbins} \
--pulseids ${pulseids} --subtract_dark --pixel_hist --fit


if [ $? -ne 0 ]; then echo "error in $0 :QUIT detector characterize:";fi
echo "program end at " `date`