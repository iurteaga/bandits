#!/bin/sh

# Directives
#PBS -N bayesian_bandits_sampling_linear_500
#PBS -W group_list=yetidsi
#PBS -M iu2153@columbia.edu
#PBS -m bae
#PBS -V

# Wall time
# 1 day walltime=01:00:00:00
# 12 hour walltime=12:00:00
#PBS -l walltime=05:00:00:00

# Output and error directories
#PBS -o localhost:/vega/dsi/users/iu2153/bayesianBandits/yeti/out/$PBS_JOBNAME.out
#PBS -e localhost:/vega/dsi/users/iu2153/bayesianBandits/yeti/log/$PBS_JOBNAME.log

# Clean previous output/error files
rm /vega/dsi/users/iu2153/bayesianBandits/yeti/out/$PBS_JOBNAME.out
rm /vega/dsi/users/iu2153/bayesianBandits/yeti/log/$PBS_JOBNAME.log

# Start
echo "Starting script $PBS_JOBNAME at $(date)"

# Call script
# Load python3 module (with installed updated modules)
module load anaconda/4.1.1-python-3.5.2

# Run script
python $PBS_O_INITDIR/../scripts/evaluate_bayesian_bandits_sampling_linear.py -K 2 -t_max 500 -R 1000 -theta_diff 0.1

# Finished
echo "Done at $(date)"

# END
