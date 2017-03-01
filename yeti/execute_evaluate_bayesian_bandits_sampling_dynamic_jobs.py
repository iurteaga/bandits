#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import sys, os, re
import argparse
from itertools import *

# Main code
def main(M, A, t_max, R, exec_type, theta_min, theta_max, theta_diff):
    print('Executing jobs for bayesian {}-armed bayesian bandit with {} samples for {} time-instants and {} realizations'.format(A, M, t_max, R))
  
    # Load template job script
    with open('bayesian_bandits_sampling_job.sh') as template_script:
        template_script_data=template_script.read()
        # Bandit configuration
        # for theta in combinations_with_replacement(np.arange(0.1,1.0,theta_diff),A):
        for theta in combinations(np.arange(theta_min,theta_max,theta_diff),A):
            # For each theta (in string format)
            theta_str='_'.join([str(i) for i in theta])
            
            # Open new job script file to write
            new_job_name='bayesian_bandits_sampling_dynamic_job_A_{}_tmax_{}_R_{}_{}_M_{}_theta_{}.sh'.format(A,t_max,R,exec_type,M,theta_str)
            with open(new_job_name, 'w') as new_job_script:
                # Change script job name
                new_script_data=re.sub('bayesian_bandits_sampling_job',new_job_name,template_script_data)
                # Change what python script to run
                new_script_data=re.sub('python_job_to_run','python $PBS_O_INITDIR/../scripts/evaluate_bayesian_bandits_sampling_dynamic.py -M {} -A {} -t_max {} -R {} -exec_type {} -theta {}'.format(M,A,t_max,R,exec_type, theta_str.replace('_', ' ')),new_script_data)
                # Write to file and close
                new_job_script.write(new_script_data)
                new_job_script.close()
            # Execute new script
            print('qsub {}'.format(new_job_name))
            os.system('qsub {}'.format(new_job_name))
            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Evaluate Bayesian bandits.')
    parser.add_argument('-M', type=int, default=1000, help='Number of samples for the MC integration')
    parser.add_argument('-A', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-exec_type', type=str, default='online', help='Type of execution to run: online or all')
    parser.add_argument('-theta_min', type=float, default=0, help='Minimum theta')
    parser.add_argument('-theta_max', type=float, default=1, help='Maximum theta')
    parser.add_argument('-theta_diff', type=float, default=0.5, help='Differences for theta')

    # Get arguments
    args = parser.parse_args()
    
    # Call main function
    main(args.M, args.A, args.t_max, args.R, args.exec_type, args.theta_min, args.theta_max, args.theta_diff)
