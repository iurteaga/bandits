#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import sys, os, re
import argparse
from itertools import *

# Main code
def main(type_context, t_max, R, prior_K, exec_type):
    print('Executing linear mixture Variational {} contextual bandits with {} time-instants and {} realizations'.format(type_context, t_max, R))
  
    # Load template job script
    with open('bayesian_bandits_sampling_job.sh') as template_script:
        template_script_data=template_script.read()

        ########## Different configurations to run ##########
        ### Config 1 balanced
        A=2
        K=2
        d_context=2
        pi=np.array(
            [[0.5,0.5],
            [0.5,0.5]])
        theta=np.array(
            [[[0,0],[1,1]],
            [[2,2],[3,3]]])
        sigma=np.array(
            [[1.,1.],
            [1.,1.]])

        # Execute
        # Open new job script file to write
        new_job_name='evaluate_VariationalContextualBanditsSampling_job_A{}_K{}_dcontext{}_typecontext{}_tmax{}_R{}_pi{}_theta{}_sigma{}_priorK{}'.format(A, K, d_context, type_context, t_max, R, str.replace(str.strip(np.array_str(pi.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(theta.flatten()),' []'), ' ', '_'), str.replace(str.strip(np.array_str(sigma.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'))
        new_job_name='evaluate_VariationalContextualBanditsSampling_job_config1_balanced_typecontext{}_tmax{}_R{}_priorK{}'.format(type_context, t_max, R, str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'))
        with open(new_job_name+'.sh', 'w') as new_job_script:
            # Change script job name
            new_script_data=re.sub('bayesian_bandits_sampling_job',new_job_name,template_script_data)
            # Change what python script to run
            new_script_data=re.sub('python_job_to_run', 'python $PBS_O_INITDIR/../scripts/evaluate_VariationalContextualBanditsSampling.py -A {} -K {} -d_context {} -type_context {} -pi {} -theta {} -sigma {} -t_max {} -R {} -prior_K {} -exec_type {}'.format(A,K,d_context, type_context, str.strip(np.array_str(pi.flatten()),' []'), str.strip(np.array_str(theta.flatten()),' []'), str.strip(np.array_str(sigma.flatten()),' []'), t_max, R, str.strip(np.array_str(prior_K.flatten()),' []'), exec_type),new_script_data)
            # Write to file and close
            new_job_script.write(new_script_data)
            new_job_script.close()
        
        # Execute new script
        print('qsub {}'.format(new_job_name+'.sh'))
        os.system('qsub {}'.format(new_job_name+'.sh'))

        ### Config 1 unbalanced
        d_context=2
        K=2
        pi=np.array(
            [[0.2,0.8],
            [0.8,0.2]])
        theta=np.array(
            [[[0,0],[1,1]],
            [[2,2],[3,3]]])
        sigma=np.array(
            [[1.,1.],
            [1.,1.]])
        
        # Execute
        # Open new job script file to write
        new_job_name='evaluate_VariationalContextualBanditsSampling_job_A{}_K{}_dcontext{}_typecontext{}_tmax{}_R{}_pi{}_theta{}_sigma{}_priorK{}'.format(A, K, d_context, type_context, t_max, R, str.replace(str.strip(np.array_str(pi.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(theta.flatten()),' []'), ' ', '_'), str.replace(str.strip(np.array_str(sigma.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'))
        new_job_name='evaluate_VariationalContextualBanditsSampling_job_config1_unbalanced_typecontext{}_tmax{}_R{}_priorK{}'.format(type_context, t_max, R, str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'))
        with open(new_job_name+'.sh', 'w') as new_job_script:
            # Change script job name
            new_script_data=re.sub('bayesian_bandits_sampling_job',new_job_name,template_script_data)
            # Change what python script to run
            new_script_data=re.sub('python_job_to_run', 'python $PBS_O_INITDIR/../scripts/evaluate_VariationalContextualBanditsSampling.py -A {} -K {} -d_context {} -type_context {} -pi {} -theta {} -sigma {} -t_max {} -R {} -prior_K {} -exec_type {}'.format(A,K,d_context, type_context, str.strip(np.array_str(pi.flatten()),' []'), str.strip(np.array_str(theta.flatten()),' []'), str.strip(np.array_str(sigma.flatten()),' []'), t_max, R, str.strip(np.array_str(prior_K.flatten()),' []'), exec_type),new_script_data)
            # Write to file and close
            new_job_script.write(new_script_data)
            new_job_script.close()
        
        # Execute new script
        print('qsub {}'.format(new_job_name+'.sh'))
        os.system('qsub {}'.format(new_job_name+'.sh'))
        
        ### Config 2 balanced
        A=2
        K=2
        d_context=2
        pi=np.array(
            [[0.5,0.5],
            [0.5,0.5]])
        theta=np.array(
            [[[1,1],[2,2]],
            [[0,0],[3,3]]])
        sigma=np.array(
            [[1.,1.],
            [1.,1.]])

        # Execute
        # Open new job script file to write
        new_job_name='evaluate_VariationalContextualBanditsSampling_job_A{}_K{}_dcontext{}_typecontext{}_tmax{}_R{}_pi{}_theta{}_sigma{}_priorK{}'.format(A, K, d_context, type_context, t_max, R, str.replace(str.strip(np.array_str(pi.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(theta.flatten()),' []'), ' ', '_'), str.replace(str.strip(np.array_str(sigma.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'))
        new_job_name='evaluate_VariationalContextualBanditsSampling_job_config2_balanced_typecontext{}_tmax{}_R{}_priorK{}'.format(type_context, t_max, R, str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'))
        with open(new_job_name+'.sh', 'w') as new_job_script:
            # Change script job name
            new_script_data=re.sub('bayesian_bandits_sampling_job',new_job_name,template_script_data)
            # Change what python script to run
            new_script_data=re.sub('python_job_to_run', 'python $PBS_O_INITDIR/../scripts/evaluate_VariationalContextualBanditsSampling.py -A {} -K {} -d_context {} -type_context {} -pi {} -theta {} -sigma {} -t_max {} -R {} -prior_K {} -exec_type {}'.format(A,K,d_context, type_context, str.strip(np.array_str(pi.flatten()),' []'), str.strip(np.array_str(theta.flatten()),' []'), str.strip(np.array_str(sigma.flatten()),' []'), t_max, R, str.strip(np.array_str(prior_K.flatten()),' []'), exec_type),new_script_data)
            # Write to file and close
            new_job_script.write(new_script_data)
            new_job_script.close()
        
        # Execute new script
        print('qsub {}'.format(new_job_name+'.sh'))
        os.system('qsub {}'.format(new_job_name+'.sh'))

        ### Config 2 unbalanced
        d_context=2
        K=2
        pi=np.array(
            [[0.3,0.7],
            [0.3,0.7]])
        theta=np.array(
            [[[1,1],[2,2]],
            [[0,0],[3,3]]])
        sigma=np.array(
            [[1.,1.],
            [1.,1.]])
        
        # Execute
        # Open new job script file to write
        new_job_name='evaluate_VariationalContextualBanditsSampling_job_A{}_K{}_dcontext{}_typecontext{}_tmax{}_R{}_pi{}_theta{}_sigma{}_priorK{}'.format(A, K, d_context, type_context, t_max, R, str.replace(str.strip(np.array_str(pi.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(theta.flatten()),' []'), ' ', '_'), str.replace(str.strip(np.array_str(sigma.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'))
        new_job_name='evaluate_VariationalContextualBanditsSampling_job_config2_unbalanced_typecontext{}_tmax{}_R{}_priorK{}'.format(type_context, t_max, R, str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'))
        with open(new_job_name+'.sh', 'w') as new_job_script:
            # Change script job name
            new_script_data=re.sub('bayesian_bandits_sampling_job',new_job_name,template_script_data)
            # Change what python script to run
            new_script_data=re.sub('python_job_to_run', 'python $PBS_O_INITDIR/../scripts/evaluate_VariationalContextualBanditsSampling.py -A {} -K {} -d_context {} -type_context {} -pi {} -theta {} -sigma {} -t_max {} -R {} -prior_K {} -exec_type {}'.format(A,K,d_context, type_context, str.strip(np.array_str(pi.flatten()),' []'), str.strip(np.array_str(theta.flatten()),' []'), str.strip(np.array_str(sigma.flatten()),' []'), t_max, R, str.strip(np.array_str(prior_K.flatten()),' []'), exec_type),new_script_data)
            # Write to file and close
            new_job_script.write(new_script_data)
            new_job_script.close()
        
        # Execute new script
        print('qsub {}'.format(new_job_name+'.sh'))
        os.system('qsub {}'.format(new_job_name+'.sh'))
        
        ### Config 3 balanced
        A=2
        K=2
        d_context=2
        pi=np.array(
            [[0.5,0.5],
            [0.5,0.5]])
        theta=np.array(
            [[[0,0],[1,1]],
            [[-1,-1],[2,2]]])
        sigma=np.array(
            [[1.,1.],
            [1.,1.]])

        # Execute
        # Open new job script file to write
        new_job_name='evaluate_VariationalContextualBanditsSampling_job_A{}_K{}_dcontext{}_typecontext{}_tmax{}_R{}_pi{}_theta{}_sigma{}_priorK{}'.format(A, K, d_context, type_context, t_max, R, str.replace(str.strip(np.array_str(pi.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(theta.flatten()),' []'), ' ', '_'), str.replace(str.strip(np.array_str(sigma.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'))
        new_job_name='evaluate_VariationalContextualBanditsSampling_job_config3_balanced_typecontext{}_tmax{}_R{}_priorK{}'.format(type_context, t_max, R, str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'))
        with open(new_job_name+'.sh', 'w') as new_job_script:
            # Change script job name
            new_script_data=re.sub('bayesian_bandits_sampling_job',new_job_name,template_script_data)
            # Change what python script to run
            new_script_data=re.sub('python_job_to_run', 'python $PBS_O_INITDIR/../scripts/evaluate_VariationalContextualBanditsSampling.py -A {} -K {} -d_context {} -type_context {} -pi {} -theta {} -sigma {} -t_max {} -R {} -prior_K {} -exec_type {}'.format(A,K,d_context, type_context, str.strip(np.array_str(pi.flatten()),' []'), str.strip(np.array_str(theta.flatten()),' []'), str.strip(np.array_str(sigma.flatten()),' []'), t_max, R, str.strip(np.array_str(prior_K.flatten()),' []'), exec_type),new_script_data)
            # Write to file and close
            new_job_script.write(new_script_data)
            new_job_script.close()
        
        # Execute new script
        print('qsub {}'.format(new_job_name+'.sh'))
        os.system('qsub {}'.format(new_job_name+'.sh'))

        ### Config 3 unbalanced
        d_context=2
        K=2
        pi=np.array(
            [[0.3,0.7],
            [0.3,0.7]])
        theta=np.array(
            [[[0,0],[1,1]],
            [[-1,-1],[2,2]]])
        sigma=np.array(
            [[1.,1.],
            [1.,1.]])
        
        # Execute
        # Open new job script file to write
        new_job_name='evaluate_VariationalContextualBanditsSampling_job_A{}_K{}_dcontext{}_typecontext{}_tmax{}_R{}_pi{}_theta{}_sigma{}_priorK{}'.format(A, K, d_context, type_context, t_max, R, str.replace(str.strip(np.array_str(pi.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(theta.flatten()),' []'), ' ', '_'), str.replace(str.strip(np.array_str(sigma.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'))
        new_job_name='evaluate_VariationalContextualBanditsSampling_job_config3_unbalanced_typecontext{}_tmax{}_R{}_priorK{}'.format(type_context, t_max, R, str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'))
        with open(new_job_name+'.sh', 'w') as new_job_script:
            # Change script job name
            new_script_data=re.sub('bayesian_bandits_sampling_job',new_job_name,template_script_data)
            # Change what python script to run
            new_script_data=re.sub('python_job_to_run', 'python $PBS_O_INITDIR/../scripts/evaluate_VariationalContextualBanditsSampling.py -A {} -K {} -d_context {} -type_context {} -pi {} -theta {} -sigma {} -t_max {} -R {} -prior_K {} -exec_type {}'.format(A,K,d_context, type_context, str.strip(np.array_str(pi.flatten()),' []'), str.strip(np.array_str(theta.flatten()),' []'), str.strip(np.array_str(sigma.flatten()),' []'), t_max, R, str.strip(np.array_str(prior_K.flatten()),' []'), exec_type),new_script_data)
            # Write to file and close
            new_job_script.write(new_script_data)
            new_job_script.close()
        
        # Execute new script
        print('qsub {}'.format(new_job_name+'.sh'))
        os.system('qsub {}'.format(new_job_name+'.sh'))       
            
            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Evaluate Variational Contextual bandits.')
    parser.add_argument('-type_context', type=str, default='static', help='Type of context: static (default), randn, rand')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-prior_K', nargs='+', type=int, default=2, help='Assumed prior mixtures (per arm)')
    parser.add_argument('-exec_type', type=str, default='online', help='Type of execution to run: online or all')

    # Get arguments
    args = parser.parse_args()
    
    # Call main function
    main(args.type_context, args.t_max, args.R, np.array(args.prior_K), args.exec_type)
