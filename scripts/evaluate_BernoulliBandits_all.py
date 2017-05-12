#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import pickle
import sys, os
import argparse
from itertools import *
import pdb
from matplotlib import colors

# Add path and import Bayesian Bandits
sys.path.append('../src')
from OptimalBandit import *
from BayesianBanditSampling import *
from plot_Bandits import *

# Main code
def main(A, t_max, M, N_max, R, exec_type, theta):

    ############################### MAIN CONFIG  ############################### 
    print('{}-armed Bernoulli bandit with optimal, TS and sampling policies with {} MC samples for {} time-instants and {} realizations'.format(A, M, t_max, R))

    # Directory configuration
    dir_string='../results/{}/A={}/t_max={}/R={}/M={}/N_max={}/theta={}'.format(os.path.basename(__file__).split('.')[0], A, t_max, R, M, N_max, str.replace(str.strip(np.array_str(theta.flatten()),' []'), '  ', '_'))
    os.makedirs(dir_string, exist_ok=True)
    
    ########## Bernoulli Bandit configuration ##########
    # No context
    context=None    

    # Reward function and prior
    reward_function={'dist':stats.bernoulli, 'theta':theta}
    reward_prior={'dist': stats.beta, 'alpha': np.ones((A,1)), 'beta': np.ones((A,1))}
    
    ############################### BANDITS  ############################### 
    ### Monte Carlo integration types
    MC_types=['MC_rewards', 'MC_expectedRewards', 'MC_arms']
    
    # Bandits to evaluate as a list
    bandits=[]
    bandits_labels=[]
    
    ### Optimal bandit
    bandits.append(OptimalBandit(A, reward_function))
    bandits_labels.append('Optimal Bandit')
    
    ### Thompson sampling: when sampling with static n=1
    thompsonSampling={'sampling_type':'static', 'arm_N_samples':1}

    # MC samples   
    for thompsonSampling['M'] in np.array([1,M]):
        # MC types
        for MC_type in MC_types:
            thompsonSampling['MC_type']=MC_type
            bandits.append(BayesianBanditSampling(A, reward_function, reward_prior, thompsonSampling))
            bandits_labels.append('TS, {}, M={}'.format(MC_type, thompsonSampling['M']))

    ### Inverse Pfa sampling
    # Truncated Gaussian with log10(1/Pfa)
    invPfaSampling={'sampling_type':'infPfa', 'Pfa':'tGaussian', 'f(1/Pfa)':np.log10, 'M': M, 'N_max':N_max}
    
    # MC types
    for MC_type in MC_types:
        invPfaSampling['MC_type']=MC_type
        bandits.append(BayesianBanditSampling(A, reward_function, reward_prior, invPfaSampling))
        bandits_labels.append('tGaussian: log10(1/Pfa), {}, M={}'.format(MC_type, invPfaSampling['M']))

    # Markov with log(1/Pfa)
    invPfaSampling={'sampling_type':'infPfa', 'Pfa':'Markov', 'f(1/Pfa)':np.log, 'M': M, 'N_max':N_max}
    
    # MC types
    for MC_type in MC_types:
        invPfaSampling['MC_type']=MC_type
        bandits.append(BayesianBanditSampling(A, reward_function, reward_prior, invPfaSampling))
        bandits_labels.append('Markov: log(1/Pfa), {}, M={}'.format(MC_type, invPfaSampling['M']))
        
    # Chebyshev with log(1/Pfa)
    invPfaSampling={'sampling_type':'infPfa', 'Pfa':'Chebyshev', 'f(1/Pfa)':np.log, 'M': M, 'N_max':N_max}
    
    # MC types
    for MC_type in MC_types:
        invPfaSampling['MC_type']=MC_type
        bandits.append(BayesianBanditSampling(A, reward_function, reward_prior, invPfaSampling))
        bandits_labels.append('Chebyshev: log(1/Pfa), {}, M={}'.format(MC_type, invPfaSampling['M']))
                           
    ### BANDIT EXECUTION
    # Execute each bandit
    for (n,bandit) in enumerate(bandits):
        bandit.execute_realizations(R, t_max, context, exec_type)
    
    # Save bandits info
    with open(dir_string+'/bandits.pickle', 'wb') as f:
        pickle.dump(bandits, f)
    with open(dir_string+'/bandits_labels.pickle', 'wb') as f:
        pickle.dump(bandits_labels, f)

    ############################### PLOTTING  ############################### 
    ## Plotting arrangements (in general)
    bandits_colors=[colors.cnames['black'], colors.cnames['skyblue'], colors.cnames['cyan'], colors.cnames['blue'], colors.cnames['palegreen'], colors.cnames['lime'], colors.cnames['green'], colors.cnames['yellow'], colors.cnames['orange'], colors.cnames['red'], colors.cnames['purple'], colors.cnames['fuchsia'], colors.cnames['pink'], colors.cnames['saddlebrown'], colors.cnames['chocolate'], colors.cnames['burlywood']]
    
    # Plotting direcotries
    dir_plots=dir_string+'/plots'
    os.makedirs(dir_plots, exist_ok=True)

    # Plotting time: all
    t_plot=t_max
    
    # Plot regret
    plot_std=False
    bandits_plot_regret(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_regret(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    
    # Plot rewards expected
    plot_std=True
    bandits_plot_rewards_expected(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)

    # Plot action predictive density
    plot_std=True
    bandits_plot_arm_density(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    
    # Plot actions
    plot_std=False
    bandits_plot_actions(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_actions(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)

    # Plot correct actions
    plot_std=False
    bandits_plot_actions_correct(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_actions_correct(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    ###############
            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    # Example: python3 -m pdb evaluate_BernoulliBandits_all.py -A 2 -t_max 10 -M 50 -N_max 20 -R 2 -exec_type sequential -theta 0.2 0.5
    parser = argparse.ArgumentParser(description='Evaluate Bernoulli Bandits: optimal, TS and all sampling policy approaches.')
    parser.add_argument('-A', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-M', type=int, default=1000, help='Number of samples for the MC integration')
    parser.add_argument('-N_max', type=int, default=25, help='Maximum number of arm candidate samples')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-exec_type', type=str, default='sequential', help='Type of execution to run: batch or sequential')
    parser.add_argument('-theta', nargs='+', type=float, default=None, help='Theta parameters of the Bernoulli distribution')

    # Get arguments
    args = parser.parse_args()
    
    # Make sure A and theta size match
    if args.theta != None:
        assert len(args.theta)==args.A, 'Not enough Bernoulli parameters theta={} provided for A={}'.format(args.theta, args.A)
        theta=np.reshape(args.theta, (args.A))
    else:
        # Random
        theta=np.random.randn(args.A)

    # Call main function
    main(args.A, args.t_max, args.M, args.N_max, args.R, args.exec_type, theta)
