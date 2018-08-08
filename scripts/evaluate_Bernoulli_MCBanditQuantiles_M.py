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
# Plotting
from plot_Bandits import *
# Optimal Bandit
from OptimalBandit import *
# Sampling
from BayesianBanditSampling import *
from MCBanditSampling import *
# Quantiles
from BayesianBanditQuantiles import *
from MCBanditQuantiles import *

# Main code
def main(A, t_max, M, R, exec_type, theta):

    ############################### MAIN CONFIG  ############################### 
    print('{}-armed Bernoulli bandit with Bayesian UCB and MC-BUCB policies for {} time-instants and {} realizations'.format(A, M, t_max, R))

    # Directory configuration
    dir_string='../results/{}/A={}/t_max={}/R={}/M={}/theta={}'.format(os.path.basename(__file__).split('.')[0], A, t_max, R, M, '_'.join(str.strip(np.array_str(theta.flatten()),'[]').split()))
    os.makedirs(dir_string, exist_ok=True)
    
    ########## Bernoulli Bandit configuration ##########
    # No context
    context=None    

    # Reward function and prior
    reward_function={'type':'bernoulli', 'dist':stats.bernoulli, 'theta':theta}
    reward_prior={'dist': stats.beta, 'alpha': np.ones((A,1)), 'beta': np.ones((A,1))}
    # MC Sampling
    # MC mininimum sampling uncertainty
    min_sampling_sigma=0.000001
    mc_reward_prior={'dist': stats.beta, 'alpha': np.ones((A,1)), 'beta': np.ones((A,1)), 'sampling':'density', 'sampling_sigma':min_sampling_sigma} 
    ###################### Monte Carlo sample sizes to consider #######################
    Ms=np.array([100, 500, 1000, 2000])
    
    ############################### BANDITS  ###############################    
    # Bandits to evaluate as a list
    bandits=[]
    bandits_labels=[]

    ### Quantile based
    alpha=1./np.arange(1,t_max+1)
    # Quantile info
    quantileInfo={'alpha':alpha, 'type':'analytical'}
    mc_quantile_empirical={'MC_alpha':'alpha', 'alpha':alpha, 'type':'empirical'}
    mc_quantile_sampling={'MC_alpha':'alpha', 'alpha':alpha, 'type':'sampling'}
    
    # Bayesian, analytical
    bandits.append(BayesianBanditQuantiles(A, reward_function, reward_prior, quantileInfo))
    bandits_labels.append('BUCB, alpha=1/t')

    # MC based empirical
    for m in Ms:
        # Monte-Carlo, empirical
        this_mc_reward_prior=mc_reward_prior.copy()
        this_mc_reward_prior['M']=m
        bandits.append(MCBanditQuantiles(A, reward_function, this_mc_reward_prior, mc_quantile_empirical))
        bandits_labels.append('MC-BUCB, empirical M={}, alpha=1/t'.format(m))
        # Monte-Carlo, sampling
        this_mc_quantile_sampling=mc_quantile_sampling.copy()
        this_mc_quantile_sampling['n_samples']=m
        bandits.append(MCBanditQuantiles(A, reward_function, this_mc_reward_prior, this_mc_quantile_sampling))
        bandits_labels.append('MC-BUCB, sampling M={}, alpha=1/t'.format(m))
        
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
    ## Plotting arrangements
    bandits_colors=[colors.cnames['black'], colors.cnames['red'], colors.cnames['salmon'], colors.cnames['navy'], colors.cnames['blue'], colors.cnames['green'], colors.cnames['lime'], colors.cnames['orange'], colors.cnames['yellow'], colors.cnames['cyan'], colors.cnames['lightblue'],  colors.cnames['purple'], colors.cnames['fuchsia'], colors.cnames['saddlebrown'], colors.cnames['peru']]
        
    # Plotting direcotries
    dir_plots=dir_string+'/plots'
    os.makedirs(dir_plots, exist_ok=True)

    # Plotting time: all
    t_plot=t_max
    
    ## GENERAL
    # Plot regret
    plot_std=False
    bandits_plot_regret(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_regret(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)

    # Plot cumregret
    plot_std=False
    bandits_plot_cumregret(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_cumregret(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    
    # Plot rewards expected
    plot_std=True
    bandits_plot_rewards_expected(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    
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
    
    ## Quantile bandits
    # Plot action quantiles
    plot_std=True
    bandits_plot_arm_quantile(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    ###############
            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    # Example: python3 -m pdb evaluate_Bernoulli_MCBanditQuantiles_M.py -A 2 -t_max 10 -M 50 -R 2 -exec_type sequential -theta 0.2 0.5
    parser = argparse.ArgumentParser(description='Evaluate Bernoulli BanditQuantiles MC posterior: Bayesian UCB and MC-BUCB policies')
    parser.add_argument('-A', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-M', type=int, default=1000, help='Number of samples for the MC integration')
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
    main(args.A, args.t_max, args.M, args.R, args.exec_type, theta)
