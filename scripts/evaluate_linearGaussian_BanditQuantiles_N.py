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
def main(A, t_max, M, R, exec_type, theta, sigma, d_context, type_context):

    ############################### MAIN CONFIG  ############################### 
    print('{}-armed Contextual Linear Gaussian bandit with Bayesian UCB and MC-BUCB policy sampling for {} time-instants and {} realizations'.format(A, M, t_max, R))

    # Directory configuration
    dir_string='../results/{}/A={}/t_max={}/R={}/M={}/d_context={}/type_context={}/theta={}/sigma={}'.format(os.path.basename(__file__).split('.')[0], A, t_max, R, M, d_context, type_context, '_'.join(str.strip(np.array_str(theta.flatten()),'[]').split()),  '_'.join(str.strip(np.array_str(sigma.flatten()),'[]').split()))
    os.makedirs(dir_string, exist_ok=True)
    
    ########## Contextual Bandit configuration ##########
    # Context
    if type_context=='static':
        # Static context
        context=np.ones((d_context,t_max))
    elif type_context=='randn':
        # Dynamic context: standard Gaussian
        context=np.random.randn(d_context,t_max)
    elif type_context=='rand':
        # Dynamic context: uniform
        context=np.random.rand(d_context,t_max)
    else:
        # Unknown context
        raise ValueError('Invalid context type={}'.format(type_context))
    
    # Reward function
    reward_function={'type':'linear_gaussian', 'dist':stats.norm, 'theta':theta, 'sigma':sigma}
    # Reward prior
    Sigmas=np.zeros((A, d_context, d_context))
    for a in np.arange(A):
        Sigmas[a,:,:]=np.eye(d_context)

    reward_prior_known={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas}
    reward_prior_unknown={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'alpha':np.ones((A,1)), 'beta':np.ones((A,1))}
    mc_reward_prior_known={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'sampling':'density', 'sampling_sigma':0.01, 'M':M}
    mc_reward_prior_unknown={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'alpha':np.ones((A,1)), 'beta':np.ones((A,1)), 'sampling':'density', 'sampling_sigma':0.01, 'M':M}
    ###################### Quantile sampling sizes N to consider #######################
    Ns=np.array([100, 500, 1000])

    ############################### BANDITS  ###############################    
    # Bandits to evaluate as a list
    bandits=[]
    bandits_labels=[]

    ### Quantile based
    alpha=1./np.arange(1,t_max+1)
    # Bayesian, analytical
    quantileInfo={'alpha':alpha, 'type':'analytical'}
    
    # Known sigma
    bandits.append(BayesianBanditQuantiles(A, reward_function, reward_prior_known, quantileInfo))
    bandits_labels.append('BUCB (known $\sigma_y^2$), alpha=1/t')

    # Unknown sigma
    bandits.append(BayesianBanditQuantiles(A, reward_function, reward_prior_unknown, quantileInfo))
    bandits_labels.append('BUCB (unknown $\sigma_y^2$), alpha=1/t')
    
    # MC Empirical type
    quantileInfo={'MC_alpha':'alpha', 'alpha':alpha, 'type':'empirical'}
    
    # Known sigma
    bandits.append(MCBanditQuantiles(A, reward_function, mc_reward_prior_known, quantileInfo))
    bandits_labels.append('MC-BUCB (known $\sigma_y^2$), M={}, alpha=1/t'.format(M))
    # Unknown sigma
    bandits.append(MCBanditQuantiles(A, reward_function, mc_reward_prior_unknown, quantileInfo))
    bandits_labels.append('MC-BUCB (unknown $\sigma_y^2$), M={}, alpha=1/t'.format(M))

    # Bayesian, sampling
    for n in Ns:
        # Analytical sampling
        quantileInfo={'alpha':alpha, 'type':'sampling', 'n_samples':n}
        # Known sigma
        bandits.append(BayesianBanditQuantiles(A, reward_function, reward_prior_known, quantileInfo))
        bandits_labels.append('BUCB (known $\sigma_y^2$), sampling N={}, alpha=1/t'.format(n))
        # Unknown sigma
        bandits.append(BayesianBanditQuantiles(A, reward_function, reward_prior_unknown, quantileInfo))
        bandits_labels.append('BUCB (known $\sigma_y^2$), sampling N={}, alpha=1/t'.format(n))
        
        # MC Sampling
        quantileInfo={'MC_alpha':'alpha', 'alpha':alpha, 'type':'sampling', 'n_samples':n}
        # Known sigma
        bandits.append(MCBanditQuantiles(A, reward_function, mc_reward_prior_known, quantileInfo))
        bandits_labels.append('MC-BUCB (known $\sigma_y^2$), M={}, sampling N={}, alpha=1/t'.format(M,n))
        # Unknown sigma
        bandits.append(MCBanditQuantiles(A, reward_function, mc_reward_prior_unknown, quantileInfo))
        bandits_labels.append('MC-BUCB (unknown $\sigma_y^2$), M={}, sampling N={}, alpha=1/t'.format(M,n))

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
    bandits_colors=[colors.cnames['black'], colors.cnames['silver'], colors.cnames['red'], colors.cnames['salmon'], colors.cnames['navy'], colors.cnames['blue'], colors.cnames['green'], colors.cnames['lime'], colors.cnames['orange'], colors.cnames['yellow'], colors.cnames['cyan'], colors.cnames['lightblue'],  colors.cnames['purple'], colors.cnames['fuchsia'], colors.cnames['saddlebrown'], colors.cnames['peru']]

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
    # Example: python3 -m pdb evaluate_linearGaussian_BanditQuantiles_N.py -A 2 -t_max 10 -M 50 -R 2 -exec_type batch -d_context 2 -type_context randn -theta 1 1 -1 -1 -sigma 1 1
    parser = argparse.ArgumentParser(description='Evaluate Contextual Linear Gaussian BanditQuantiles: Bayesian UCB and MC-BUCB policies with quantile sampling')
    parser.add_argument('-A', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-M', type=int, default=1000, help='Number of samples for the MC integration')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-exec_type', type=str, default='sequential', help='Type of execution to run: batch or sequential')
    parser.add_argument('-d_context', type=int, default=2, help='Dimensionality of context')
    parser.add_argument('-type_context', type=str, default='static', help='Type of context: static (default), randn, rand')
    parser.add_argument('-theta', nargs='+', type=float, default=None, help='Thetas per arm')
    parser.add_argument('-sigma', nargs='+', type=float, default=None, help='Variances per arm')

    # Get arguments
    args = parser.parse_args()
    
    # Regressors
    if args.theta != None:
        assert len(args.theta)==args.A*args.d_context, 'Not enough regression parameters theta={} provided for A={} and d_context={}'.format(args.theta, args.A, args.d_context)
        theta=np.reshape(args.theta, (args.A,args.d_context))
    else:
        # Random
        theta=np.random.randn(args.A,args.d_context)

    # Emission variance
    if args.sigma != None:
        assert len(args.sigma)==args.A , 'Not enough emission variance parameters sigma={} provided for A={} '.format(args.sigma, args.A)
        sigma=np.reshape(args.sigma,(args.A))
    else:
        # Unit variance
        sigma=np.ones((args.A))

    # Call main function
    main(args.A, args.t_max, args.M, args.R, args.exec_type, theta, sigma, args.d_context, args.type_context)
