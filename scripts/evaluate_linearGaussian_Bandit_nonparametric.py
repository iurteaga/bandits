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
# Sampling
from BayesianBanditSampling import *
from MCMCBanditSampling import *

# Main code
def main(A, t_max, M, N_max, R, exec_type, theta, sigma, d_context, type_context):

    ############################### MAIN CONFIG  ############################### 
    print('{}-armed Contextual Linear Gaussian bandit and nonparametric bandit with TS for {} time-instants and {} realizations'.format(A, M, t_max, R))

    # Directory configuration
    dir_string='../results/{}/A={}/t_max={}/R={}/M={}/N_max={}/d_context={}/type_context={}/theta={}/sigma={}'.format(os.path.basename(__file__).split('.')[0], A, t_max, R, M, N_max, d_context, type_context, '_'.join(str.strip(np.array_str(theta.flatten()),'[]').split()),  '_'.join(str.strip(np.array_str(sigma.flatten()),'[]').split()))
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
    
    ############################### BANDITS  ###############################    
    # Bandits to evaluate as a list
    bandits=[]
    bandits_labels=[]
       
    ### Thompson sampling: when sampling with static n=1 and no MC
    thompsonSampling={'sampling_type':'static', 'MC_type':'MC_arms', 'M':1, 'arm_N_samples':1}

    ########## Linear Gaussian with unknown sigma
    # Reward function
    reward_function={'type':'linear_gaussian', 'dist':stats.norm, 'theta':theta, 'sigma':sigma}
    # Reward prior sigmas
    Sigmas=np.zeros((A, d_context, d_context))
    for a in np.arange(A):
        Sigmas[a,:,:]=np.eye(d_context)

    reward_prior_unknown={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'alpha':np.ones((A,1)), 'beta':np.ones((A,1))}
    # Instantiate bandit
    bandits.append(BayesianBanditSampling(A, reward_function, reward_prior_unknown, thompsonSampling))
    bandits_labels.append('TS linear Gaussian')

    ########## Nonparametric linear Gaussian mixture
    # MCMC (Gibbs) parameters
    gibbs_max_iter=10
    gibbs_loglik_eps=0.01
    # Plotting
    gibbs_plot_save='show'
    gibbs_plot_save=None
    if gibbs_plot_save != None and gibbs_plot_save != 'show':
        # Plotting directories
        gibbs_plots=dir_string+'/gibbs_plots'
        os.makedirs(gibbs_plots, exist_ok=True)
        os.makedirs(gibbs_plots+'/nonparametric', exist_ok=True)
        gibbs_plot_save=gibbs_plots+'/nonparametric'

    ########## Priors
    gamma=0.1
    alpha=1.
    beta=1.
    sigma_0=1.
    pitman_yor_d=0
    assert (0<=pitman_yor_d) and (pitman_yor_d<1) and (gamma >-pitman_yor_d)
    # Reward function
    np_reward_function={'type':'linear_gaussian_mixture', 'dist':stats.norm, 'pi':np.ones((A,1)), 'theta':theta[:,None,], 'sigma':sigma*np.ones((A,1))}
    np_thompsonSampling={'sampling_type':'static', 'MC_type':'MC_arms', 'M':1, 'arm_N_samples':1, 'mixture_expectation':'pi_expected'}
    # Hyperparameters
    # Concentration parameter
    prior_d=pitman_yor_d*np.ones(A)
    prior_gamma=gamma*np.ones(A)
    # NIG for linear Gaussians
    prior_alpha=alpha*np.ones(A)
    prior_beta=beta*np.ones(A)

    # Initial thetas
    prior_theta=np.ones((A,d_context))            
    prior_Sigma=np.zeros((A,d_context, d_context))
    # Initial covariances: uncorrelated
    for a in np.arange(A):
        prior_Sigma[a,:,:]=sigma_0*np.eye(d_context)
            
    # Reward prior as dictionary
    np_reward_prior={'type':'linear_gaussian_mixture', 'dist':'NIG', 'K':'nonparametric', 'd':prior_d, 'gamma':prior_gamma, 'alpha':prior_alpha, 'beta':prior_beta, 'theta':prior_theta, 'Sigma':prior_Sigma, 'gibbs_max_iter':gibbs_max_iter, 'gibbs_loglik_eps':gibbs_loglik_eps, 'gibbs_plot_save':gibbs_plot_save}

    # Instantitate bandit    
    bandits.append(MCMCBanditSampling(A, np_reward_function, np_reward_prior, np_thompsonSampling))
    bandits_labels.append('Gibbs-TS nonparametric linear Gaussian mixture')

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
    
    ## Sampling bandits
    # Plot action predictive density
    plot_std=True
    bandits_plot_arm_density(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    
    ## Quantile bandits
    # Plot action quantiles
    plot_std=True
    bandits_plot_arm_quantile(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    ###############
            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    # Example: python3 -m pdb evaluate_linearGaussian_Bandit_nonparametric.py -A 2 -t_max 10 -M 50 -N_max 20 -R 2 -exec_type batch -d_context 2 -type_context randn -theta 1 1 -1 -1 -sigma 1 1
    parser = argparse.ArgumentParser(description='Evaluate Contextual Linear Gaussian Bandits: TS, truncated Gaussian arm sampling and BUCB policies.')
    parser.add_argument('-A', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-M', type=int, default=1000, help='Number of samples for the MC integration')
    parser.add_argument('-N_max', type=int, default=25, help='Maximum number of arm candidate samples')
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
    main(args.A, args.t_max, args.M, args.N_max, args.R, args.exec_type, theta, sigma, args.d_context, args.type_context)
