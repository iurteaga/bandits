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
from MCMCBanditSampling import *
from VariationalBanditSampling import *
from plot_Bandits import *

# Main code
def main(A, K, t_max, R, exec_type, pi, theta, sigma, d_context, type_context, prior_K, mixture_expectations):

    ############################### MAIN CONFIG  ############################### 
    print('{}-armed Contextual Linear Gaussian mixture bandit with optimal and sampling policies with Variational inference for {} time-instants and {} realizations'.format(A, t_max, R))

    # Directory configuration
    dir_string='../results/{}/A={}/t_max={}/R={}/d_context={}/type_context={}/pi={}/theta={}/sigma={}/prior_K={}/{}'.format(os.path.basename(__file__).split('.')[0], A, t_max, R, d_context, type_context, str.replace(str.strip(np.array_str(pi.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(theta.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(sigma.flatten()),' []'), '  ', '_'), str.replace(str.strip(np.array_str(prior_K.flatten()),' []'), ' ', '_'), '_'.join(mixture_expectations))
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
    reward_function={'type':'linear_gaussian_mixture', 'dist':stats.norm, 'pi':pi, 'theta':theta, 'sigma':sigma}

    ########## Inference
    # Variational parameters
    variational_max_iter=100
    variational_lb_eps=0.0001
    # Plotting
    variational_plot_save='show'
    variational_plot_save=None
    if variational_plot_save != None and variational_plot_save != 'show':
        # Plotting directories
        variational_plots=dir_string+'/variational_plots'
        os.makedirs(variational_plots, exist_ok=True)
    ########## Priors
    gamma=0.1
    alpha=1.
    beta=1.
    sigma=1.
    
    ############################### BANDITS  ###############################    
    # Bandits to evaluate as a list
    bandits=[]
    bandits_labels=[]

    ### Optimal bandit
    bandits.append(OptimalBandit(A, reward_function))
    bandits_labels.append('Optimal Bandit')
            
    ### Thompson sampling: when sampling with static n=1 and no Monte Carlo
    thompsonSampling={'sampling_type':'static', 'arm_N_samples':1, 'M':1, 'MC_type':'MC_arms'}

    # Mixture expectation
    for mixture_expectation in mixture_expectations:
        thompsonSampling['mixture_expectation']=mixture_expectation
        
        # Different mixture priors
        for this_K in prior_K:
            # New dirs
            if variational_plot_save != None and variational_plot_save != 'show':
                os.makedirs(variational_plots+'/prior_K{}'.format(this_K), exist_ok=True)
                variational_plot_save=variational_plots+'/prior_K{}'.format(this_K)

            # Hyperparameters
            # Dirichlet for mixture weights
            prior_gamma=gamma*np.ones((A,this_K))
            # NIG for linear Gaussians
            prior_alpha=alpha*np.ones((A,this_K))
            prior_beta=beta*np.ones((A,this_K))

            # Initial thetas
            prior_theta=np.ones((A,this_K,d_context))
            # Different initial thetas
            for k in np.arange(this_K):
                prior_theta[:,k,:]=k
                
            prior_Sigma=np.zeros((A, this_K, d_context, d_context))
            # Initial covariances: uncorrelated
            for a in np.arange(A):
                for k in np.arange(this_K):
                    prior_Sigma[a,k,:,:]=sigma*np.eye(d_context)

            # Variational            
            # Reward prior as dictionary: plotting Variational lower bound
            reward_prior={'type':'linear_gaussian_mixture', 'dist':'NIG', 'K':this_K, 'gamma':prior_gamma, 'alpha':prior_alpha, 'beta':prior_beta, 'theta':prior_theta, 'Sigma':prior_Sigma, 'variational_max_iter':variational_max_iter, 'variational_lb_eps':variational_lb_eps, 'variational_plot_save':variational_plot_save}
        
            # Instantitate bandit    
            bandits.append(VariationalBanditSampling(A, reward_function, reward_prior, thompsonSampling))
            bandits_labels.append('VTS, prior_K={}, {}'.format(this_K,mixture_expectation))
                           
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
    
    # Plot cumregret
    plot_std=False
    bandits_plot_cumregret(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_cumregret(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    
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
    # Example: python3 -m pdb evaluate_linearGaussianMixture_BanditSampling_all.py -A 2 -K 2 -t_max 10 -R 2 -exec_type sequential -d_context 2 -type_context randn -pi 0.5 0.5 0.1 0.9 -theta 1 1 0 0 -1 -1 2 2 -sigma 1 1 1 1 -prior_K 2 -mixture_expectations pi_expected
    parser = argparse.ArgumentParser(description='Evaluate Contextual Linear Gaussian Mixture Bandits: Variational inference with sampling policy.')
    parser.add_argument('-A', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-K', type=int, default=2, help='Number of mixtures per arm of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    #parser.add_argument('-M', type=int, default=1000, help='Number of samples for the MC integration')
    #parser.add_argument('-N_max', type=int, default=25, help='Maximum number of arm candidate samples')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-exec_type', type=str, default='sequential', help='Type of execution to run: batch or sequential')
    parser.add_argument('-d_context', type=int, default=2, help='Dimensionality of context')
    parser.add_argument('-type_context', type=str, default='static', help='Type of context: static (default), randn, rand')
    parser.add_argument('-pi', nargs='+', type=float, default=None, help='Mixture proportions per arm')
    parser.add_argument('-theta', nargs='+', type=float, default=None, help='Thetas per arm and mixtures')
    parser.add_argument('-sigma', nargs='+', type=float, default=None, help='Variances per arm and mixtures')
    parser.add_argument('-prior_K', nargs='+', type=int, default=2, help='Assumed prior number of mixtures (per arm)')
    parser.add_argument('-mixture_expectations', nargs='+', type=str, default='pi_expected', help='Mixture expectation type: z_sampling, pi_sampling, pi_expected')

    # Get arguments
    args = parser.parse_args()
    
    # Doublechecking
    # Mixture proportions
    if args.pi != None:
        assert len(args.pi)==args.A * args.K, 'Not enough mixture weights pi={} provided for A={} and K={}'.format(args.pi, args.A, args.K)
        pi=np.reshape(args.pi,(args.A,args.K))
        assert np.all(pi.sum(axis=1)==np.ones(args.A)), 'Mixture proportions pi={} must sum up to one per arm: {}'.format(pi, pi.sum(axis=1))
    else:
        # Random
        pi=np.random.rand(args.A,args.K)
        pi=pi/pi.sum(axis=1)

    # Regressors
    if args.theta != None:
        assert len(args.theta)==args.A*args.K*args.d_context, 'Not enough regression parameters theta={} provided for A={}, K={} and d_context={}'.format(args.theta, args.A, args.K, args.d_context)
        theta=np.reshape(args.theta, (args.A,args.K,args.d_context))
    else:
        # Random
        theta=np.random.randn(args.A,args.K,args.d_context)

    # Emission variance
    if args.sigma != None:
        assert len(args.sigma)==args.A * args.K, 'Not enough emission variance parameters sigma={} provided for A={} and K={}'.format(args.sigma, args.A, args.K)
        sigma=np.reshape(args.sigma,(args.A,args.K))
    else:
        # Unit variance
        sigma=np.ones((args.A,args.K))

    # Call main function
    main(args.A, args.K, args.t_max, args.R, args.exec_type, pi, theta, sigma, args.d_context, args.type_context, np.array(args.prior_K), np.array(args.mixture_expectations))
