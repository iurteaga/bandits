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
from MCBanditSampling import *
# Quantiles
from MCBanditQuantiles import *

# Main code
def main(A, t_max, M, N_max, R, exec_type, theta, d_context, type_context):

    ############################### MAIN CONFIG  ############################### 
    print('{}-armed Contextual logistic bandit with TS and arm sampling policies with {} MC samples for {} time-instants and {} realizations'.format(A, M, t_max, R))

    # Directory configuration
    dir_string='../results/{}/A={}/t_max={}/R={}/M={}/N_max={}/d_context={}/type_context={}/theta={}'.format(os.path.basename(__file__).split('.')[0], A, t_max, R, M, N_max, d_context, type_context, '_'.join(str.strip(np.array_str(theta.flatten()),'[]').split()))
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
    reward_function={'type':'logistic', 'dist':stats.norm, 'theta':theta}
    # Reward prior
    Sigmas=np.zeros((A, d_context, d_context))
    for a in np.arange(A):
        Sigmas[a,:,:]=np.eye(d_context)

    # MC Sampling
    # MC mininimum sampling uncertainty
    min_sampling_sigma=0.000001
    #mc_reward_prior={'dist':'Gaussian', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'sampling':'resampling', 'M':M}
    #mc_reward_prior={'dist':'Gaussian', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'sampling':'random_walk', 'sampling_sigma':min_sampling_sigma, 'M':M}
    #mc_reward_prior={'dist':'Gaussian', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'sampling':'kernel', 'sampling_alpha':0.9, 'M':M}
    mc_reward_prior={'dist':'Gaussian', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'sampling':'density', 'sampling_sigma':min_sampling_sigma, 'M':M}
    
    ############################### BANDITS  ###############################    
    # Bandits to evaluate as a list
    bandits=[]
    bandits_labels=[]
       
    ### Thompson sampling: when sampling with static n=1 and no MC
    thompsonSampling={'sampling_type':'static', 'MC_type':'MC_arms', 'M':1, 'arm_N_samples':1}
    # Monte-Carlo based
    bandits.append(MCBanditSampling(A, reward_function, mc_reward_prior, thompsonSampling))
    bandits_labels.append('MC Thompson Sampling, M_theta={}'.format(mc_reward_prior['M']))

    ### Inverse Pfa sampling
    # Truncated Gaussian with log10(1/Pfa), MC over arms with provided M and N_max
    invPfaSampling={'sampling_type':'infPfa', 'Pfa':'tGaussian', 'f(1/Pfa)':np.log10, 'MC_type':'MC_arms', 'M': M, 'N_max':N_max}
    # Monte-Carlo based
    bandits.append(MCBanditSampling(A, reward_function, mc_reward_prior, invPfaSampling))
    bandits_labels.append('MC tGaussian log10(1/Pfa) M_a={}, M_theta={}'.format(invPfaSampling['M'], mc_reward_prior['M']))
    
    # Chebyshev with log10(1/Pfa), MC over arms with provided M and N_max
    invPfaSampling={'sampling_type':'infPfa', 'Pfa':'Chebyshev', 'f(1/Pfa)':np.log10, 'MC_type':'MC_arms', 'M': M, 'N_max':N_max}
    bandits.append(MCBanditSampling(A, reward_function, mc_reward_prior, invPfaSampling))
    bandits_labels.append('MC Chebyshev log10(1/Pfa), M_a={}, M_theta={}'.format(invPfaSampling['M'], mc_reward_prior['M']))
    
    # Markov with log10(1/Pfa), MC over arms with provided M and N_max
    invPfaSampling={'sampling_type':'infPfa', 'Pfa':'Markov', 'f(1/Pfa)':np.log10, 'MC_type':'MC_arms', 'M': M, 'N_max':N_max}
    bandits.append(MCBanditSampling(A, reward_function, mc_reward_prior, invPfaSampling))
    bandits_labels.append('MC Markov log10(1/Pfa), M_a={}, M_theta={}'.format(invPfaSampling['M'], mc_reward_prior['M']))
                           
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
    bandits_colors=[colors.cnames['black'], colors.cnames['red'], colors.cnames['blue'], colors.cnames['green'], colors.cnames['yellow'], colors.cnames['cyan'], colors.cnames['purple'], colors.cnames['fuchsia'], colors.cnames['saddlebrown'], colors.cnames['peru']]
        
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
    ###############
            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    # Example: python3 -m pdb evaluate_logistic_MCBanditSampling_all.py -A 2 -t_max 10 -M 50 -N_max 20 -R 2 -exec_type batch -d_context 2 -type_context randn -theta 1 1 -1 -1 
    parser = argparse.ArgumentParser(description='Evaluate Contextual Logistic Bandits and MC: TS and all arm sampling policies.')
    parser.add_argument('-A', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-M', type=int, default=1000, help='Number of samples for the MC integration')
    parser.add_argument('-N_max', type=int, default=25, help='Maximum number of arm candidate samples')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-exec_type', type=str, default='sequential', help='Type of execution to run: batch or sequential')
    parser.add_argument('-d_context', type=int, default=2, help='Dimensionality of context')
    parser.add_argument('-type_context', type=str, default='static', help='Type of context: static (default), randn, rand')
    parser.add_argument('-theta', nargs='+', type=float, default=None, help='Thetas per arm')

    # Get arguments
    args = parser.parse_args()
    
    # Regressors
    if args.theta != None:
        assert len(args.theta)==args.A*args.d_context, 'Not enough regression parameters theta={} provided for A={} and d_context={}'.format(args.theta, args.A, args.d_context)
        theta=np.reshape(args.theta, (args.A,args.d_context))
    else:
        # Random
        theta=np.random.randn(args.A,args.d_context)

    # Call main function
    main(args.A, args.t_max, args.M, args.N_max, args.R, args.exec_type, theta, args.d_context, args.type_context)
