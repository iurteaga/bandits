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
def main(A, t_max, M, N_max, R, exec_type, dynamic_parameters, d_context, type_context, a_0, lambda_0, c_0):

    ############################### MAIN CONFIG  ############################### 
    print('{}-armed Contextual logistic bandit with TS, sampling policy and BUCB with {} MC samples for {} time-instants and {} realizations'.format(A, M, t_max, R))

    # Directory configuration
    dir_string='../results/{}/A={}/t_max={}/R={}/M={}/N_max={}/d_context={}/type_context={}/{}/a0={}_lambda0={}_c0={}'.format(os.path.basename(__file__).split('.')[0], A, t_max, R, M, N_max, d_context, type_context, dynamic_parameters['type'],a_0, lambda_0, c_0)
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
    reward_function_dynknown={'dynamics':'linear_mixing_known', 'type':'logistic', 'dist':stats.norm, 'theta':dynamic_parameters['theta'], 'dynamics_A':dynamic_parameters['dynamics_A'], 'dynamics_C':dynamic_parameters['dynamics_C']}
    reward_function_dynunknown={'dynamics':'linear_mixing_unknown', 'type':'logistic', 'dist':stats.norm, 'theta':dynamic_parameters['theta'], 'dynamics_A':dynamic_parameters['dynamics_A'], 'dynamics_C':dynamic_parameters['dynamics_C']}
        
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
    mc_reward_prior_dynknown={'dist':'Gaussian', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'sampling':'density', 'sampling_sigma':min_sampling_sigma, 'M':M}
    mc_reward_prior_dynunknown={'dist':'Gaussian', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'A_0':a_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'Lambda_0':lambda_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'nu_0':d_context, 'C_0':c_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'sampling':'density', 'sampling_sigma':min_sampling_sigma, 'M':M}
    ############################### BANDITS  ###############################    
    # Bandits to evaluate as a list
    bandits=[]
    bandits_labels=[]
       
    ### Thompson sampling: when sampling with static n=1 and no MC
    thompsonSampling={'sampling_type':'static', 'MC_type':'MC_arms', 'M':1, 'arm_N_samples':1}
    # Monte-Carlo based, known dynamics
    bandits.append(MCBanditSampling(A, reward_function_dynknown, mc_reward_prior_dynknown, thompsonSampling))
    bandits_labels.append('MC Thompson Sampling, M_theta={}, known dynamics'.format(mc_reward_prior_dynknown['M']))
    # Monte-Carlo based, unknown dynamics
    bandits.append(MCBanditSampling(A, reward_function_dynunknown, mc_reward_prior_dynunknown, thompsonSampling))
    bandits_labels.append('MC Thompson Sampling, M_theta={}, unknown dynamics'.format(mc_reward_prior_dynunknown['M']))

    ### Inverse Pfa sampling
    '''
    # Truncated Gaussian with log10(1/Pfa), MC over arms with provided M and N_max
    invPfaSampling={'sampling_type':'infPfa', 'Pfa':'tGaussian', 'f(1/Pfa)':np.log10, 'MC_type':'MC_arms', 'M': M, 'N_max':N_max}
    # Monte-Carlo based, known dynamics
    bandits.append(MCBanditSampling(A, reward_function_dynknown, mc_reward_prior_dynknown, invPfaSampling))
    bandits_labels.append('MC tGaussian log10(1/Pfa) M_a={}, M_theta={}, known dynamics'.format(invPfaSampling['M'], mc_reward_prior_dynknown['M']))
    # Monte-Carlo based, unknown dynamics
    bandits.append(MCBanditSampling(A, reward_function_dynunknown, mc_reward_prior_dynunknown, invPfaSampling))
    bandits_labels.append('MC tGaussian log10(1/Pfa) M_a={}, M_theta={}, unknown dynamics'.format(invPfaSampling['M'], mc_reward_prior_dynunknown['M']))
    '''

    ### Quantile based, Monte-Carlo, empirical
    alpha=1./np.arange(1,t_max+1)
    # alpha=alpha
    quantileInfo={'MC_alpha':'alpha', 'alpha':alpha, 'type':'empirical', 'n_samples':N_max}
    # Monte-Carlo based, known dynamics
    bandits.append(MCBanditQuantiles(A, reward_function_dynknown, mc_reward_prior_dynknown, quantileInfo))
    bandits_labels.append('MC BUCB, alpha=1/t, M_theta={}, known dynamics'.format(mc_reward_prior_dynknown['M']))
    # Monte-Carlo based, unknown dynamics
    bandits.append(MCBanditQuantiles(A, reward_function_dynunknown, mc_reward_prior_dynunknown, quantileInfo))
    bandits_labels.append('MC BUCB, alpha=1/t, M_theta={}, unknown dynamics'.format(mc_reward_prior_dynunknown['M']))
                               
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
    # Example: python3 -m pdb evaluate_dynamic_logistic_MCBandit_all.py -M 50 -N_max 20 -R 2 -exec_type sequential -dynamic_parameters dynamic_parameters_separated_A2_dcontext2_tmax1500 -type_context static -a_0 0.95 -lambda_0 1.0 -c_0 1.0
    parser = argparse.ArgumentParser(description='Evaluate Dynamic Contextual Contextual Bandits: TS, truncated Gaussian arm sampling and BUCB policies.')
    parser.add_argument('-M', type=int, default=1000, help='Number of samples for the MC integration')
    parser.add_argument('-N_max', type=int, default=25, help='Maximum number of arm candidate samples')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-exec_type', type=str, default='sequential', help='Type of execution to run: batch or sequential')
    parser.add_argument('-dynamic_parameters', type=str, help='File with dynamic parameter evolution details')
    parser.add_argument('-type_context', type=str, default='static', help='Type of context: static (default), randn, rand')
    parser.add_argument('-a_0', type=float, default=1., help='Prior a_0')
    parser.add_argument('-lambda_0', type=float, default=1., help='Prior lambda_0')
    parser.add_argument('-c_0', type=float, default=1., help='Prior c_0')
    
    # Get arguments
    args = parser.parse_args()

    # Load dynamic reward parameter info
    with open('./{}.pickle'.format(args.dynamic_parameters), 'rb') as f:
        dynamic_parameters=pickle.load(f)
        
    # Figure out dimensionalities
    A,d_context,t_max=dynamic_parameters['theta'].shape

    # Call main function
    main(A, t_max, args.M, args.N_max, args.R, args.exec_type, dynamic_parameters, d_context, args.type_context, args.a_0, args.lambda_0, args.c_0)
