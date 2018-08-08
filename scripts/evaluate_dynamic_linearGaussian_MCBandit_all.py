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
def main(A, t_max, M, N_max, R, exec_type, dynamic_parameters, d_context, type_context, sigma, a_0, lambda_0, c_0):

    ############################### MAIN CONFIG  ############################### 
    print('{}-armed Dynamic Contextual Linear Gaussian bandit with TS, truncated Gaussian arm sampling policy with {} MC samples and BUCB for {} time-instants and {} realizations'.format(A, M, t_max, R))

    # Directory configuration
    dir_string='../results/{}/A={}/t_max={}/R={}/M={}/N_max={}/d_context={}/type_context={}/{}/sigma={}/a0={}_lambda0={}_c0={}'.format(os.path.basename(__file__).split('.')[0], A, t_max, R, M, N_max, d_context, type_context, dynamic_parameters['type'],  '_'.join(str.strip(np.array_str(sigma.flatten()),'[]').split()),a_0, lambda_0, c_0)
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
    reward_function_dynknown={'dynamics':'linear_mixing_known', 'type':'linear_gaussian', 'dist':stats.norm, 'theta':dynamic_parameters['theta'], 'dynamics_A':dynamic_parameters['dynamics_A'], 'dynamics_C':dynamic_parameters['dynamics_C'], 'sigma':sigma}
    reward_function_dynunknown={'dynamics':'linear_mixing_unknown', 'type':'linear_gaussian', 'dist':stats.norm, 'theta':dynamic_parameters['theta'], 'dynamics_A':dynamic_parameters['dynamics_A'], 'dynamics_C':dynamic_parameters['dynamics_C'], 'sigma':sigma}
        
    # Reward prior
    Sigmas=np.zeros((A, d_context, d_context))
    for a in np.arange(A):
        Sigmas[a,:,:]=np.eye(d_context)
    
    reward_prior_sknown={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas}
    reward_prior_sunknown={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'alpha':np.ones((A,1)), 'beta':np.ones((A,1))}
    # MC Sampling
    # MC mininimum sampling uncertainty
    min_sampling_sigma=0.000001
    mc_reward_prior_sknown_dynknown={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'sampling':'density', 'sampling_sigma':min_sampling_sigma, 'M':M}
    mc_reward_prior_sunknown_dynknown={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'alpha':np.ones((A,1)), 'beta':np.ones((A,1)), 'sampling':'density', 'sampling_sigma':min_sampling_sigma, 'M':M}
    mc_reward_prior_sknown_dynunknown={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'A_0':a_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'Lambda_0':lambda_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'nu_0':d_context, 'C_0':c_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'sampling':'density', 'sampling_sigma':min_sampling_sigma, 'M':M}
    mc_reward_prior_sunknown_dynunknown={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'A_0':a_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'Lambda_0':lambda_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'nu_0':d_context, 'C_0':c_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'alpha':np.ones((A,1)), 'beta':np.ones((A,1)), 'sampling':'density', 'sampling_sigma':min_sampling_sigma, 'M':M}
    ############################### BANDITS  ###############################    
    # Bandits to evaluate as a list
    bandits=[]
    bandits_labels=[]
    
    ### Thompson sampling: when sampling with static n=1 and no MC
    thompsonSampling={'sampling_type':'static', 'MC_type':'MC_arms', 'M':1, 'arm_N_samples':1}

    # Known sigma
    bandits.append(BayesianBanditSampling(A, reward_function_dynknown, reward_prior_sknown, thompsonSampling))
    bandits_labels.append('Thompson Sampling (known $\sigma_y^2$), known dynamics')

    # Unknown sigma
    bandits.append(BayesianBanditSampling(A, reward_function_dynknown, reward_prior_sunknown, thompsonSampling))
    bandits_labels.append('Thompson Sampling (unknown $\sigma_y^2$), known dynamics')

    # Monte-Carlo based
    # Known sigma, known dynamics
    bandits.append(MCBanditSampling(A, reward_function_dynknown, mc_reward_prior_sknown_dynknown, thompsonSampling))
    bandits_labels.append('MC Thompson Sampling (known $\sigma_y^2$), known dynamics, M_theta={}'.format(mc_reward_prior_sknown_dynknown['M']))
    # Known sigma, unknown dynamics
    bandits.append(MCBanditSampling(A, reward_function_dynunknown, mc_reward_prior_sknown_dynunknown, thompsonSampling))
    bandits_labels.append('MC Thompson Sampling (known $\sigma_y^2$), unknown dynamics, M_theta={}'.format(mc_reward_prior_sknown_dynunknown['M']))
    # Unknown sigma, known dynamics
    bandits.append(MCBanditSampling(A, reward_function_dynknown, mc_reward_prior_sunknown_dynknown, thompsonSampling))
    bandits_labels.append('MC Thompson Sampling (unknown $\sigma_y^2$), known dynamics, M_theta={}'.format(mc_reward_prior_sunknown_dynknown['M']))
    # Unknown sigma, unknown dynamics
    bandits.append(MCBanditSampling(A, reward_function_dynunknown, mc_reward_prior_sunknown_dynunknown, thompsonSampling))
    bandits_labels.append('MC Thompson Sampling (unknown $\sigma_y^2$), unknown dynamics, M_theta={}'.format(mc_reward_prior_sunknown_dynunknown['M']))

    ### Inverse Pfa sampling
    '''
    # Truncated Gaussian with log10(1/Pfa), MC over arms with provided M and N_max
    invPfaSampling={'sampling_type':'infPfa', 'Pfa':'tGaussian', 'f(1/Pfa)':np.log10, 'MC_type':'MC_arms', 'M': M, 'N_max':N_max}

    # Known sigma, known dynamics
    bandits.append(BayesianBanditSampling(A, reward_function_dynknown, reward_prior_sknown, invPfaSampling))
    bandits_labels.append('tGaussian (known $\sigma_y^2$), known dynamics: log10(1/Pfa), M_a={}'.format(invPfaSampling['M']))
    # Unknown sigma, known dynamics
    bandits.append(BayesianBanditSampling(A, reward_function_dynknown, reward_prior_sunknown, invPfaSampling))
    bandits_labels.append('tGaussian (unknown $\sigma_y^2$), known dynamics: log10(1/Pfa), M_a={}'.format(invPfaSampling['M']))
    
    # Monte-Carlo based
    # Known sigma, known dynamics
    bandits.append(MCBanditSampling(A, reward_function_dynknown, mc_reward_prior_sknown_dynknown, invPfaSampling))
    bandits_labels.append('MC tGaussian (known $\sigma_y^2$), known dynamics: log10(1/Pfa), M_a={}, M_theta={}'.format(invPfaSampling['M'], mc_reward_prior_sknown_dynknown['M']))
    # Known sigma, unknown dynamics
    bandits.append(MCBanditSampling(A, reward_function_dynunknown, mc_reward_prior_sknown_dynunknown, invPfaSampling))
    bandits_labels.append('MC tGaussian (known $\sigma_y^2$), unknown dynamics: log10(1/Pfa), M_a={}, M_theta={}'.format(invPfaSampling['M'], mc_reward_prior_sknown_dynunknown['M']))
    # Unknown sigma, known dynamics
    bandits.append(MCBanditSampling(A, reward_function_dynknown, mc_reward_prior_sunknown_dynknown, invPfaSampling))
    bandits_labels.append('MC tGaussian (unknown $\sigma_y^2$), known dynamics: log10(1/Pfa), M_a={}, M_theta={}'.format(invPfaSampling['M'], mc_reward_prior_sunknown_dynknown['M']))
    # Unknown sigma, unknown dynamics
    bandits.append(MCBanditSampling(A, reward_function_dynunknown, mc_reward_prior_sunknown_dynunknown, invPfaSampling))
    bandits_labels.append('MC tGaussian (unknown $\sigma_y^2$), unknown dynamics: log10(1/Pfa), M_a={}, M_theta={}'.format(invPfaSampling['M'], mc_reward_prior_sunknown_dynunknown['M']))
    '''
    
    ### Quantile based
    alpha=1./np.arange(1,t_max+1)
    quantileInfo={'alpha':alpha, 'type':'analytical'}
    
    # Known sigma, known dynamics
    bandits.append(BayesianBanditQuantiles(A, reward_function_dynknown, reward_prior_sknown, quantileInfo))
    bandits_labels.append('BUCB (known $\sigma_y^2$), known dynamics, alpha=1/t')
    # Unknown sigma, known dynamics
    bandits.append(BayesianBanditQuantiles(A, reward_function_dynknown, reward_prior_sunknown, quantileInfo))
    bandits_labels.append('BUCB (unknown $\sigma_y^2$), known dynamics, alpha=1/t')
    
    # MC Empirical type
    quantileInfo={'MC_alpha':'alpha', 'alpha':alpha, 'type':'empirical'}
    
    # Known sigma, known dynamics
    bandits.append(MCBanditQuantiles(A, reward_function_dynknown, mc_reward_prior_sknown_dynknown, quantileInfo))
    bandits_labels.append('MC-BUCB (known $\sigma_y^2$), known dynamics, M_theta={}, alpha=1/t'.format(mc_reward_prior_sknown_dynknown['M']))
    # Known sigma, unknown dynamics
    bandits.append(MCBanditQuantiles(A, reward_function_dynunknown, mc_reward_prior_sknown_dynunknown, quantileInfo))
    bandits_labels.append('MC-BUCB (known $\sigma_y^2$), unknown dynamics, M_theta={}, alpha=1/t'.format(mc_reward_prior_sknown_dynunknown['M']))
    # Unknown sigma, known dynamics
    bandits.append(MCBanditQuantiles(A, reward_function_dynknown, mc_reward_prior_sunknown_dynknown, quantileInfo))
    bandits_labels.append('MC-BUCB (unknown $\sigma_y^2$), known dynamics, M_theta={}, alpha=1/t'.format(mc_reward_prior_sunknown_dynknown['M']))
    # Unknown sigma, unknown dynamics
    bandits.append(MCBanditQuantiles(A, reward_function_dynunknown, mc_reward_prior_sunknown_dynunknown, quantileInfo))
    bandits_labels.append('MC-BUCB (unknown $\sigma_y^2$), unknown dynamics, M_theta={}, alpha=1/t'.format(mc_reward_prior_sunknown_dynunknown['M']))
                  
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
    bandits_colors=[colors.cnames['black'], colors.cnames['silver'], colors.cnames['darkred'], colors.cnames['red'], colors.cnames['salmon'], colors.cnames['orange'], colors.cnames['purple'], colors.cnames['fuchsia'], colors.cnames['navy'], colors.cnames['blue'], colors.cnames['cyan'], colors.cnames['lightblue'], colors.cnames['saddlebrown'], colors.cnames['peru'], colors.cnames['green'], colors.cnames['lime'], colors.cnames['gold'], colors.cnames['yellow']]
        
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
    # Example: python3 -m pdb evaluate_dynamic_linearGaussian_MCBandit_all.py -M 50 -N_max 20 -R 2 -exec_type sequential -dynamic_parameters dynamic_parameters_separated_A2_dcontext2_tmax1500 -type_context static -sigma 0.1 0.1 -a_0 0.95 -lambda_0 1.0 -c_0 1.0
    parser = argparse.ArgumentParser(description='Evaluate Dynamic Contextual Linear Gaussian Bandits: TS, truncated Gaussian arm sampling and BUCB policies.')
    parser.add_argument('-M', type=int, default=1000, help='Number of samples for the MC integration')
    parser.add_argument('-N_max', type=int, default=25, help='Maximum number of arm candidate samples')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-exec_type', type=str, default='sequential', help='Type of execution to run: batch or sequential')
    parser.add_argument('-dynamic_parameters', type=str, help='File with dynamic parameter evolution details')
    parser.add_argument('-type_context', type=str, default='static', help='Type of context: static (default), randn, rand')
    parser.add_argument('-sigma', nargs='+', type=float, default=None, help='Reward scale per arm')
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

    # Emission scale
    if args.sigma != None:
        assert len(args.sigma)==A , 'Not enough emission scale parameters sigma={} provided for A={} '.format(args.sigma, A)
        sigma=np.reshape(args.sigma,A)
    else:
        # Unit variance
        sigma=np.ones(A)

    # Call main function
    main(A, t_max, args.M, args.N_max, args.R, args.exec_type, dynamic_parameters, d_context, args.type_context, sigma, args.a_0, args.lambda_0, args.c_0)
