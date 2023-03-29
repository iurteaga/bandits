# iurteaga

# My imports
import sys, os, shutil, argparse, time
import pickle
import pdb
from itertools import *
import numpy as np
import scipy.stats as stats
import pandas as pd

# Add path and import my Bayesian Bandits
sys.path.append('../src')
from MCBanditSampling import *
from MCMCBanditSampling import *

# Auxiliary function to create bandit data
# Wanted to import showdown data utilities 
# but there are flag related dependencies, so keeping it here
def sample_data(t_max=None, type_context='rand'):
    """Sample simulated linear_gaussian_mixture data'.

    Args:
        num_contexts: Number of contexts to sample.
        type_context: When simulating, what type of context to draw (static, rand, randn)

    Returns:
        dataset: Sampled matrix with rows: (context, reward_1, ..., reward_num_act).
        opt_rewards: Vector of expected optimal reward for each context.
        opt_actions: Vector of optimal action for each context.
        num_actions: Number of available actions.
        context_dim: Dimension of each context.
    """
    # Simulated 'linear_gaussian_mixture'
    ### Example config
    A=2
    K=2
    d_context=2
    pi=np.array(
        [[0.5,0.5],
        [0.3,0.7]])
    theta=np.array(
        [[[1,1],[2,2]],
        [[0,0],[3,3]]])
    sigma=np.array(
        [[1.,1.],
        [1.,1.]])    
    
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

    # Rewards
    rewards=np.zeros((A,t_max))    
    # Need to iterate becaue np.random.multinomial does not implement broadcasting
    for a in np.arange(A):
        # First, pick mixture
        mixture=np.where(np.random.multinomial(1,pi[a], size=t_max))[1]
        # Then draw
        rewards[a,:]=stats.norm.rvs(loc=np.einsum('dt,td->t', np.reshape(context, (d_context, t_max)), np.reshape(theta[a,mixture], (t_max, d_context))), scale=sigma[a,mixture])
    
    # Optimal action with respect to rewards
    #opt_actions=np.argmax(rewards, axis=0)
    # Optimal action with respect to expected rewards
    true_expected_rewards=np.einsum('ak,akd,dt->at', pi, theta, context)
    opt_actions=np.argmax(true_expected_rewards, axis=0)
    opt_rewards=np.amax(rewards,axis=0)
    true_expected_rewards=true_expected_rewards.T
    num_actions=A
    context_dim=d_context
    
    return context, rewards, opt_rewards, opt_actions, true_expected_rewards, num_actions, context_dim


# Main function
def main(t_max, type_context, gibbs_max_iter, gibbs_loglik_eps):
      
    ####################### DATASET #######################
    # Create dataset    
    sampled_vals = sample_data(t_max, type_context)
    contexts, rewards, opt_rewards, opt_actions, true_expected_rewards, num_actions, context_dim = sampled_vals
    # The key data components are
    #   contexts: array of dimensionality context_dim \times t_max
    #   rewards: array of dimensionality num_actions \times t_max
    
    ########### Bayesian Nonparametric bandit Initialization #######################
    # Thompson sampling policy
    thompsonSampling={
                        'sampling_type':'static',
                        'arm_N_samples':1,
                        'M':1,
                        'MC_type':'MC_arms',
                        'mixture_expectation':'pi_expected'
                    }

    # That assumes a mixture of gaussian reward function (might not be true)
    reward_function={
                        'type':'linear_gaussian_mixture',
                        'dist':stats.norm,
                    }
    
    # With BNP Prior hyperparameters
    gamma=0.1
    pitman_yor_d=0
    assert (0<=pitman_yor_d) and (pitman_yor_d<1) and (gamma >-pitman_yor_d)
    
    # Defaults
    alpha=1.
    beta=1.
    sigma=1.
        
    # Concentration parameter, per action
    prior_d=pitman_yor_d*np.ones(num_actions)
    prior_gamma=gamma*np.ones(num_actions)
    # NIG for linear Gaussians
    prior_alpha=alpha*np.ones(num_actions)
    prior_beta=beta*np.ones(num_actions)

    # Initial thetas
    prior_theta=np.ones((num_actions,context_dim))            
    prior_Sigma=np.zeros((num_actions,context_dim, context_dim))
    # Initial covariances: uncorrelated
    for a in np.arange(num_actions):
        prior_Sigma[a,:,:]=sigma*np.eye(context_dim)
    
    # Reward prior as dictionary
    reward_prior={
                    'type':'linear_gaussian_mixture',
                    'dist':'NIG',
                    'K':'nonparametric',
                    'd':prior_d,
                    'gamma':prior_gamma,
                    'alpha':prior_alpha,
                    'beta':prior_beta,
                    'theta':prior_theta,
                    'Sigma':prior_Sigma,
                    'gibbs_max_iter':gibbs_max_iter,
                    'gibbs_loglik_eps':gibbs_loglik_eps,
                    'gibbs_plot_save':None
                    }
                    
    # Initialize BNP-TS agent
    bnp_bandit_agent=MCMCBanditSampling(
        num_actions,
        reward_function,
        reward_prior,
        thompsonSampling
    )
    
    ########### BNP-bandit: interactive execution #######################
    played_arms=np.zeros(t_max, dtype=int)
    observed_rewards=np.zeros(t_max)
    for t in np.arange(t_max):
        played_arms[t]=bnp_bandit_agent.execute_interactive(
            t,
            context=contexts[:,t],
            y=observed_rewards[t-1] if t>0 else None,
            t_max=t_max
        )
        
        # Keep track of the reward for the executed action
        observed_rewards[t]=rewards[played_arms[t],t]
    
    ####################### Evaluation #######################    
    # Based on played_arms and observed_rewards
    #########################################################
    
if __name__ == '__main__':
    # Example run:
    #   python3 evaluate_nonparametric_bandit.py t_max=100 -gibbs_max_iter 5
    
    # Input parser
    # TODO: for some unknown reason, the app and flags packages do not like integer arguments
    parser = argparse.ArgumentParser(description='Evaluate an interactive nonparametric bandit.')
    parser.add_argument('-t_max', type=str, default='5', help='Number of contexts (i.e., time instances) to consider')
    parser.add_argument('-type_context', type=str, default='rand', help='Type of context to draw for simulated data')
    parser.add_argument('-gibbs_max_iter', type=int, default='5', help='Number of gibbs iterations the bandit agent runs per-interaction')
    parser.add_argument('-gibbs_loglik_eps', type=float, default=0.0001, help='Log-likelihood epsilon for the bandit agent per-interaction gibbs convergence')
    # Get arguments
    args = parser.parse_args()
    
    main(
        int(args.t_max),
        args.type_context,
        int(args.gibbs_max_iter),
        args.gibbs_loglik_eps,
    )
