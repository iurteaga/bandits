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
def sample_data(data_type, num_contexts=None, type_context='rand'):
    """Sample data from given 'data_type'.

    Args:
        data_type: Dataset from which to sample.
        num_contexts: Number of contexts to sample.
        type_context: When simulating, what type of context to draw (static, rand, randn)

    Returns:
        dataset: Sampled matrix with rows: (context, reward_1, ..., reward_num_act).
        opt_rewards: Vector of expected optimal reward for each context.
        opt_actions: Vector of optimal action for each context.
        num_actions: Number of available actions.
        context_dim: Dimension of each context.
    """
    # Simulated
    if 'linear_gaussian_mixture' in data_type:
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
        
        # Draw context
        t_max=num_contexts
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
        
        # Define objects to return
        dataset=np.concatenate((context, rewards), axis=0).T
        # Optimal action with respect to rewards
        #opt_actions=np.argmax(rewards, axis=0)
        # Optimal action with respect to expected rewards
        true_expected_rewards=np.einsum('ak,akd,dt->at', pi, theta, context)
        opt_actions=np.argmax(true_expected_rewards, axis=0)
        opt_rewards=np.amax(rewards,axis=0)
        true_expected_rewards=true_expected_rewards.T
        num_actions=A
        context_dim=d_context
    
    else:
        raise ValueError('Unknown data_type {}'.format(data_type))
    
    return dataset, opt_rewards, opt_actions, true_expected_rewards, num_actions, context_dim


# Main function
def main(data_type, num_contexts, type_context, gibbs_max_iter, gammas, r):
      
    # Directories
    this_results_dir='../results/{}_t{}_{}context_gibbsmaxiter{}_gammas{}'.format(
                        data_type,
                        num_contexts,
                        type_context,
                        gibbs_max_iter,
                        '_'.join([str(gamma) for gamma in gammas])
                    )
    os.makedirs(this_results_dir, exist_ok=True)
    
    ####################### DATASET #######################
    # Create dataset    
    sampled_vals = sample_data(data_type, num_contexts, type_context)
    dataset, opt_rewards, opt_actions, true_expected_rewards, num_actions, context_dim = sampled_vals
    
    ####################### Nonparametric bandit #######################
    # My bandits
    # Context
    context=dataset[:,:context_dim].T
    assert context.shape==(context_dim, num_contexts)
    # Rewards
    rewards=dataset[:,context_dim:].T
    # True expected rewards per action
    assert (rewards.shape==true_expected_rewards.T.shape)
    opt_true_expected_rewards=true_expected_rewards.T[opt_actions,np.arange(num_contexts)]
    # Empirical mean rewards per action
    mean_rewards=rewards.mean(axis=1)
    # Optimal mean rewards
    opt_mean_rewards=mean_rewards[opt_actions]
    assert rewards.shape==(num_actions, num_contexts)
    # Reward function
    reward_function={
                        'type':'linear_gaussian_mixture',
                        'dist':stats.norm,
                        'preloaded_bandit_rewards':rewards
                    }    
    
    # My bandits to evaluate as a list
    my_bandits=[]
    my_bandits_labels=[]
    ### Thompson sampling policy
    thompsonSampling={
                        'sampling_type':'static',
                        'arm_N_samples':1,
                        'M':1,
                        'MC_type':'MC_arms',
                        'mixture_expectation':'pi_expected'
                    }

    # MCMC (Gibbs) inference parameters for nonparametric bandit
    #gibbs_max_iter is provided as argument
    gibbs_loglik_eps=0.0001
    
    # Evaluate with different gammas
    for gamma in gammas:
        # Hyperparameter priors
        # gamma=as argument!
        pitman_yor_d=0
        assert (0<=pitman_yor_d) and (pitman_yor_d<1) and (gamma >-pitman_yor_d)
        # Default wide
        alpha=1.
        beta=1.
        sigma=1.
        
        # Concentration parameter
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
        
        # Instantitate bandit
        my_bandits.append(MCMCBanditSampling(num_actions, reward_function, reward_prior, thompsonSampling))
        my_bandits_labels.append('Nonparametric-TS gamma={}'.format(gamma))
        
    # Execute each bandit once, sequentially
    R=1
    exec_type='sequential'
    my_rewards=np.nan*np.ones((num_contexts, len(my_bandits)))
    my_actions=np.nan*np.ones((num_contexts, len(my_bandits)))
    my_time=np.zeros(len(my_bandits))
    
    for (n_bandit,bandit) in enumerate(my_bandits):
        t_init=time.time()
        bandit.execute_realizations(R, num_contexts, context, exec_type)
        # Only for played arms
        played_arms=bandit.actions.sum(axis=0)>0
        my_rewards[played_arms,n_bandit]=bandit.rewards.T[played_arms].sum(axis=1)
        my_actions[played_arms,n_bandit]=np.where(bandit.actions.T[played_arms])[1]
        my_time[n_bandit]=time.time()-t_init
    
    ####################### Evaluation #######################
    # Save results
    with open('{}/bandits_results_r{}.npz'.format(this_results_dir, r), 'wb') as f:
        np.savez_compressed(f,
                            labels=my_bandits_labels,
                            times=my_time,
                            actions=my_actions,
                            rewards=my_rewards,
                            regrets=opt_rewards[:,None]-my_rewards,
                            opt_rewards=opt_rewards,
                            opt_true_expected_rewards=opt_true_expected_rewards,
                            opt_mean_rewards=opt_mean_rewards,
                            opt_actions=opt_actions
                            )
    
    #########################################################
    
if __name__ == '__main__':
    # Example run:
    #   python3 evaluate_nonparametric_bandit.py -data_type linear_gaussian_mixture -num_contexts 150 -gibbs_max_iter 5 -gammas 1
    
    # Input parser
    # TODO: for some unknown reason, the app and flags packages do not like integer arguments
    parser = argparse.ArgumentParser(description='Evaluate bandits based on logged data.')
    parser.add_argument('-data_type', type=str, default='linear', help='Data type to run algos with')
    parser.add_argument('-num_contexts', type=str, default='5', help='Number of contexts (i.e., time instances) to consider')
    parser.add_argument('-type_context', type=str, default='rand', help='Type of context to draw for simulated data')
    parser.add_argument('-gibbs_max_iter', type=int, default='5', help='Number of gibbs iterations to run')
    parser.add_argument('-gammas', nargs='+', type=float, default=None, help='Gamma parameters for nonparametric model')
    parser.add_argument('-r', type=str, default='random', help='Realization identifier')
    
    # Get arguments
    args = parser.parse_args()
    
    # Make sure A and theta size match
    if args.gammas != None:
        gammas=np.array(args.gammas)
    else:
        # Default
        gammas=np.array([0.1,1,10])
    
    if args.r == 'random':
        r=np.random.randn()
    else:
        r=int(args.r)
            
    main(args.data_type, int(args.num_contexts), args.type_context, int(args.gibbs_max_iter), gammas, r)
