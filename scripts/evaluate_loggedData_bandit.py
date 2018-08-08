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
# Sampling Bandits
from BayesianBanditSampling import *
from MCBanditSampling import *
from MCMCBanditSampling import *
from VariationalBanditSampling import *

# Main code
def main(logged_data_file, reward_model, context_type, policy, n_permutations, R):

    ############################### MAIN CONFIG  ############################### 
    print('Evaluating logged data {} with reward_model={}, policy={} for {} permutations and {} realizations'.format(logged_data_file, reward_model, policy, n_permutations, R))

    # Directory configuration
    dir_string='../results/evaluate_logged_data/{}/{}/context={}/{}/nperm={}/R={}/'.format(logged_data_file.split('/')[-1], reward_model, context_type, policy, n_permutations, R)
    os.makedirs(dir_string, exist_ok=True)
    
    ########## Bandit configuration ##########
    if policy=='TS':
        ### Thompson sampling
        thompsonSampling={'sampling_type':'static', 'arm_N_samples':1, 'M':1, 'MC_type':'MC_arms'}
    else:
        raise ValueError('Policy {} not implemented'.format(policy))
    
    # Load logged data: it should be a dataframe
    with open('{}.pickle'.format(logged_data_file), 'rb') as f:
        logged_dataframe=pickle.load(f)

        # Retrieve logged data
        if logged_data_file.split('/')[2]=='Webscope_R6B':
            A=logged_dataframe['displayed_article'].cat.categories.size
            logged_arms=logged_dataframe['displayed_article'].cat.codes.values
            logged_rewards=logged_dataframe['click'].values
            logged_context=np.vstack(logged_dataframe['user_features'].values).T
        elif logged_data_file.split('/')[2]=='R6':
            A=logged_dataframe['displayed_article'].cat.categories.size
            logged_arms=logged_dataframe['displayed_article'].cat.codes.values
            logged_rewards=logged_dataframe['click'].values
            logged_context=np.vstack(logged_dataframe['user_features'].values).T
            if context_type=='user_articles':
                # Load article set and features for this day
                with open('{}.pickle'.format(logged_data_file.replace('_bandit_','_article_set_')), 'rb') as f:
                        article_set=pickle.load(f)

                # Get features of each relevant article
                article_set_features=np.zeros((A,6))
                for (a_idx, article) in enumerate(logged_dataframe['displayed_article'].cat.categories):
                    article_set_features[a_idx,:]=article_set[article]
                
                # In https://dl.acm.org/citation.cfm?doid=1772690.1772758 they propose
                #z_a=np.einsum('ab, dt->abdt', article_set_features, logged_context)
                # We decide to compute correlation between user and article features via dot product
                z_ua=np.einsum('ad, dt->at', article_set_features, logged_context)
                # And concatenate them with user features
                logged_context=np.vstack([logged_context, z_ua])
                    
        # Figure out dimensionality
        d_context,t_max=logged_context.shape
        assert (t_max==logged_arms.size) and (t_max==logged_rewards.size)
        if context_type=='none':
            d_context=1
        
        # Number of models
        reward_models=reward_model.split(',')
        n_reward_models=len(reward_models)
        
        # Evaluation is slightly different due to logged data (NaNs)
        # Realizations are run to account for stochastic nature of policies
        # Permutations of data are run to account for influence of history in policies: different order of observed rewards
        actions=np.zeros((n_reward_models, n_permutations, R, A, t_max))
        regrets=np.zeros((n_reward_models, n_permutations, R, t_max))
        rewards=np.zeros((n_reward_models, n_permutations, R, t_max))
        
        # Shuffle dataset n_permutation times
        for n_permutation in np.arange(n_permutations):
            print('Permutation={}/{}'.format(n_permutation,n_permutations)) 
            logged_data_indexes=np.random.permutation(t_max)
            # Reward function populated with logged data
            reward_function={'logged_data':True, 'logged_arms': logged_arms[logged_data_indexes], 'logged_rewards': logged_rewards[logged_data_indexes]}
            # Bandits as a list
            bandits=[]
            bandits_labels=[]
            # Bandit reward model and prior configurations
            for this_reward_model in reward_models:
                if this_reward_model=='bernoulli':
                    this_reward_function={k:v for (k,v) in reward_function.items()}    
                    this_reward_function['type']='bernoulli'
                    reward_prior={'dist': stats.beta, 'alpha': np.ones((A,1)), 'beta': np.ones((A,1))}
                    # Instantiate bandit
                    bandits.append(BayesianBanditSampling(A, this_reward_function, reward_prior, thompsonSampling))
                    bandits_labels.append('Bernoulli TS')              

                elif this_reward_model=='linearGaussian':
                    this_reward_function={k:v for (k,v) in reward_function.items()}
                    this_reward_function['type']='linear_gaussian'
                    Sigmas=np.zeros((A, d_context, d_context))
                    for a in np.arange(A):
                        Sigmas[a,:,:]=np.eye(d_context)
                    reward_prior={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'alpha':np.ones((A,1)), 'beta':np.ones((A,1))}
                    # Instantiate bandit
                    bandits.append(BayesianBanditSampling(A, this_reward_function, reward_prior, thompsonSampling))
                    bandits_labels.append('Linear Gaussian TS')

                elif this_reward_model=='linearGaussian_dynamic':
                    this_reward_function={k:v for (k,v) in reward_function.items()}
                    this_reward_function['type']='linear_gaussian'
                    Sigmas=np.zeros((A, d_context, d_context))
                    for a in np.arange(A):
                        Sigmas[a,:,:]=np.eye(d_context)
                    # MC Sampling
                    min_sampling_sigma=0.001
                    M=1000
                    a_0=1.0
                    lambda_0=1.0
                    c_0=1.0
                    reward_prior={'dist':'NIG', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'A_0':a_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'Lambda_0':lambda_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'nu_0':d_context, 'C_0':c_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'alpha':np.ones((A,1)), 'beta':np.ones((A,1)), 'sampling':'density', 'sampling_sigma':min_sampling_sigma, 'M':M}
                    # Instantiate bandit
                    bandits.append(MCBanditSampling(A, this_reward_function, reward_prior, thompsonSampling))
                    bandits_labels.append('Linear Gaussian dynamic TS, M={}'.format(reward_prior['M']))

                elif this_reward_model=='logistic':
                    this_reward_function={k:v for (k,v) in reward_function.items()}
                    this_reward_function['type']='logistic'
                    Sigmas=np.zeros((A, d_context, d_context))
                    for a in np.arange(A):
                        Sigmas[a,:,:]=np.eye(d_context)
                    # MC Sampling
                    min_sampling_sigma=0.001
                    M=1000
                    reward_prior={'dist':'Gaussian', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'sampling':'density', 'sampling_sigma':min_sampling_sigma, 'M':M}
                    # Instantiate bandit
                    bandits.append(MCBanditSampling(A, this_reward_function, reward_prior, thompsonSampling))
                    bandits_labels.append('Logistic MC-TS, M={}'.format(reward_prior['M']))

                elif reward_model=='logistic_dynamic':
                    this_reward_function={k:v for (k,v) in reward_function.items()}
                    this_reward_function['type']='logistic'
                    Sigmas=np.zeros((A, d_context, d_context))
                    for a in np.arange(A):
                        Sigmas[a,:,:]=np.eye(d_context)
                    # MC Sampling
                    min_sampling_sigma=0.001
                    M=1000
                    a_0=1.0
                    lambda_0=1.0
                    c_0=1.0
                    reward_prior={'dist':'Gaussian', 'theta':np.ones((A,d_context)), 'Sigma':Sigmas, 'A_0':a_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'Lambda_0':lambda_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'nu_0':d_context, 'C_0':c_0*np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'sampling':'density', 'sampling_sigma':min_sampling_sigma, 'M':M}
                    # Instantiate bandit
                    bandits.append(MCBanditSampling(A, this_reward_function, reward_prior, thompsonSampling))
                    bandits_labels.append('Logistic dynamic TS, M={}'.format(reward_prior['M']))
            
                elif 'linearGaussianMixture' in this_reward_model:
                    this_reward_function={k:v for (k,v) in reward_function.items()}
                    this_reward_function['type']='linear_gaussian_mixture'
                    if this_reward_model.split('_')[1]=='nonparametric':
                        # Nonparametric with MCMC
                        # Gibbs parameters
                        gibbs_max_iter=10
                        gibbs_loglik_eps=0.01
                        gibbs_plot_save=None
                        ########## Priors
                        gamma=0.1
                        alpha=1.
                        beta=1.
                        sigma=1.
                        pitman_yor_d=0
                        assert (0<=pitman_yor_d) and (pitman_yor_d<1) and (gamma >-pitman_yor_d)
                        mixture_expectation='pi_expected'
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
                            prior_Sigma[a,:,:]=sigma*np.eye(d_context)
                        # Reward prior as dictionary
                        reward_prior={'type':'linear_gaussian_mixture', 'dist':'NIG', 'K':'nonparametric', 'd':prior_d, 'gamma':prior_gamma, 'alpha':prior_alpha, 'beta':prior_beta, 'theta':prior_theta, 'Sigma':prior_Sigma, 'gibbs_max_iter':gibbs_max_iter, 'gibbs_loglik_eps':gibbs_loglik_eps, 'gibbs_plot_save':gibbs_plot_save}
                    
                        # Instantitate bandit    
                        # Policy
                        thompsonSampling['mixture_expectation']=mixture_expectation
                        bandits.append(MCMCBanditSampling(A, this_reward_function, reward_prior, thompsonSampling))
                        bandits_labels.append('Gibbs-TS, nonparametric, {}'.format(mixture_expectation))
                    else:
                        ########## Priors
                        gamma=0.1
                        alpha=1.
                        beta=1.
                        sigma=1.
                        this_K=np.int64(this_reward_model.split('K')[1])
                        mixture_expectation='pi_expected'
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
                        ########## Inference
                        if this_reward_model.split('_')[1]=='MCMC':
                            # Gibbs parameters
                            gibbs_max_iter=10
                            gibbs_loglik_eps=0.01
                            gibbs_plot_save=None
                            # Reward prior as dictionary
                            reward_prior={'type':'linear_gaussian_mixture', 'dist':'NIG', 'K':this_K, 'gamma':prior_gamma, 'alpha':prior_alpha, 'beta':prior_beta, 'theta':prior_theta, 'Sigma':prior_Sigma, 'gibbs_max_iter':gibbs_max_iter, 'gibbs_loglik_eps':gibbs_loglik_eps, 'gibbs_plot_save':gibbs_plot_save}                
                            # Instantitate bandit    
                            bandits.append(MCMCBanditSampling(A, this_reward_function, reward_prior, thompsonSampling))
                            bandits_labels.append('Gibbs-TS, prior_K={}, {}'.format(this_K,mixture_expectation))

                        if this_reward_model.split('_')[1]=='variational':
                            # Variational parameters
                            variational_max_iter=100
                            variational_lb_eps=0.0001
                            variational_plot_save=None
                            # Reward prior as dictionary: plotting Variational lower bound
                            reward_prior={'type':'linear_gaussian_mixture', 'dist':'NIG', 'K':this_K, 'gamma':prior_gamma, 'alpha':prior_alpha, 'beta':prior_beta, 'theta':prior_theta, 'Sigma':prior_Sigma, 'variational_max_iter':variational_max_iter, 'variational_lb_eps':variational_lb_eps, 'variational_plot_save':variational_plot_save}
                            # Instantitate bandit    
                            # Policy
                            thompsonSampling['mixture_expectation']=mixture_expectation
                            bandits.append(VariationalBanditSampling(A, this_reward_function, reward_prior, thompsonSampling))
                            bandits_labels.append('VTS, prior_K={}, {}'.format(this_K,mixture_expectation))

            # Execute each bandit
            with open(dir_string+'/bandits_labels.pickle', 'wb') as f:
                pickle.dump(bandits_labels, f)
            for (n_bandit,bandit) in enumerate(bandits):
                print('Executing {}'.format(bandits_labels[n_bandit]))
                # Determine context
                if context_type != 'none':
                    context=logged_context[:,logged_data_indexes]
                else:
                    if bandit.reward_function['type']=='bernoulli':
                        context=None
                    else:
                        context=np.ones((1,t_max))
                        
                # Per realization
                for r in np.arange(R):
                    print('r={}/{}'.format(r,R)) 
                    # Execute
                    bandit.execute(t_max, context)
                    # Collect performance metrics
                    actions[n_bandit,n_permutation,r,:,:]=bandit.actions
                    rewards[n_bandit,n_permutation,r,:]=bandit.rewards.sum(axis=0)        
                    # Save performance
                    with open(dir_string+'/bandit_performance.npz', 'wb') as f:
                        np.savez_compressed(f,actions=actions,rewards=rewards)

        '''
        pdb.set_trace()
        # Actually played
        (~np.isnan(rewards)).sum(axis=3)
        # Total rewards
        np.nansum(rewards,axis=3)
        # Averaged rewards
        np.nansum(rewards,axis=3)/(~np.isnan(rewards)).sum(axis=3)
        ''' 

# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    # Example no context:
    #   python3 -m pdb evaluate_loggedData_bandit.py -logged_data_file ../data/R6/bandits/20090506_bandit_8 -reward_model bernoulli,linearGaussian -context_type none -n_permutations 2 -R 2
    # Example user context:
    #   python3 -m pdb evaluate_loggedData_bandit.py -logged_data_file ../data/R6/bandits/20090506_bandit_8 -reward_model linearGaussian,logistic -context_type user -n_permutations 2 -R 2
    # Example user and article context:
    #   python3 -m pdb evaluate_loggedData_bandit.py -logged_data_file ../data/R6/bandits/20090506_bandit_8 -reward_model linearGaussian,logistic -context_type user_articles -n_permutations 2 -R 2
    parser = argparse.ArgumentParser(description='Evaluate bandits based on logged data.')
    parser.add_argument('-logged_data_file', type=str, help='File with mushroom data to use')
    parser.add_argument('-reward_model', type=str, default='linearGaussian', help='String describing bandit model to assume')
    parser.add_argument('-context_type', type=str, default='user', help='String describing what context type to use')
    parser.add_argument('-policy', type=str, default='TS', help='String describing bandit policy to use')
    parser.add_argument('-n_permutations', type=int, default=2, help='Number of data permutations to try')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    
    # Get arguments
    args = parser.parse_args()
    
    if os.path.exists('{}.pickle'.format(args.logged_data_file)):
        # Call main function
        main(args.logged_data_file, args.reward_model, args.context_type, args.policy, args.n_permutations, args.R)
    else:
        raise ValueError('Could not find logged_data_file={}'.format(args.logged_data_file))
