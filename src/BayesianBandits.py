#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
from collections import defaultdict
import abc
import sys

from Bandits import * 

#TODO: consider arms with different reward functions
#TODO: should we compute and plot expected returns for Bayesian bandits over time?


# Class definitions
class BayesianBandit(abc.ABC,Bandit):
    """Class (Abstract) for Bayesian Bandits
    
    These type of bandits pick an action based on the probability of the action having the highest expected return
        They update the predictive density of each action based on previous rewards and actions
        They draw the next action from the latest predictive action density

    Attributes (besides inherited):
        reward_prior: the assumed prior of the multi-armed bandit's reward function (dictionary)
        reward_posterior: the posterior of the multi-armed bandit's reward function (dictionary)
        actions_predictive_density: predictive density of each action (per realization)
        actions_predictive_density_R: predictive density of each action (for R realizations, 'mean' and 'var')
    """
    
    def __init__(self, K, reward_function, reward_prior):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
            reward_prior: the assumed prior for the reward function of the multi-armed bandit (dictionary)
        """
        
        # Initialize bandit (without parameters)
        super().__init__(K, reward_function)
        
        # Reward prior (dictionary)
        self.reward_prior=reward_prior
        
        # Initialize reward posterior (dictonary)
            # TODO: Rethink dictionary list structure
        self.reward_posterior=defaultdict(list)
        for key, val in self.reward_prior.items():
            if key == 'name':
                self.reward_posterior['dist']=reward_prior['dist']
            else:
                self.reward_posterior[key].append(val)
    
    @abc.abstractmethod
    def compute_action_predictive_density(self, t):
        """ Compute the predictive density of each action based on rewards and actions up until time t
            This is an abstract method, as different alternatives are considered on how to compute this density
        
        Args:
            t: time of the execution of the bandit
        """
           
    def update_reward_posterior(self, t):
        """ Update the posterior of the reward density, based on rewards and actions up until time t
            This function is fully dependent on the type of prior and reward function
            
        Args:
            t: time of the execution of the bandit
        """
        
        # Binomial/Bernoulli reward with beta conjugate prior
        if self.reward_function['dist'].name == 'bernoulli' and self.reward_prior['dist'].name == 'beta':
            # Number of successes up to t (included)        
            s_t=self.returns[:,:t+1].sum(axis=1, keepdims=True)
            # Number of trials up to t (included)
            n_t=self.actions[:,:t+1].sum(axis=1, keepdims=True)
            self.reward_posterior['alpha'][-1]=self.reward_prior['alpha']+s_t
            self.reward_posterior['beta'][-1]=self.reward_prior['beta']+(n_t-s_t)

        # TODO: Add other reward/prior combinations
        else:
            raise ValueError('Invalid reward_function={} with reward_prior={} combination'.format(self.reward_function['dist'].name, self.reward_prior['dist'].name))
        
    def execute(self, t_max):
        """ Execute the Bayesian bandit """
        
        # Initialize
        self.actions_predictive_density=np.zeros((self.K,t_max))
        self.actions=np.zeros((self.K,t_max))
        self.returns=np.zeros((self.K,t_max))
        self.returns_expected=np.zeros((self.K,t_max))
        
        # Execute the bandit for each time instant
        for t in np.arange(0,t_max):            
            # Compute predictive density for expected returns at this time instant
            self.compute_action_predictive_density(t)

            # Draw action from predictive density at this time instant
            self.actions[:,t]=np.random.multinomial(1,self.actions_predictive_density[:,t])
            action = np.where(self.actions[:,t]==1)[0][0]

            # Compute return for true reward function
            self.returns[action,t]=self.reward_function['dist'].rvs(*self.reward_function['args'], **self.reward_function['kwargs'])[action]

            # Update parameter posteriors
            self.update_reward_posterior(t)

    def execute_realizations(self, R, t_max):
        """ Execute the bandit for R realizations """

        # Allocate overall variables
        self.returns_R={'mean':np.zeros((1,t_max)), 'm2':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
        self.returns_expected_R={'mean':np.zeros((self.K,t_max)), 'm2':np.zeros((self.K,t_max)), 'var':np.zeros((self.K,t_max))}
        self.actions_R={'mean':np.zeros((self.K,t_max)), 'm2':np.zeros((self.K,t_max)), 'var':np.zeros((self.K,t_max))}
        self.actions_predictive_density_R={'mean':np.zeros((self.K,t_max)), 'm2':np.zeros((self.K,t_max)), 'var':np.zeros((self.K,t_max))}

        # Execute all
        for r in np.arange(1,R+1):
            # Run one realization
            self.execute(t_max)

            # Update overall mean and variance sequentially
            self.returns_R['mean'], self.returns_R['m2'], self.returns_R['var']=online_update_mean_var(r, self.returns.sum(axis=0), self.returns_R['mean'], self.returns_R['m2'])
            self.returns_expected_R['mean'], self.returns_expected_R['m2'], self.returns_expected_R['var']=online_update_mean_var(r, self.returns_expected, self.returns_expected_R['mean'], self.returns_expected_R['m2'])
            self.actions_R['mean'], self.actions_R['m2'], self.actions_R['var']=online_update_mean_var(r, self.actions, self.actions_R['mean'], self.actions_R['m2'])
            self.actions_predictive_density_R['mean'], self.actions_predictive_density_R['m2'], self.actions_predictive_density_R['var']=online_update_mean_var(r, self.actions_predictive_density, self.actions_predictive_density_R['mean'], self.actions_predictive_density_R['m2'])

class BayesianBanditMonteCarlo(BayesianBandit):
    """Class for Bayesian Bandits that compute the actions predictive density via Monte Carlo sampling
    
    These type of bandits pick an action based on the probability of the action having the highest expected return
        They update the predictive density of each action using Monte Carlo sampling and based on previous rewards and actions
        They draw the next action from the latest predictive action density

    Attributes (besides inherited):
        M: number of samples to use in the Monte Carlo integration
    """
    
    def __init__(self, K, reward_function, reward_prior, M):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
            reward_prior: the assumed prior for the reward function of the multi-armed bandit (dictionary)
            M: number of samples to use in the Monte Carlo integration
        """
        
        # Initialize bandit (without parameters)
        super().__init__(K, reward_function, reward_prior)
    
        self.M=M
        
    def compute_action_predictive_density(self, t):
        """ Compute the predictive density of each action using Monte Carlo sampling and based on rewards and actions up until time t
            Overrides abstract method
        
        Args:
            t: time of the execution of the bandit
        """
        
        # Get posterior hyperparameters
        if self.reward_prior['dist'].name == 'beta':
            posterior_hyperparams=(self.reward_posterior['alpha'][-1], self.reward_posterior['beta'][-1])
        else:
            raise ValueError('reward_prior={} not implemented yet'.format(self.reward_prior['dist'].name))
        
        # Sample reward's parameters, given updated hyperparameters
        reward_params_samples=self.reward_posterior['dist'][0].rvs(*posterior_hyperparams, size=(self.K,self.M))
           
        # Compute expected returns, using updated parameters
        if self.reward_function['dist'].name == 'bernoulli':                   
            # Compute expected rewards, from Bernoulli with sampled parameters
            returns_expected_samples=reward_params_samples
        else:
            # In general,
            returns_expected_samples=self.reward_function['dist'].mean(reward_params_samples)
            
        self.returns_expected[:,t]=returns_expected_samples.mean(axis=1)
        
        # Pure Monte Carlo integration: count number of times expected reward is maximum        
        self.actions_predictive_density[:,t]=(((returns_expected_samples.argmax(axis=0)[None,:]==np.arange(self.K)[:,None]).astype(int)).sum(axis=1))/self.M
        
class BayesianBanditNumerical(BayesianBandit):
    """Class for Bayesian Bandits that compute the actions predictive density via numerical integration
    
    These type of bandits pick an action based on the probability of the action having the highest expected return
        They update the predictive density of each action by numerical integration and based on previous rewards and actions
        They draw the next action from the latest predictive action density

    Attributes (besides inherited):
        M: number of gridpoints to use for the numerical integration
    """
    
    def __init__(self, K, reward_function, reward_prior, M):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
            reward_prior: the assumed prior for the reward function of the multi-armed bandit (dictionary)
            M: number of gridpoints to use for the numerical integration
        """
        
        # Initialize bandit (without parameters)
        super().__init__(K, reward_function, reward_prior)
    
        self.M=M
        
    def compute_action_predictive_density(self, t):
        """ Compute the predictive density of each action by numerical integration and based on rewards and actions up until time t
            Overrides abstract method
        
        Args:
            t: time of the execution of the bandit
        """
        
        # Compute grid of return parameters
        if self.reward_prior['dist'].name == 'beta':
            #(0,1) range: TODO, use loc and scale to make it generic
            a=0
            b=1
            delta=(b-a)/self.M
            reward_params_grid=np.linspace(0+sys.float_info.epsilon,1-+sys.float_info.epsilon,self.M)
            
            # And also update posterior hyperparameters
            posterior_hyperparams=(self.reward_posterior['alpha'][-1], self.reward_posterior['beta'][-1])
        else:
            raise ValueError('reward_prior={} not implemented yet'.format(self.reward_prior['dist'].name))
               
        # Use updated return parameters
        if self.reward_function['dist'].name == 'bernoulli':                   
            # Compute expected rewards, from Bernoulli with sampled parameters
            returns_expected_grid=reward_params_grid
        else:
            # In general,
            returns_expected_grid=self.reward_function['dist'].mean(reward_params_grid)
        
        # All CDFs, with updated posterior hyperparameters
        all_cdfs=self.reward_posterior['dist'][0].cdf(returns_expected_grid, *posterior_hyperparams)
        all_index=np.arange(0,self.K)
        
        # Product of K-1 CDFs in k
        cdf_products=np.zeros((self.K,self.M))
        for k in all_index:
            cdf_products[k,:]=all_cdfs[np.setdiff1d(all_index,k),:].prod(axis=0, keepdims=True)
        
        # Numerical integration
        predictive_density=((self.reward_posterior['dist'][0].pdf(reward_params_grid, *posterior_hyperparams)*cdf_products).sum(axis=1))/delta
        self.actions_predictive_density[:,t]=predictive_density/predictive_density.sum()

class BayesianBanditHybridMonteCarlo(BayesianBandit):
    """Class for Bayesian Bandits that compute the actions predictive density via a hybrid Monte Carlo sampling
    
    These type of bandits pick an action based on the probability of the action having the highest expected return
        They update the predictive density of each action using a hybrid Monte Carlo sampling and based on previous rewards and actions
        They draw the next action from the latest predictive action density

    Attributes (besides inherited):
        M: number of samples to use in the Monte Carlo integration
    """
    
    def __init__(self, K, reward_function, reward_prior, M):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
            reward_prior: the assumed prior for the reward function of the multi-armed bandit (dictionary)
            M: number of samples to use in the Monte Carlo integration
        """
        
        # Initialize bandit (without parameters)
        super().__init__(K, reward_function, reward_prior)
    
        self.M=M
        
    def compute_action_predictive_density(self, t):
        """ Compute the predictive density of each action using Monte Carlo sampling and based on rewards and actions up until time t
            Overrides abstract method
        
        Args:
            t: time of the execution of the bandit
        """
        
        # Get posterior hyperparameters
        if self.reward_prior['dist'].name == 'beta':
            posterior_hyperparams=(self.reward_posterior['alpha'][-1], self.reward_posterior['beta'][-1])
        else:
            raise ValueError('reward_prior={} not implemented yet'.format(self.reward_prior['dist'].name))
        
        # Sample reward's parameters, given updated hyperparameters
        reward_params_samples=self.reward_posterior['dist'][0].rvs(*posterior_hyperparams, size=(self.K,self.M))
        
        # Compute expected returns, using updated parameters
        if self.reward_function['dist'].name == 'bernoulli':                   
            # Compute expected rewards, from Bernoulli with sampled parameters
            returns_expected_samples=reward_params_samples
        else:
            # In general,
            returns_expected_samples=self.reward_function['dist'].mean(reward_params_samples)
        
        # Product of K-1 CDFs in k
        all_index=np.arange(0,self.K)
        cdf_products=np.zeros((self.K,self.M))
        for k in all_index:
            not_k=np.setdiff1d(all_index,k)
            cdf_products[k,:]=self.reward_posterior['dist'][0].cdf(returns_expected_samples[k,:], *posterior_hyperparams)[not_k].prod(axis=0, keepdims=True)
            
        # Hybrid Monte-Carlo integration: Weighted CDF products at sampled points
        predictive_density=(cdf_products.sum(axis=1))/self.M
        self.actions_predictive_density[:,t]=predictive_density/predictive_density.sum()

	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
