#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
from collections import defaultdict

from BayesianBandits import * 

#TODO: consider arms with different reward functions

class BayesianContextualBandit(BayesianBandit):
    """Class (Abstract) for Bayesian Contextual Bandits
    
    These type of bandits pick an action based on the probability of the action's context having the highest expected return
        They update the predictive density of each action's context based on previous rewards, actions and context
        They draw the next action from the latest predictive density

    Attributes (besides inherited):
        context_per_action: the context for each action (\ie vector of features per action)
        actions_context: the sequence of actions' context
        reward_prior: the assumed prior of the multi-armed bandit's reward function (dictionary)
        reward_posterior: the posterior of the multi-armed bandit's reward function (dictionary)
        actions_predictive_density: predictive density of each action
    """
    
    def __init__(self, K, context_per_action, reward_function, reward_prior):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit 
            context_per_action: the context for each action (\ie vector of features per action)
            reward_function: the reward function of the multi-armed bandit
            reward_prior: the assumed prior for the reward function of the multi-armed bandit (dictionary)
        """
        
        # Initialize bandit (without parameters)
        super().__init__(K, reward_function, reward_prior)
        
        # Context per action
        assert context_per_action.shape[0]==K, 'context_per_action must be K={} by d_context, not {}'.format(K, context_per_action.shape)
        self.d_context=context_per_action.shape[1]
        self.context_per_action = context_per_action
    
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
        
        # Linear Gaussian reward with Normal Inverse Gamma conjugate prior
        if self.reward_function['type'] == 'linear_gaussian' and self.reward_prior['dist'] == 'NIG':
            # Update and append Sigma
            self.reward_posterior['Sigma'].append(
            np.linalg.inv(np.linalg.inv(self.reward_prior['Sigma'])+self.actions_context[:,:t+1].dot(self.actions_context[:,:t+1].T))
            )
            # Update and append Theta
            self.reward_posterior['theta'].append(
            self.reward_posterior['Sigma'][-1].dot(np.linalg.inv(self.reward_prior['Sigma']).dot(self.reward_prior['theta'])+self.actions_context[:,:t+1].dot(self.returns[:,:t+1].sum(axis=0, keepdims=True).T))
            )
            # Update and append alpha
            self.reward_posterior['alpha'].append(
            self.reward_prior['alpha']+t/2
            )
            # Update and append beta
            self.reward_posterior['beta'].append(
            self.reward_prior['beta']+1/2*(
            self.reward_prior['theta'].T.dot(np.linalg.inv(self.reward_prior['Sigma']).dot(self.reward_prior['theta'])) +
            self.returns[:,:t+1].sum(axis=0, keepdims=True).dot(self.returns[:,:t+1].sum(axis=0, keepdims=True).T) -
            self.reward_posterior['theta'][-1].T.dot(np.linalg.inv(self.reward_posterior['Sigma'][-1]).dot(self.reward_posterior['theta'][-1]))
            )
            )
            
        # TODO: Add other reward/prior combinations
        else:
            raise ValueError('Invalid reward_function={} with reward_prior={} combination'.format(self.reward_function['type'], self.reward_prior['dist']))
        
    def execute(self, t_max):
        """ Execute the Bayesian bandit
            Overriden to keep track of context
        """
            
        
        # Initialize
        self.actions_predictive_density=np.zeros((self.K,t_max))
        self.actions=np.zeros((self.K,t_max))
        self.actions_context=np.zeros((self.d_context,t_max))
        self.returns=np.zeros((self.K,t_max))
        
        # Execute the bandit for each time instant
        for t in np.arange(0,t_max):            
            # Compute predictive density for expected returns at this time instant
            self.compute_action_predictive_density(t)

            # Draw action from predictive density at this time instant
            self.actions[:,t]=np.random.multinomial(1,self.actions_predictive_density[:,t],1)
            action = np.where(self.actions[:,t]==1)[0][0]
            self.actions_context[:,t]=self.context_per_action[action,:]

            # Compute return for true reward function
            self.returns[action,t]=self.reward_function['dist'].rvs(*self.reward_function['args'], **self.reward_function['kwargs'])[action] 

            # Update parameter posteriors
            self.update_reward_posterior(t)

class BayesianContextualBanditMonteCarlo(BayesianContextualBandit):
    """Class for Bayesian Contextual Bandits that compute the actions predictive density via Monte Carlo sampling
    
    These type of bandits pick an action based on the probability of the action's context having the highest expected return
        They update the predictive density of each action using Monte Carlo sampling, based on previous rewards, actions and contexts
        They draw the next action from the latest predictive action density

    Attributes (besides inherited):
        M: number of samples to use in the Monte Carlo integration
    """
    
    def __init__(self, K, context_per_action, reward_function, reward_prior, M):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit
            context_per_action: the context for each action (\ie vector of features per action)
            reward_function: the reward function of the multi-armed bandit
            reward_prior: the assumed prior for the reward function of the multi-armed bandit (dictionary)
            M: number of samples to use in the Monte Carlo integration
        """
        
        # Initialize bandit (without parameters)
        super().__init__(K, context_per_action, reward_function, reward_prior)
    
        self.M=M
        
    def compute_action_predictive_density(self, t):
        """ Compute the predictive density of each action using Monte Carlo sampling and based on rewards and action's context up until time t
            Overrides abstract method
        
        Args:
            t: time of the execution of the bandit
        """
        
        # Use updated poster hyperparameters
        if self.reward_function['type'] == 'linear_gaussian' and self.reward_prior['dist'] == 'NIG':
            # Sample reward's parameters (theta), given updated hyperparameters
            # First sample variance from inverse gamma
            sigma_samples=stats.invgamma.rvs(self.reward_posterior['alpha'][-1], scale=self.reward_posterior['beta'][-1], size=(1,self.M))       
            reward_params_samples=self.reward_posterior['theta'][-1]+np.sqrt(sigma_samples)*(stats.multivariate_normal.rvs(cov=self.reward_posterior['Sigma'][-1], size=self.M).reshape(self.M,self.d_context).T)
        
            # Compute expected rewards, linearly combining context and sampled parameters
            # In general,
            #    returns_expected_samples=self.reward_function(self.context_per_action.dot(reward_params_samples)).mean()
            # For Gaussian, simply:
            returns_expected_samples=self.context_per_action.dot(reward_params_samples)
            
        # TODO: Add other reward/prior combinations
        else:
            raise ValueError('Invalid reward_function={} with reward_prior={} combination'.format(self.reward_function['type'], self.reward_prior['dist']))
        
        # Pure Monte Carlo integration: count number of times expected reward is maximum        
        self.actions_predictive_density[:,t]=(((returns_expected_samples.argmax(axis=0)[None,:]==np.arange(self.K)[:,None]).astype(int)).sum(axis=1))/self.M
        
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
