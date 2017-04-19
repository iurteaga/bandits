#!/usr/bin/python

# Imports: python modules
import abc
import sys
import copy
import pickle
import pdb
import numpy as np
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt
from collections import defaultdict

######## Helper functions ########
def online_update_mean_var(r, new_instance, this_mean, this_m2):
    this_delta=new_instance - this_mean
    new_mean=this_mean+this_delta/r
    new_m2=this_m2+this_delta*(new_instance-new_mean)

    if r < 2:
        new_var=np.nan
    else:
        new_var=new_m2/(r-1)

    return (new_mean, new_m2, new_var)

######## Class definition ########
class Bandit(abc.ABC,object):
    """Abstract Class for Bandits

    Attributes:
        A: size of the multi-armed bandit 
        reward_function: dictionary with information about the reward distribution and its parameters is provided
        context: d_context by (at_least) t_max array with context for every time instant (None if does not apply)
        actions: the actions that the bandit takes (per realization) as A by t_max array
        rewards: rewards obtained by each arm of the bandit (per realization) as A by t_max array
        regrets: regret of the bandit (per realization) as t_max array
        rewards_expected: the expected rewards computed for each arm of the bandit (per realization) as A by t_max array
        actions_R: dictionary with the actions that the bandit takes (for R realizations)
        rewards_R: dictionary with the rewards obtained by the bandit (for R realizations)
        regrets_R: dictioinary with the regret of the bandit (for R realizations)
        rewards_expected_R: the expected rewards of the bandit (for R realizations)
    """
    
    def __init__(self, A, reward_function):
        """ Initialize the Bandit object and its attributes
        
        Args:
            A: the size of the bandit
            reward_function: the reward function of the bandit
        """
        
        # General Attributes
        self.A=A
        self.reward_function=reward_function
        self.context=None
                
        # Per realization
        self.actions=None
        self.rewards=None
        self.regrets=None
        self.rewards_expected=None
        
        # For all realizations
        self.true_expected_rewards=None
        self.actions_R=None
        self.rewards_R=None
        self.regrets_R=None
        self.rewards_expected_R=None

    def play_arm(self, a, t):
        """ Play bandit's arm a with true reward function
        
        Args:
            a: arm to play
            t: time index (or set of indexes)
        """

        if self.reward_function['dist'].name == 'bernoulli':
            # For Bernoulli distribution: expected value is \theta
            self.rewards[a,t]=self.reward_function['dist'].rvs(self.reward_function['theta'][a])
        elif self.reward_function['type'] == 'linear_gaussian':
            # For Linear Gaussian contextual bandit, expected value is dot product of context and parameters \theta
            self.rewards[a,t]=self.reward_function['dist'].rvs(loc=np.einsum('dt,dt->t', np.reshape(self.context[:,t], (self.d_context, t.size)), np.reshape(self.reward_function['theta'][a].T,(self.d_context, t.size))), scale=self.reward_function['sigma'][a])
        elif self.reward_function['type'] == 'linear_gaussian_mixture':
            # First, pick mixture
                # TODO: becaue np.random.multinomial does not implement broadcasting
            if t.size==1:
                mixture=np.where(np.random.multinomial(1,self.reward_function['pi'][a]))[0][0]
            else:
                mixture=np.zeros(t.size, dtype='int')
                for t_idx in np.arange(t.size):
                    mixture[t_idx]=np.where(np.random.multinomial(1,self.reward_function['pi'][a][t_idx]))[0][0]
            # Then draw
            self.rewards[a,t]=self.reward_function['dist'].rvs(loc=np.einsum('dt,dt->t', np.reshape(self.context[:,t], (self.d_context, t.size)), np.reshape(self.reward_function['theta'][a,mixture].T,(self.d_context, t.size))), scale=self.reward_function['sigma'][a,mixture])
        # TODO: Add other reward functions
        else:
            raise ValueError('Reward function={} not implemented yet'.format(self.reward_function))
            
    def compute_true_expected_rewards(self):
        """ Compute the expected rewards of the bandit for the true reward function
        
        Args:
            None
        """
                
        if self.reward_function['dist'].name == 'bernoulli':
            # For Bernoulli distribution: expected value is \theta
            self.true_expected_rewards=self.reward_function['theta'][:,None]*np.ones(self.rewards.shape[1])
        elif self.reward_function['type'] == 'linear_gaussian':
            # For contextual linear Gaussian bandit, expected value is dot product of context and parameters \theta
            self.true_expected_rewards=np.einsum('dt,ad->at', self.context, self.reward_function['theta'])
        elif self.reward_function['type'] == 'linear_gaussian_mixture':
            # For contextual linear Gaussian mixture model bandit, weighted average of each mixture's expected value
            self.true_expected_rewards=np.einsum('ak,akd,dt->at', self.reward_function['pi'], self.reward_function['theta'], self.context)
        # TODO: Add other reward functions
        else:
            raise ValueError('Reward function={} not implemented yet'.format(self.reward_function))

    @abc.abstractmethod
    def execute_realizations(self, R, t_max, context=None, exec_type='sequential'):
        """ Execute R realizations of the bandit
        Args:
            R: number of realizations to run
            t_max: maximum execution time for the bandit
            context: d_context by (at_least) t_max array with context for every time instant (None if does not apply)
            exec_type: batch (keep data from all realizations) or sequential (update mean and variance of realizations data)
        """
        
    @abc.abstractmethod
    def execute(self, t_max, context=None):
        """ Execute the Bayesian bandit
        Args:
            t_max: maximum execution time for the bandit
            context: d_context by (at_least) t_max array with context for every time instant (None if does not apply)
        """
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
