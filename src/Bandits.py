#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
from collections import defaultdict

def online_update_mean_var(r, new_instance, this_mean, this_m2):
    this_delta=new_instance - this_mean
    new_mean=this_mean+this_delta/r
    new_m2=this_m2+this_delta*(new_instance-new_mean)

    if r < 2:
        new_var=np.nan
    else:
        new_var=new_m2/(r-1)

    return (new_mean, new_m2, new_var)


#TODO: consider arms with different reward functions
# Class definitions
class Bandit(object):
    """General Class for Bandits

    Attributes:
        K: size of the multi-armed bandit 
        reward_function: the reward function of the multi-armed bandit: dictionary, where distribution and parameters provided
        actions: the actions that the bandit takes (per realization)
        returns: the returns obtained by the bandit (per realization)
        returns_expected: the expected returns of the bandit (per realization)
        actions_R: the actions that the bandit takes (for R realizations, 'mean' and 'var')
        returns_R: the returns obtained by the bandit (for R realizations, 'mean' and 'var')
        returns_expected_R: the expected returns of the bandit (for R realizations, 'mean' and 'var')
    """
    
    def __init__(self, K, reward_function):
        """ Initialize the Bandit object and its attributes
        
        Args:
            K: the size of the bandit
            reward_function: the reward function of the bandit
        """
        self.K=K
        self.reward_function=reward_function
        # Per realization
        self.actions=None
        self.returns=None
        self.returns_expected=None
        
    def compute_expected_returns(self, comp_type='empirical'):
        """ Compute the expected returns of the bandit for the corresponding reward function (frozen)
        
        Args:
            comp_type: how to compute the expectation: 'empirical' (default) or 'analytical'
        """
                
        if comp_type == 'empirical':
            self.returns_expected=self.reward_function.mean()
        elif comp_type == 'analytical':
            raise ValueError('comp_type="analytical" not implemented yet')
        else:
            raise ValueError('Invalid comp_type={}: use "empirical" (default) or "analytical"'.format(comp_type=repr(comp_type)))

      
class OptimalBandit(Bandit):
    """Class for Optimal Bandits
    
    This Bandit always picks the action with the highest expected return

    Attributes (besides inherited):
    """
    
    def __init__(self, K, reward_function):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
        """

        # Initialize reward function with provided parameters (frozen distribution)
        super().__init__(K, reward_function['dist'](*reward_function['args'], **reward_function['kwargs']))
        
        # Compute expected returns
        self.compute_expected_returns()
        
        if self.actions is None:
            # Decide optimal action, maximizing expected return
            self.actions=self.returns_expected.argmax()
       
    def execute(self, t_max):
        """ Execute the optimal bandit """
        
        # Simply draw from optimal action as many times as indicated
        self.returns=self.reward_function.rvs(size=(self.K,t_max))[self.actions,:]
        
    def execute_realizations(self, R, t_max):
        """ Execute the optimal bandit for R realizations """

        # Allocate overall variables
        self.returns_R={'mean':np.zeros((1, t_max)), 'm2':np.zeros((1, t_max)), 'var':np.zeros((1, t_max))}

        # Execute all
        for r in np.arange(1,R+1):
            # Run one realization
            self.execute(t_max)

            # Update overall mean and variance sequentially
            self.returns_R['mean'], self.returns_R['m2'], self.returns_R['var']=online_update_mean_var(r, self.returns.sum(axis=0), self.returns_R['mean'], self.returns_R['m2'])

        # Actions are the same all the time
        self.actions_R={'mean':self.actions, 'var':np.zeros((self.K,t_max))}

        # Expected returns are the same all the time
        self.returns_expected_R={'mean':self.returns_expected*np.ones((self.K,t_max)), 'var':np.zeros((self.K,t_max))}
        
        
class ProbabilisticBandit(Bandit):
    """Class for Probabilistic Bandits
    
    This Bandit randomly picks an action, given the expected return's probability

    Attributes (besides inherited):
    """
    
    def __init__(self, K, reward_function):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
        """

        # Initialize reward function with provided parameters (frozen distribution)
        super().__init__(K, reward_function['dist'](*reward_function['args'], **reward_function['kwargs']))
        
        # Compute expected returns
        self.compute_expected_returns()
       
    def execute(self, t_max):
        """ Execute the probabilistic bandit """
        
        # Initialize
        self.actions=np.zeros((self.K,t_max))
        self.returns=np.zeros((self.K,t_max))
        
        # Execute the bandit for each time instant
        for t in np.arange(0,t_max):
            # Draw action from (normalized) expected returns at this time instant
            self.actions[:,t]=np.random.multinomial(1,self.returns_expected[:,0]/np.sum(self.returns_expected[:,0]),1)
            action = np.where(self.actions[:,t]==1)[0][0]

            # Compute return for true reward function
            self.returns[action,t]=self.reward_function.rvs()[action]
	
    def execute_realizations(self, R, t_max):
        """ Execute the probabilistic bandit for R realizations """

        # Allocate overall variables
        self.returns_R={'mean':np.zeros((1,t_max)), 'm2':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
        self.actions_R={'mean':np.zeros((self.K,t_max)), 'm2':np.zeros((self.K,t_max)), 'var':np.zeros((self.K,t_max))}        

        # Execute all
        for r in np.arange(1,R+1):
            # Run one realization
            self.execute(t_max)

            # Update overall mean and variance sequentially
            self.returns_R['mean'], self.returns_R['m2'], self.returns_R['var']=online_update_mean_var(r, self.returns.sum(axis=0), self.returns_R['mean'], self.returns_R['m2'])
            self.actions_R['mean'], self.actions_R['m2'], self.actions_R['var']=online_update_mean_var(r, self.actions, self.actions_R['mean'], self.actions_R['m2'])

        # Expected returns are the same all the time
        self.returns_expected_R={'mean':self.returns_expected*np.ones((self.K,t_max)), 'var':np.zeros((self.K,t_max))}
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
