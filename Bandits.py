#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
from collections import defaultdict

#TODO: consider arms with different reward functions
# Class definitions
class Bandit(object):
    """General Class for Bandits

    Attributes:
        K: size of the multi-armed bandit 
        reward_function: the reward function of the multi-armed bandit: dictionary, where distribution and parameters provided
        actions: the actions that the bandit takes
        returns: the returns obtained by the bandit
        returns_expected: the expected returns of the bandit
    """
    
    def __init__(self, K, reward_function):
        """ Initialize the Bandit object and its attributes
        
        Args:
            K: the size of the bandit
            reward_function: the reward function of the bandit
        """
        self.K=K
        self.reward_function=reward_function
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
	
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
