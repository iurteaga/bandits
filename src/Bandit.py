#!/usr/bin/python

# Imports: python modules
import abc
import numpy as np
import scipy.stats as stats

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
        cumregrets: cumulative regret of the bandit (per realization) as t_max array
        rewards_expected: the expected rewards computed for each arm of the bandit (per realization) as A by t_max array
        actions_R: dictionary with the actions that the bandit takes (for R realizations)
        rewards_R: dictionary with the rewards obtained by the bandit (for R realizations)
        regrets_R: dictioinary with the regret of the bandit (for R realizations)
        cumregrets_R: dictioinary with the cumulative regret of the bandit (for R realizations)
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
        self.cumregrets=None
        self.rewards_expected=None
        
        # For all realizations
        self.true_expected_rewards=None
        self.actions_R=None
        self.rewards_R=None
        self.regrets_R=None
        self.cumregrets_R=None
        self.rewards_expected_R=None

    def play_arm(self, a, t):
        """ Play bandit's arm a with true reward function
        
        Args:
            a: arm to play
            t: time index (or set of indexes)
        """
        #### REAL DATA SETS
        # For logged datasets
        if 'logged_data' in self.reward_function:
            # For logged data
            if self.reward_function['logged_arms'][t]==a:
                # Logged data matches selected arm, return reward
                self.rewards[a,t]=self.reward_function['logged_rewards'][t]
            else:
                # Logged data DOES NOT match selected arm, return NaN
                self.rewards[a,t]=np.nan
                
        # Mushroom dataset based rewards
        elif 'mushroom' in self.reward_function:
            # For the mushroom dataset
            # If not eating (arm 0), reward 0
            # If eating mushroom (arm1)
            if a==1:
                # Default reward
                self.rewards[a,t]=5+np.random.rand()
                # Unless it's poisounous and with probability 0.5
                #if self.reward_function['mushroom'][t]==1 and np.random.rand()<=0.5:
                if self.reward_function['mushroom'][t]==1:
                    self.rewards[a,t]=-15+np.random.rand()

        #### SIMULATED DATA SETS
        elif self.reward_function['type'] == 'bernoulli':
            # For Bernoulli distribution: expected value is \theta
            self.rewards[a,t]=self.reward_function['dist'].rvs(self.reward_function['theta'][a])
        
        elif self.reward_function['type'] == 'linear_gaussian':
            if 'dynamics' in self.reward_function:
                # For Linear Gaussian contextual bandit, expected value is dot product of context and parameters \theta
                self.rewards[a,t]=self.reward_function['dist'].rvs(loc=np.einsum('dt,td->t', np.reshape(self.context[:,t], (self.d_context, t.size)), np.reshape(self.reward_function['theta'][a,:,t], (t.size, self.d_context))), scale=self.reward_function['sigma'][a])
            else:
                # For Linear Gaussian contextual bandit, expected value is dot product of context and parameters \theta
                self.rewards[a,t]=self.reward_function['dist'].rvs(loc=np.einsum('dt,td->t', np.reshape(self.context[:,t], (self.d_context, t.size)), np.reshape(self.reward_function['theta'][a], (t.size, self.d_context))), scale=self.reward_function['sigma'][a])
        
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
            self.rewards[a,t]=self.reward_function['dist'].rvs(loc=np.einsum('dt,td->t', np.reshape(self.context[:,t], (self.d_context, t.size)), np.reshape(self.reward_function['theta'][a,mixture], (t.size, self.d_context))), scale=self.reward_function['sigma'][a,mixture])
        
        elif self.reward_function['type'] == 'logistic':
            if 'dynamics' in self.reward_function:
                # For logistic function, we have Bernoulli distribution with expected value x^\top\theta
                xTheta=np.einsum('dt,td->t', np.reshape(self.context[:,t], (self.d_context, t.size)), np.reshape(self.reward_function['theta'][a,:,t], (t.size, self.d_context)))
            else:
                # For logistic function, we have Bernoulli distribution with expected value x^\top\theta
                xTheta=np.einsum('dt,td->t', np.reshape(self.context[:,t], (self.d_context, t.size)), np.reshape(self.reward_function['theta'][a], (t.size, self.d_context)))
            self.rewards[a,t]=stats.bernoulli.rvs(np.exp(xTheta)/(1+np.exp(xTheta)))
        # TODO: Add other reward functions
        else:
            raise ValueError('Reward function={} not implemented yet'.format(self.reward_function))
            
    def compute_true_expected_rewards(self):
        """ Compute the expected rewards of the bandit for the true reward function
        
        Args:
            None
        """
        #### REAL DATA SETS
        # For logged datasets
        if 'logged_data' in self.reward_function:
            # Allocate
            self.true_expected_rewards=np.zeros((self.A,self.reward_function['logged_rewards'].size))
            # TODO: does empirical mean make sense as true expected reward?
            for a in np.arange(self.A):
                this_a=(self.reward_function['logged_arms']==a)
                self.true_expected_rewards[a,:]=(self.reward_function['logged_rewards'][this_a].sum())/(this_a.sum())

        # Mushroom dataset based rewards
        elif 'mushroom' in self.reward_function:
            # For the mushroom dataset
            # Zero reward if not eating (arm 0)
            self.true_expected_rewards=np.zeros((2,self.reward_function['mushroom'].size))
            # If eating mushroom (arm 0): zero reward for not eating poisounous (1), reward of 5 if eating edible (0)
            self.true_expected_rewards[1,self.reward_function['mushroom']==0]=5

        #### SIMULATED DATA SETS
        elif self.reward_function['type'] == 'bernoulli':
            # For Bernoulli distribution: expected value is \theta
            self.true_expected_rewards=self.reward_function['theta'][:,None]*np.ones(self.rewards.shape[1])
        elif self.reward_function['type'] == 'linear_gaussian':
            if 'dynamics' in self.reward_function:
                # For contextual linear Gaussian bandit, expected value is dot product of context and parameters \theta
                self.true_expected_rewards=np.einsum('dt,adt->at', self.context, self.reward_function['theta'])
            else:
                # For contextual linear Gaussian bandit, expected value is dot product of context and parameters \theta
                self.true_expected_rewards=np.einsum('dt,ad->at', self.context, self.reward_function['theta'])
        elif self.reward_function['type'] == 'linear_gaussian_mixture':
            # For contextual linear Gaussian mixture model bandit, weighted average of each mixture's expected value
            self.true_expected_rewards=np.einsum('ak,akd,dt->at', self.reward_function['pi'], self.reward_function['theta'], self.context)
        elif self.reward_function['type'] == 'logistic':
            if 'dynamics' in self.reward_function:
                # For the logistic function with 0/1 rewards the expected value
                xTheta=np.einsum('dt,adt->at', self.context, self.reward_function['theta'])
            else:
                # For the logistic function with 0/1 rewards the expected value
                xTheta=np.einsum('dt,ad->at', self.context, self.reward_function['theta'])
            self.true_expected_rewards=np.exp(xTheta)/(1+np.exp(xTheta))
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
