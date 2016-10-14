#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
from collections import defaultdict
import abc
import sys
import matplotlib.pyplot as plt

#TODO: consider arms with different reward functions

# Class definitions
class Bandit(object):
    """General Class for Bandits

    Attributes:
        K: size of the multi-armed bandit 
        reward_function: the reward function of the multi-armed bandit
        reward_params_true: true parameters of the multi-armed bandit's reward function
        actions: the actions that the bandit takes
        returns: the returns obtained by the bandit
        returns_expected: the expected returns of the bandit, for the provided reward function
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

      
class OptimalBandit(Bandit):
    """Class for Optimal Bandits
    
    This Bandit always picks the action with the highest expected return

    Attributes (besides inherited):
    """
    
    def __init__(self, K, reward_function, reward_params_true):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
            reward_params_true: the true parameters of the reward function of the multi-armed bandit
        """
        
        # Reward true parameters (as a tuple)
        self.reward_params_true=reward_params_true

        # Initialize with provided parameters
        super().__init__(K, reward_function(*self.reward_params_true))
        
        # Compute expected returns
        self.compute_expected_returns()
        
        if self.actions is None:
            # Decide optimal action, maximizing expected return
            self.actions=self.returns_expected.argmax()
            
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
       
    def execute(self, t_max):
        """ Execute the optimal bandit """
        
        # Simply draw from optimal action as many times as indicated
        self.returns=self.reward_function.rvs(size=(self.K,t_max))[self.actions,:]
  
        
class BayesianBandit(abc.ABC,Bandit):
    """Class (Abstract) for Bayesian Bandits
    
    These type of bandits pick an action based on the probability of the action having the highest expected return
        They update the predictive density of each action based on previous rewards and actions
        They draw the next action from the latest predictive action density

    Attributes (besides inherited):
        reward_prior: the assumed prior of the multi-armed bandit's reward function (dictionary)
        reward_posterior: the posterior of the multi-armed bandit's reward function (dictionary)
        returns_expected: expected returns of the multi-armed bandit, based on posterior reward function'sparameters
        actions_predictive_density: predictive density of each action
    """
    
    def __init__(self, K, reward_function, reward_params_true, reward_prior):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
            reward_params_true: the true parameters of the reward function of the multi-armed bandit
            reward_prior: the assumed prior for the reward function of the multi-armed bandit (dictionary)
        """
        
        # Initialize bandit (without parameters)
        super().__init__(K, reward_function)
        
        # Reward true parameters (as a tuple)
        self.reward_params_true=reward_params_true
        
        # Reward prior (dictionary)
        self.reward_prior=reward_prior
        
        # Initialize reward posterior (dictonary)
            # TODO: Rethink dictionary list structure
        self.reward_posterior=defaultdict(list)
        for key, val in self.reward_prior.items():
            if key == 'name':
                self.reward_posterior['function']=reward_prior['function']
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
        if self.reward_function.name == 'bernoulli' and self.reward_prior['function'].name == 'beta':
            # Number of successes up to t (included)        
            s_t=self.returns[:,:t+1].sum(axis=1, keepdims=True)
            # Number of trials up to t (included)
            n_t=self.actions[:,:t+1].sum(axis=1, keepdims=True)
            self.reward_posterior['alpha'].append(self.reward_prior['alpha']+s_t)
            self.reward_posterior['beta'].append(self.reward_prior['beta']+(n_t-s_t))

        # TODO: Add other reward/prior combinations
        else:
            raise ValueError('Invalid reward_function={} with reward_prior={} combination'.format(self.reward_function.name, self.reward_prior.name))
        
         
    def compute_expected_returns(self, t, reward_params, comp_type='empirical'):
        """ Compute the expected returns of the bandit for the corresponding reward function at time t
        
        Args:
            t: time of the execution of the bandit
            reward_params: posterior parameters of the reward function at time t (tuple)
            comp_type: how to compute the expectation: 'empirical' (default) or 'analytical'
        """
                
        if comp_type == 'empirical':
            self.returns_expected[:,t]=self.reward_function(reward_params).mean()
        elif comp_type == 'analytical':
            raise ValueError('comp_type="analytical" not implemented yet')
        else:
            raise ValueError('Invalid comp_type={}: use "empirical" (default) or "analytical"'.format(comp_type=repr(comp_type)))
        
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
            self.actions[:,t]=np.random.multinomial(1,self.actions_predictive_density[:,t],1)
            action = np.where(self.actions[:,t]==1)[0][0]

            # Compute return for true reward function
            self.returns[action,t]=self.reward_function.rvs(*self.reward_params_true)[action]     

            # Update parameter posteriors
            self.update_reward_posterior(t)

class BayesianBanditMonteCarlo(BayesianBandit):
    """Class for Bayesian Bandits that compute the actions predictive density via Monte Carlo sampling
    
    These type of bandits pick an action based on the probability of the action having the highest expected return
        They update the predictive density of each action using Monte Carlo sampling and based on previous rewards and actions
        They draw the next action from the latest predictive action density

    Attributes (besides inherited):
        M: number of samples to use in the Monte Carlo integration
    """
    
    def __init__(self, K, reward_function, reward_params_true, reward_prior, M):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
            reward_params_true: the true parameters of the reward function of the multi-armed bandit
            reward_prior: the assumed prior for the reward function of the multi-armed bandit (dictionary)
            M: number of samples to use in the Monte Carlo integration
        """
        
        # Initialize bandit (without parameters)
        super().__init__(K, reward_function, reward_params_true, reward_prior)
    
        self.M=M
        
    def compute_action_predictive_density(self, t):
        """ Compute the predictive density of each action using Monte Carlo sampling and based on rewards and actions up until time t
            Overrides abstract method
        
        Args:
            t: time of the execution of the bandit
        """
        
        # Use updated poster hyperparameters
        # TODO: can we do more efficiently?
        hyperparams=()
        for key in self.reward_posterior.keys():
            if key != 'function':
                hyperparams=(*hyperparams, self.reward_posterior[key][-1])
        # Sample reward's parameters, given such hyperparameters
            # TODO: Rethink dictionary list structure
        reward_params_samples=self.reward_posterior['function'][0].rvs(*hyperparams, size=(self.K,self.M))
        
        # Compute expected rewards
        returns_expected_samples=self.reward_function(reward_params_samples).mean()
        
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
    
    def __init__(self, K, reward_function, reward_params_true, reward_prior, M):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
            reward_params_true: the true parameters of the reward function of the multi-armed bandit
            reward_prior: the assumed prior for the reward function of the multi-armed bandit (dictionary)
            M: number of gridpoints to use for the numerical integration
        """
        
        # Initialize bandit (without parameters)
        super().__init__(K, reward_function, reward_params_true, reward_prior)
    
        self.M=M
        
    def compute_action_predictive_density(self, t):
        """ Compute the predictive density of each action by numerical integration and based on rewards and actions up until time t
            Overrides abstract method
        
        Args:
            t: time of the execution of the bandit
        """
        
        # Points to evaluate: grid points
        # Beta prior
        if self.reward_prior['function'].name == 'beta':
            a=0
            b=1
            delta=(b-a)/self.M
            reward_params_grid=np.linspace(0+sys.float_info.epsilon,1-+sys.float_info.epsilon,self.M)
        else:
            raise ValueError('Invalid reward_prior={}'.format(self.reward_prior.name))
        
        
        # Compute expected rewards
        returns_expected_grid=self.reward_function(reward_params_grid).mean()
        
        # All CDFs, with updated poster hyperparameters
        # TODO: can we do more efficiently?
        hyperparams=()
        for key in self.reward_posterior.keys():
            if key != 'function':
                hyperparams=(*hyperparams, self.reward_posterior[key][-1])

        all_cdfs=self.reward_posterior['function'][0].cdf(returns_expected_grid, *hyperparams)
        all_index=np.arange(0,self.K)
        
        # Product of K-1 CDFs in k
        cdf_products=np.zeros((self.K,self.M))
        for k in all_index:
            cdf_products[k,:]=all_cdfs[np.setdiff1d(all_index,k),:].prod(axis=0, keepdims=True)
        
        # Numerical integration
        predictive_density=((self.reward_posterior['function'][0].pdf(reward_params_grid, *hyperparams)*cdf_products).sum(axis=1))/delta
        self.actions_predictive_density[:,t]=predictive_density/predictive_density.sum()

class BayesianBanditHybridMonteCarlo(BayesianBandit):
    """Class for Bayesian Bandits that compute the actions predictive density via a hybrid Monte Carlo sampling
    
    These type of bandits pick an action based on the probability of the action having the highest expected return
        They update the predictive density of each action using a hybrid Monte Carlo sampling and based on previous rewards and actions
        They draw the next action from the latest predictive action density

    Attributes (besides inherited):
        M: number of samples to use in the Monte Carlo integration
    """
    
    def __init__(self, K, reward_function, reward_params_true, reward_prior, M):
        """ Initialize Optimal Bandits with public attributes 
        
        Args:
            K: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
            reward_params_true: the true parameters of the reward function of the multi-armed bandit
            reward_prior: the assumed prior for the reward function of the multi-armed bandit (dictionary)
            M: number of samples to use in the Monte Carlo integration
        """
        
        # Initialize bandit (without parameters)
        super().__init__(K, reward_function, reward_params_true, reward_prior)
    
        self.M=M
        
    def compute_action_predictive_density(self, t):
        """ Compute the predictive density of each action using Monte Carlo sampling and based on rewards and actions up until time t
            Overrides abstract method
        
        Args:
            t: time of the execution of the bandit
        """
        
        # Use updated poster hyperparameters
        # TODO: can we do more efficiently?
        hyperparams=()
        for key in self.reward_posterior.keys():
            if key != 'function':
                hyperparams=(*hyperparams, self.reward_posterior[key][-1])
        # Sample reward's parameters, given such hyperparameters
            # TODO: Rethink dictionary list structure
        reward_params_samples=self.reward_posterior['function'][0].rvs(*hyperparams, size=(self.K,self.M))
        
        # Compute expected rewards
        returns_expected_samples=self.reward_function(reward_params_samples).mean()
        
        # Product of K-1 CDFs in k
        all_index=np.arange(0,self.K)
        cdf_products=np.zeros((self.K,self.M))
        for k in all_index:
            not_k=np.setdiff1d(all_index,k)
            cdf_products[k,:]=self.reward_posterior['function'][0].cdf(returns_expected_samples[k,:], *hyperparams)[not_k].prod(axis=0, keepdims=True)
            
        # Hybrid Monte-Carlo integration: Weighted CDF products at sampled points
        predictive_density=(cdf_products.sum(axis=1))/self.M
        self.actions_predictive_density[:,t]=predictive_density/predictive_density.sum()


def execute_bandits(K, bandits, R, t_max, plot_colors=False):
    """ Execute a set of bandits for t_max time instants over R realization 
        
        Args:
            bandits: list of bandits to execute
            R: number of realizations to run
            t_max: maximum time for the bandits' execution
            plot_colors: colors to use for plotting each realization (default is False, no plotting)
        Rets:
            returns: array with averaged returns (bandit idx by K by t_max)
            actions: array with averaged actions (bandit idx by K by t_max)
            predictive: array with averaged action predictive density (bandit idx by K by t_max)            
    """
    
    n_bandits=len(bandits)
    
    # Allocate space
    returns=np.zeros((n_bandits, R, bandits[0].K, t_max))
    actions=np.zeros((n_bandits, R, bandits[0].K, t_max))
    predictive=np.zeros((n_bandits, R, bandits[0].K, t_max))

    # Realizations
    for r in np.arange(R):
        
        # Execute each bandit
        for n in np.arange(n_bandits):
            bandits[n].execute(t_max)
            
            if isinstance(bandits[n], OptimalBandit):
                returns[n,r,bandits[n].actions,:]=bandits[n].returns
                actions[n,r,bandits[n].actions,:]=np.ones(t_max)
                predictive[n,r,bandits[n].actions,:]=np.ones(t_max)
            elif isinstance(bandits[n], BayesianBandit):
                returns[n,r,:,:]=bandits[n].returns
                actions[n,r,:,:]=bandits[n].actions
                predictive[n,r,:,:]=bandits[n].actions_predictive_density
            else:
                raise ValueError('Invalid bandit number {} class type={}'.format(n, bandits[n]))            
        
    # Return averages
    return (returns, actions, predictive)
    
def plot_bandits(returns_expected, bandit_returns, bandit_actions, bandit_predictive, colors, labels, t_plot=None, plot_std=True):
    """ Plot results for a set of bandits
        
        Args:
            R: number of realizations executed
            returns_expected: true expected returns
            bandit_returns: bandits' returns
            bandit_actions: bandits' actions
            bandit_predictive: bandits' action predictive density
        Rets:
    """
    # Dimensionalities
    n_bandits, R, K, t_max = bandit_returns.shape

    if t_plot is None:
        t_plot=t_max
        
    # Returns over time
    plt.figure()
    plt.plot(np.arange(t_plot), returns_expected.max()*np.ones(t_plot), 'k', label='Expected')
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_plot), bandit_returns[n,:,:,0:t_plot].sum(axis=1).mean(axis=0), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit_returns[n,:,:,0:t_plot].sum(axis=1).mean(axis=0)-bandit_returns[n,:,:,0:t_plot].sum(axis=1).std(axis=0), bandit_returns[n,:,:,0:t_plot].sum(axis=1).mean(axis=0)+bandit_returns[n,:,:,0:t_plot].sum(axis=1).std(axis=0),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$y_t$')
    plt.title('Returns over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(0,-0.25), ncol=n_bandits, loc='center', shadow=True)
    plt.show()

    # Cumulative returns over time
    plt.figure()
    plt.plot(np.arange(t_plot), returns_expected.max()*np.arange(t_plot), 'k', label='Expected')
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_plot), bandit_returns[n,:,:,0:t_plot].sum(axis=1).cumsum(axis=1).mean(axis=0), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit_returns[n,:,:,0:t_plot].sum(axis=1).cumsum(axis=1).mean(axis=0)-bandit_returns[n,:,:,0:t_plot].sum(axis=1).cumsum(axis=1).std(axis=0), bandit_returns[n,:,:,0:t_plot].sum(axis=1).cumsum(axis=1).mean(axis=0)+bandit_returns[n,:,:,0:t_plot].sum(axis=1).cumsum(axis=1).std(axis=0),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$\sum_{t=0}^Ty_t$')
    plt.title('Cumulative returns over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(0,-0.25), ncol=n_bandits, loc='center', shadow=True)
    plt.show()

    # Regret over time
    plt.figure()
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_plot), returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1).mean(axis=0), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), (returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1).mean(axis=0))-bandit_returns[n,:,:,0:t_plot].sum(axis=1).std(axis=0), (returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1).mean(axis=0))+bandit_returns[n,:,:,0:t_plot].sum(axis=1).std(axis=0),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$l_t=y_t^*-y_t$')
    plt.title('Regret over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(0,-0.25), ncol=n_bandits, loc='center', shadow=True)
    plt.show()

    # Cumulative regret over time
    plt.figure()
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_plot), (returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1)).cumsum(axis=1).mean(axis=0), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), (returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1)).cumsum(axis=1).mean(axis=0)-(returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1)).cumsum(axis=1).std(axis=0), (returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1)).cumsum(axis=1).mean(axis=0)+(returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1)).cumsum(axis=1).std(axis=0),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$L_t=\sum_{t=0}^T y_t^*-y_t$')
    plt.title('Cumulative regret over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(0,-0.25), ncol=n_bandits, loc='center', shadow=True)
    plt.show()  

    # Action predictive density probabilities over time: separate plots
    for k in np.arange(0,K):
        plt.figure()
        for n in np.arange(n_bandits):
            plt.plot(np.arange(t_plot), bandit_predictive[n,:,k,0:t_plot].mean(axis=0), colors[n], label=labels[n]+' predictive density')
            if plot_std:
                plt.fill_between(np.arange(t_plot), bandit_predictive[n,:,k,0:t_plot].mean(axis=0)-bandit_predictive[n,:,k,0:t_plot].std(axis=0), bandit_predictive[n,:,k,0:t_plot].mean(axis=0)+bandit_predictive[n,:,k,0:t_plot].std(axis=0),alpha=0.5, facecolor=colors[n])
        plt.ylabel(r'$f(a_{t+1}=a|a_{1:t}, y_{1:t})$')
        plt.xlabel('t')
        plt.title('Averaged Action Predictive density probabilities for arm {}'.format(k))
        plt.xlim([0, t_plot-1])
        legend = plt.legend(bbox_to_anchor=(0,-0.25), ncol=n_bandits, loc='center', shadow=True)
        plt.show()
    
    # Action probabilities over time: separate plots
    for k in np.arange(0,K):
        plt.figure()
        for n in np.arange(n_bandits):
            plt.plot(np.arange(t_plot), bandit_actions[n,:,k,0:t_plot].mean(axis=0), colors[n], label=labels[n]+' actions')
            if plot_std:
                plt.fill_between(np.arange(t_plot), bandit_actions[n,:,k,0:t_plot].mean(axis=0)-bandit_actions[n,:,k,0:t_plot].std(axis=0), bandit_actions[n,:,k,0:t_plot].mean(axis=0)+bandit_actions[n,:,k,0:t_plot].std(axis=0),alpha=0.5, facecolor=colors[n])
        plt.ylabel(r'$f(a_{t+1}=a|a_{1:t}, y_{1:t})$')
        plt.xlabel('t')
        plt.title('Averaged Action probabilities for arm {}'.format(k))
        plt.xlim([0, t_plot-1])
        legend = plt.legend(bbox_to_anchor=(0,-0.25), ncol=n_bandits, loc='center', shadow=True)
        plt.show()

    # Action probabilities (combined) over time: Subplots
    """
    f, *axes = plt.subplots(K, sharex=True, sharey=True)
    for k in np.arange(0,K):
        for n in np.arange(n_bandits):
            axes[0][k].plot(np.arange(t_plot), bandit_predictive[n,:,k,0:t_plot].mean(axis=0), colors[n], label=labels[n]+' predictive density')
            axes[0][k].plot(np.arange(t_plot), bandit_actions[n,:,k,0:t_plot].mean(axis=0), colors[n]+'-.', label=labels[n]+' actions')
            if plot_std:
                axes[0][k].fill_between(np.arange(t_plot), bandit_predictive[n,:,k,0:t_plot].mean(axis=0)-bandit_predictive[n,:,k,0:t_plot].std(axis=0), bandit_predictive[n,:,k,0:t_plot].mean(axis=0)+bandit_predictive[n,:,k,0:t_plot].std(axis=0),alpha=0.5, facecolor=colors[n])
                axes[0][k].fill_between(np.arange(t_plot), bandit_actions[n,:,k,0:t_plot].mean(axis=0)-bandit_actions[n,:,k,0:t_plot].std(axis=0), bandit_actions[n,:,k,0:t_plot].mean(axis=0)+bandit_actions[n,:,k,0:t_plot].std(axis=0),alpha=0.5, facecolor=colors[n])
    plt.ylabel(r'$f(a_{t+1}=a|a_{1:t}, y_{1:t})$')
    plt.xlabel('t')
    plt.title('Averaged Predictive probabilities')
    plt.axis([0, t_plot-1,0,1])
    legend = plt.legend(loc='upper right', shadow=True)
    plt.show()
    
    # Action probabilities (combined) over time: separate plots
    for k in np.arange(0,K):
        plt.figure()
        for n in np.arange(n_bandits):
            plt.plot(np.arange(t_plot), bandit_predictive[n,:,k,0:t_plot].mean(axis=0), colors[n], label=labels[n]+' predictive density')
            plt.plot(np.arange(t_plot), bandit_actions[n,:,k,0:t_plot].mean(axis=0), colors[n]+'-.', label=labels[n]+' actions')
            if plot_std:
                plt.fill_between(np.arange(t_plot), bandit_predictive[n,:,k,0:t_plot].mean(axis=0)-bandit_predictive[n,:,k,0:t_plot].std(axis=0), bandit_predictive[n,:,k,0:t_plot].mean(axis=0)+bandit_predictive[n,:,k,0:t_plot].std(axis=0),alpha=0.5, facecolor=colors[n])
                plt.fill_between(np.arange(t_plot), bandit_actions[n,:,k,0:t_plot].mean(axis=0)-bandit_actions[n,:,k,0:t_plot].std(axis=0), bandit_actions[n,:,k,0:t_plot].mean(axis=0)+bandit_actions[n,:,k,0:t_plot].std(axis=0),alpha=0.5, facecolor=colors[n])
        plt.ylabel(r'$f(a_{t+1}=a|a_{1:t}, y_{1:t})$')
        plt.xlabel('t')
        plt.title('Averaged Predictive probability for arm {}'.format(k))
        plt.axis([0, t_plot-1,0,1])
        legend = plt.legend(bbox_to_anchor=(0,-0.25), ncol=n_bandits, loc='center', shadow=True)
        plt.show()
    """
    	
	# Correct arm selection probability
    plt.figure()
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_plot), bandit_actions[n,:,returns_expected.argmax(),0:t_plot].mean(axis=0), colors[n], label=labels[n]+' actions')
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit_actions[n,:,returns_expected.argmax(),0:t_plot].mean(axis=0)-bandit_actions[n,:,returns_expected.argmax(),0:t_plot].std(axis=0), bandit_actions[n,:,returns_expected.argmax(),0:t_plot].mean(axis=0)+bandit_actions[n,:,returns_expected.argmax(),0:t_plot].std(axis=0),alpha=0.5, facecolor=colors[n])
    plt.ylabel(r'$f(a_{t+1}=a^*|a_{1:t}, y_{1:t})$')
    plt.xlabel('t')
    plt.title('Averaged Correct Action probabilities')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(0,-0.25), ncol=n_bandits, loc='center', shadow=True)
    plt.show()
	
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
