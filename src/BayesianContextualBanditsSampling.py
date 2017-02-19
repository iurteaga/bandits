#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
from collections import defaultdict
import abc
import sys
from Bandits import * 

import pdb
# To control numpy errors
#np.seterr(all='raise')

# Class definitions
class BayesianContextualBanditSampling(abc.ABC,Bandit):
    """Class (Abstract) for Bayesian Contextual Bandits with action sampling
    
    These type of bandits pick, given a context, an action by sampling the next action based on the probability of the action having the highest expected return
        They update the predictive density of each action based on previous context, rewards and actions via Bayes update rule
        They draw the next action by (dynamically) SAMPLING from the latest predictive action density, then picking most likely option
            the number of samples balances exploration/exploitation
            different alternatives for the number of action samples to use are implemented

    Attributes (besides inherited):
        reward_prior: the assumed prior of the multi-armed bandit's reward function (dictionary)
        reward_posterior: the posterior of the multi-armed bandit's reward function (dictionary)
        actions_predictive_density: predictive density of each action (mean and var in dictionary)
        sampling: sampling strategy for deciding on action
        n_samples: number of samples for action decision at each time instant
    """
    
    def __init__(self, A, reward_function, reward_prior, sampling):
        """ Initialize Bayesian Bandits with public attributes
        
        Args:
            A: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
            reward_prior: the assumed prior for the reward function of the multi-armed bandit (dictionary)
            sampling: sampling strategy for deciding on action
        """
        
        # Initialize bandit (without parameters)
        super().__init__(A, reward_function)
        
        # Reward prior (dictionary)
        self.reward_prior=reward_prior
                
        # Initialize reward posterior (dictonary)
        self.reward_posterior=reward_prior
        '''
        # TODO: The following dictionary list structure would be useful for keeping track over time of the posterior (growing with append)
        self.reward_posterior=defaultdict(list)
        for key, val in self.reward_prior.items():
            if key == 'name':
                self.reward_posterior['dist']=reward_prior['dist']
            else:
                self.reward_posterior[key].append(val)
        '''
         
        # sampling strategy
        self.sampling=sampling
    
    @abc.abstractmethod
    def compute_action_predictive_density(self, t):
        """ Compute the predictive density of each action based on rewards and actions up until time t
            This is an abstract method, as different alternatives are considered on how to compute this density
        
        Args:
            t: time of the execution of the bandit
        """
           
    def update_reward_posterior(self, t):
        # TODO: The code works for batch and sequential updates (although not efficient for sequential)
        """ Update the posterior of the reward density, based on context, rewards and actions up until time t
            This function is fully dependent on the type of prior and reward function
        Args:
            t: time of the execution of the bandit
        """

        # Linear Gaussian reward with Normal Inverse Gamma conjugate prior
        if self.reward_function['type'] == 'linear_gaussian' and self.reward_prior['dist'] == 'NIG':
            for a in np.arange(self.A):
                this_a=self.actions[a,:]==1
                # Update and append Sigma
                self.reward_posterior['Sigma'][a,:,:]=np.linalg.inv(np.linalg.inv(self.reward_prior['Sigma'][a,:,:])+np.dot(self.context[:,this_a], self.context[:,this_a].T))
                # Update and append Theta
                self.reward_posterior['theta'][a,:]=np.dot(self.reward_posterior['Sigma'][a,:,:], np.dot(np.linalg.inv(self.reward_prior['Sigma'][a,:,:]), self.reward_prior['theta'][a,:])+np.dot(self.context[:,this_a], self.returns[a,this_a].T))
                # Update and append alpha
                self.reward_posterior['alpha'][a]=self.reward_prior['alpha'][a]+this_a.size/2
                # Update and append beta
                self.reward_posterior['beta'][a]=self.reward_prior['beta'][a]+1/2*(
                np.dot(self.returns[a,this_a], self.returns[a,this_a].T) +
                np.dot(self.reward_prior['theta'][a,:].T, np.dot(np.linalg.inv(self.reward_prior['Sigma'][a,:,:]), self.reward_prior['theta'][a,:])) -
                np.dot(self.reward_posterior['theta'][a,:].T, np.dot(np.linalg.inv(self.reward_posterior['Sigma'][a,:,:]), self.reward_posterior['theta'][a,:]))
                )

        # TODO: Add other reward/prior combinations
        else:
            raise ValueError('Invalid reward_function={} with reward_prior={} combination'.format(self.reward_function['dist'].name, self.reward_prior['dist'].name))
        
    def execute(self, context, t_max):
        """ Execute the Bayesian bandit
        Args:
            context: d_context by (at_least) t_max array with context at every time instant
            t_max: maximum time for execution of the bandit
        """
        
        # Context
        self.d_context=context.shape[0]
        assert context.shape[1]>=t_max, 'Not enough context provided: context.shape[1]={} while t_max={}'.format(context.shape[1],t_max)
        self.context=context
        
        # Initialize
        self.actions_predictive_density={'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
        self.actions=np.zeros((self.A,t_max))
        self.returns=np.zeros((self.A,t_max))
        self.returns_expected=np.zeros((self.A,t_max))
        self.n_samples=np.ones(t_max)
                
        # Execute the bandit for each time instant
        for t in np.arange(0,t_max):
            #print('Running time instant {}'.format(t))
            # Compute predictive density for expected returns at this time instant
            self.compute_action_predictive_density(t)

            # Draw action from predictive density at this time instant
            # Compute number of samples, based on sampling strategy
            if self.sampling['type'] == 'static':
                # Static number of samples
                self.n_samples[t]=self.sampling['n_samples']
            elif self.sampling['type'] == 'linear':
                # Linear number of samples: n_t=n * t +n_0 and enforce at least 1 sample
                self.n_samples[t]=np.maximum(self.sampling['n_0']+self.sampling['n']*t,1)
            elif self.sampling['type'] == 'logT':
                # Logarithmic number of samples: n_t=log(t) and enforce at least 1 sample
                self.n_samples[t]=np.maximum(np.log(t),1)
            elif self.sampling['type'] == 'sqrtT':
                # Square root number of samples: n_t=sqrt(t) and enforce at least 1 sample
                self.n_samples[t]=np.maximum(np.sqrt(t),1)
            elif self.sampling['type'] == 'invVar':
                # Inverse of uncertainty in picked action, and enforce at least 1 sample
                self.n_samples[t]=np.fmin(np.maximum(1/self.actions_predictive_density['var'][self.actions_predictive_density['mean'][:,t].argmax(),t],1), self.sampling['n_max'])
            elif self.sampling['type'] == 'invPnotOpt':
                # Inverse of probability of not optimal, with exploitation rate, and enforce at least 1 sample
                self.n_samples[t]=np.fmin(np.maximum(self.sampling['n_rate']/(1-self.actions_predictive_density['mean'][self.actions_predictive_density['mean'][:,t].argmax(),t]),1), self.sampling['n_max'])
            elif self.sampling['type'] == 'loginvPnotOpt':
                # Log of inverse of probability of not optimal, and enforce at least 1 sample
                self.n_samples[t]=np.fmin(np.maximum(np.log(1/(1-self.actions_predictive_density['mean'][self.actions_predictive_density['mean'][:,t].argmax(),t])),1), self.sampling['n_max'])
            elif self.sampling['type'] == 'log10invPnotOpt':
                # Log of inverse of probability of not optimal, and enforce at least 1 sample
                self.n_samples[t]=np.fmin(np.maximum(np.log10(1/(1-self.actions_predictive_density['mean'][self.actions_predictive_density['mean'][:,t].argmax(),t])),1), self.sampling['n_max'])
            elif self.sampling['type'] == 'invPnotOptandVar':
                # Inverse of probability of not optimal and uncertainty in picked action, with exploitation rate, and enforce at least 1 sample
                p_opt=self.actions_predictive_density['mean'][self.actions_predictive_density['mean'][:,t].argmax(),t]
                var_opt=self.actions_predictive_density['var'][self.actions_predictive_density['mean'][:,t].argmax(),t]
                self.n_samples[t]=np.fmin(np.maximum(self.sampling['n_rate']/((1-p_opt)*var_opt),1), self.sampling['n_max'])
            elif self.sampling['type'] == 'ratioPopPnotOpt':
                # Ratio of probability of optimal to not optimal arm, and enforce at least 1 sample
                p_opt=self.actions_predictive_density['mean'][self.actions_predictive_density['mean'][:,t].argmax(),t]
                self.n_samples[t]=np.fmin(np.maximum(p_opt/(1-p_opt),1), self.sampling['n_max'])
            elif self.sampling['type'] == 'invPFA_tGaussian':
                # Inverse of probability of other arms being optimal (prob false alarm), and enforce at least 1 sample
                a_opt=self.actions_predictive_density['mean'][:,t].argmax()
                p_opt=self.actions_predictive_density['mean'][a_opt,t]
                # Sufficient statistics for truncated Gaussian evaluation
                xi=(p_opt-self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t])/np.sqrt(self.actions_predictive_density['var'][np.arange(self.A)!=a_opt,t])
                alpha=(-self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t])/np.sqrt(self.actions_predictive_density['var'][np.arange(self.A)!=a_opt,t])
                beta=(1-self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t])/np.sqrt(self.actions_predictive_density['var'][np.arange(self.A)!=a_opt,t])
                # p_fa
                p_fa=(1-(stats.norm.cdf(xi)-stats.norm.cdf(alpha))/(stats.norm.cdf(beta)-stats.norm.cdf(alpha))).sum()/(self.A-1)
                #n_samples
                self.n_samples[t]=np.fmin(np.maximum(1/p_fa,1), self.sampling['n_max'])
            elif self.sampling['type'] == 'loginvPFA_tGaussian':
                # Log of inverse of probability of other arms being optimal (prob false alarm), and enforce at least 1 sample
                a_opt=self.actions_predictive_density['mean'][:,t].argmax()
                p_opt=self.actions_predictive_density['mean'][a_opt,t]
                # Sufficient statistics for truncated Gaussian evaluation
                xi=(p_opt-self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t])/np.sqrt(self.actions_predictive_density['var'][np.arange(self.A)!=a_opt,t])
                alpha=(-self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t])/np.sqrt(self.actions_predictive_density['var'][np.arange(self.A)!=a_opt,t])
                beta=(1-self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t])/np.sqrt(self.actions_predictive_density['var'][np.arange(self.A)!=a_opt,t])
                # p_fa
                p_fa=(1-(stats.norm.cdf(xi)-stats.norm.cdf(alpha))/(stats.norm.cdf(beta)-stats.norm.cdf(alpha))).sum()/(self.A-1)
                #n_samples
                self.n_samples[t]=np.fmin(np.maximum(np.log(1/p_fa),1), self.sampling['n_max'])
            elif self.sampling['type'] == 'log10invPFA_tGaussian':
                # Log of inverse of probability of other arms being optimal (prob false alarm), and enforce at least 1 sample
                a_opt=self.actions_predictive_density['mean'][:,t].argmax()
                p_opt=self.actions_predictive_density['mean'][a_opt,t]
                # Sufficient statistics for truncated Gaussian evaluation
                xi=(p_opt-self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t])/np.sqrt(self.actions_predictive_density['var'][np.arange(self.A)!=a_opt,t])
                alpha=(-self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t])/np.sqrt(self.actions_predictive_density['var'][np.arange(self.A)!=a_opt,t])
                beta=(1-self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t])/np.sqrt(self.actions_predictive_density['var'][np.arange(self.A)!=a_opt,t])
                # p_fa
                p_fa=(1-(stats.norm.cdf(xi)-stats.norm.cdf(alpha))/(stats.norm.cdf(beta)-stats.norm.cdf(alpha))).sum()/(self.A-1)
                #n_samples
                self.n_samples[t]=np.fmin(np.maximum(np.log10(1/p_fa),1), self.sampling['n_max'])
            elif self.sampling['type'] == 'invPFA_Markov':
                # Log of inverse of probability of other arms being optimal (prob false alarm), and enforce at least 1 sample
                a_opt=self.actions_predictive_density['mean'][:,t].argmax()
                mu_opt=self.actions_predictive_density['mean'][a_opt,t]
                # Markov's inequality
                p_fa=(self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t]/mu_opt).sum()/(self.A-1)
                #n_samples
                self.n_samples[t]=np.fmin(np.maximum(1/p_fa,1), self.sampling['n_max'])
            elif self.sampling['type'] == 'loginvPFA_Markov':
                # Log of inverse of probability of other arms being optimal (prob false alarm), and enforce at least 1 sample
                a_opt=self.actions_predictive_density['mean'][:,t].argmax()
                mu_opt=self.actions_predictive_density['mean'][a_opt,t]
                # Markov's inequality
                p_fa=(self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t]/mu_opt).sum()/(self.A-1)
                #n_samples
                self.n_samples[t]=np.fmin(np.maximum(np.log(1/p_fa),1), self.sampling['n_max'])
            elif self.sampling['type'] == 'log10invPFA_Markov':
                # Log of inverse of probability of other arms being optimal (prob false alarm), and enforce at least 1 sample
                a_opt=self.actions_predictive_density['mean'][:,t].argmax()
                mu_opt=self.actions_predictive_density['mean'][a_opt,t]
                # Markov's inequality
                p_fa=(self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t]/mu_opt).sum()/(self.A-1)
                #n_samples
                self.n_samples[t]=np.fmin(np.maximum(np.log10(1/p_fa),1), self.sampling['n_max'])
            elif self.sampling['type'] == 'invPFA_Chebyshev':
                # Log of inverse of probability of other arms being optimal (prob false alarm), and enforce at least 1 sample
                a_opt=self.actions_predictive_density['mean'][:,t].argmax()
                mu_opt=self.actions_predictive_density['mean'][a_opt,t]
                # Chebyshev's inequality
                delta=mu_opt - self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t]
                p_fa=(self.actions_predictive_density['var'][np.arange(self.A)!=a_opt,t]/np.power(delta,2)).sum()/(self.A-1)
                #n_samples
                self.n_samples[t]=np.fmin(np.maximum(1/p_fa,1), self.sampling['n_max'])
            elif self.sampling['type'] == 'loginvPFA_Chebyshev':
                # Log of inverse of probability of other arms being optimal (prob false alarm), and enforce at least 1 sample
                a_opt=self.actions_predictive_density['mean'][:,t].argmax()
                mu_opt=self.actions_predictive_density['mean'][a_opt,t]
                # Chebyshev's inequality
                delta=mu_opt - self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t]
                p_fa=(self.actions_predictive_density['var'][np.arange(self.A)!=a_opt,t]/np.power(delta,2)).sum()/(self.A-1)
                #n_samples
                self.n_samples[t]=np.fmin(np.maximum(np.log(1/p_fa),1), self.sampling['n_max'])
            elif self.sampling['type'] == 'log10invPFA_Chebyshev':
                # Log of inverse of probability of other arms being optimal (prob false alarm), and enforce at least 1 sample
                a_opt=self.actions_predictive_density['mean'][:,t].argmax()
                mu_opt=self.actions_predictive_density['mean'][a_opt,t]
                # Chebyshev's inequality
                delta=mu_opt - self.actions_predictive_density['mean'][np.arange(self.A)!=a_opt,t]
                p_fa=(self.actions_predictive_density['var'][np.arange(self.A)!=a_opt,t]/np.power(delta,2)).sum()/(self.A-1)
                #n_samples
                self.n_samples[t]=np.fmin(np.maximum(np.log10(1/p_fa),1), self.sampling['n_max'])
            elif self.sampling['type'] == 'argMax':
                # Infinite samples are equivalent to picking maximum
                self.n_samples[t]=np.inf
            else:
                raise ValueError('Invalid sampling type={}'.format(self.sampling['type']))
            
            # Pick next action
            if self.n_samples[t] == np.inf:
                # Pick maximum 
                action = self.actions_predictive_density[:,t].argmax()
                self.actions[action,t]=1.
            else:
                # SAMPLE n_samples and pick the most likely action                
                self.actions[np.random.multinomial(1,self.actions_predictive_density['mean'][:,t], size=int(self.n_samples[t])).sum(axis=0).argmax(),t]=1
                action = np.where(self.actions[:,t]==1)[0][0]

            # Compute return for true reward function
            self.returns[action,t]=self.reward_function['dist'].rvs(*self.reward_function['args'], **self.reward_function['kwargs'])[action]

            # Update parameter posteriors
            self.update_reward_posterior(t)
        
    def execute_realizations(self, R, context, t_max, exec_type='online'):
        """ Execute the bandit for R realizations
        Args:
            R: number of realizations to run
            context: d_context by (at_least) t_max array with context at every time instant
            t_max: maximum time for execution of the bandit
        """
        
        # Allocate overall variables
        if exec_type == 'online':
            self.returns_R={'mean':np.zeros((1,t_max)), 'm2':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.returns_expected_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.actions_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.actions_predictive_density_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.n_samples_R={'mean':np.zeros(t_max), 'm2':np.zeros(t_max), 'var':np.zeros(t_max)}
        elif exec_type =='general':
            self.returns_R={'all':np.zeros((R,1,t_max)), 'mean':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.returns_expected_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.actions_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.actions_predictive_density_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.actions_predictive_density_var_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.n_samples_R={'all':np.zeros((R,t_max)),'mean':np.zeros(t_max), 'var':np.zeros(t_max)}            
        else:
            raise ValueError('Execution type={} not implemented'.format(exec_type))
            
        # Execute all
        for r in np.arange(R):
            # Run one realization
            #print('Executing realization {}'.format(r))
            self.execute(context, t_max)

            if exec_type == 'online':
                # Update overall mean and variance sequentially
                self.returns_R['mean'], self.returns_R['m2'], self.returns_R['var']=online_update_mean_var(r+1, self.returns.sum(axis=0), self.returns_R['mean'], self.returns_R['m2'])
                self.returns_expected_R['mean'], self.returns_expected_R['m2'], self.returns_expected_R['var']=online_update_mean_var(r+1, self.returns_expected, self.returns_expected_R['mean'], self.returns_expected_R['m2'])
                self.actions_R['mean'], self.actions_R['m2'], self.actions_R['var']=online_update_mean_var(r+1, self.actions, self.actions_R['mean'], self.actions_R['m2'])
                self.actions_predictive_density_R['mean'], self.actions_predictive_density_R['m2'], self.actions_predictive_density_R['var']=online_update_mean_var(r+1, self.actions_predictive_density['mean'], self.actions_predictive_density_R['mean'], self.actions_predictive_density_R['m2'])
                self.n_samples_R['mean'], self.n_samples_R['m2'], self.n_samples_R['var']=online_update_mean_var(r+1, self.n_samples, self.n_samples_R['mean'], self.n_samples_R['m2'])
            else:
                self.returns_R['all'][r,0,:]=self.returns.sum(axis=0)
                self.returns_expected_R['all'][r,:,:]=self.returns_expected
                self.actions_R['all'][r,:,:]=self.actions
                self.actions_predictive_density_R['all'][r,:,:]=self.actions_predictive_density['mean']
                self.actions_predictive_density_var_R['all'][r,:,:]=self.actions_predictive_density['var']
                self.n_samples_R['all'][r,:]=self.n_samples
                
        if exec_type == 'general':
            # Compute sufficient statistics
            self.returns_R['mean']=self.returns_R['all'].mean(axis=0)
            self.returns_R['var']=self.returns_R['all'].var(axis=0)
            self.returns_expected_R['mean']=self.returns_expected_R['all'].mean(axis=0)
            self.returns_expected_R['var']=self.returns_expected_R['all'].var(axis=0)
            self.actions_R['mean']=self.actions_R['all'].mean(axis=0)
            self.actions_R['var']=self.actions_R['all'].var(axis=0)
            self.actions_predictive_density_R['mean']=self.actions_predictive_density_R['all'].mean(axis=0)
            self.actions_predictive_density_R['var']=self.actions_predictive_density_R['all'].var(axis=0)
            self.n_samples_R['mean']=self.n_samples_R['all'].mean(axis=0)
            self.n_samples_R['var']=self.n_samples_R['all'].var(axis=0)

class BayesianContextualBanditSamplingMonteCarlo(BayesianContextualBanditSampling):
    """Class for Bayesian Contextual Bandits with action sampling that compute the actions predictive density via Monte Carlo sampling
    
        These class updates the predictive density of each action using Monte Carlo sampling, based on previous rewards and actions

    Attributes (besides inherited):
        M: number of samples to use in the Monte Carlo integration
    """
    
    def __init__(self, A, reward_function, reward_prior, n_samples, M):
        """ Initialize Bayesian Bandits with public attributes 
        
        Args:
            A: size of the multi-armed bandit 
            reward_function: the reward function of the multi-armed bandit
            reward_prior: the assumed prior for the reward function of the multi-armed bandit (dictionary)
            M: number of samples to use in the Monte Carlo integration
        """
        
        # Initialize bandit (without parameters)
        super().__init__(A, reward_function, reward_prior, n_samples)
    
        # Monte carlo samples
        self.M=M
        
    def compute_action_predictive_density(self, t):
        """ Compute the predictive density of each action using Monte Carlo sampling, based on rewards and actions up until time t
            Overrides abstract method
        
        Args:
            t: time of the execution of the bandit
        """

        returns_expected_samples=np.zeros((self.A, self.M))
        if self.reward_function['type'] == 'linear_gaussian' and self.reward_prior['dist'] == 'NIG':
            for a in np.arange(self.A):
                # Sample reward's parameters (theta), given updated hyperparameters
                # First sample variance from inverse gamma
                sigma_samples=stats.invgamma.rvs(self.reward_posterior['alpha'][a], scale=self.reward_posterior['beta'][a], size=(1,self.M))
                # Then multivariate Gaussian
                reward_params_samples=self.reward_posterior['theta'][a,:][:,None]+np.sqrt(sigma_samples)*(stats.multivariate_normal.rvs(cov=self.reward_posterior['Sigma'][a,:,:], size=self.M).reshape(self.M,self.d_context).T)
            
                # Compute expected rewards, linearly combining context and sampled parameters
                returns_expected_samples[a,:]=np.dot(self.context[:,t], reward_params_samples)

        # Expected returns
        self.returns_expected[:,t]=returns_expected_samples.mean(axis=1)
                   
        # Monte Carlo integration for action predictive density
        # Mean times expected reward is maximum
        self.actions_predictive_density['mean'][:,t]=((returns_expected_samples.argmax(axis=0)[None,:]==np.arange(self.A)[:,None]).astype(int)).mean(axis=1)
        # Variance of times expected reward is maximum
        self.actions_predictive_density['var'][:,t]=((returns_expected_samples.argmax(axis=0)[None,:]==np.arange(self.A)[:,None]).astype(int)).var(axis=1)
        
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
