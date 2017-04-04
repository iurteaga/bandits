#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import scipy.special as special
from collections import defaultdict
import abc
import sys
import copy
import pickle
from Bandits import * 
import matplotlib.pyplot as plt
import pdb
# To control numpy errors
#np.seterr(all='raise')

# Class definitions
class VariationalContextualBanditSampling(abc.ABC,Bandit):
    """Class (Abstract) for Variation Contextual Bandits with action sampling
    
    These type of bandits pick, given a context, an action by sampling the next action based on the probability of the action having the highest expected return
        They update the predictive density of each action based on previous context, rewards and actions via Variational inference
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
        self.reward_posterior=copy.deepcopy(reward_prior)
         
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
        """ Update the posterior of the reward density, based on context, rewards and actions up until time t
            This function runs a variational approximation to the posterior
        Args:
            t: time of the execution of the bandit
        """
        # Linear Mixture of Gaussian with Normal Inverse Gamma conjugate prior
        if self.reward_prior['type'] == 'linear_gaussian_mixture' and self.reward_prior['dist'] == 'NIG':
            # Variational update
            lower_bound=np.zeros(self.reward_prior['variational_max_iter']+1)
            lower_bound[0]=np.finfo(float).min
            n_iter=1
            self.update_reward_posterior_variational_resp(t)
            self.update_reward_posterior_variational_params()
            lower_bound[n_iter]=self.update_reward_posterior_variational_lowerbound(t)
            print('t={}, n_iter={} with lower bound={}'.format(t, n_iter, lower_bound[n_iter]))
            # Iterate while not converged or not max iterations
            while (n_iter < self.reward_prior['variational_max_iter'] and abs(lower_bound[n_iter] - lower_bound[n_iter-1]) >= (self.reward_prior['variational_lb_eps']*abs(lower_bound[n_iter-1]))):
                n_iter+=1
                self.update_reward_posterior_variational_resp(t)
                self.update_reward_posterior_variational_params()
                lower_bound[n_iter]=self.update_reward_posterior_variational_lowerbound(t)
                print('t={}, n_iter={} with lower bound={}'.format(t, n_iter, lower_bound[n_iter]))
                #print('t={}, n_iter={} with responsibilities={}'.format(t, n_iter, np.array_str(self.r[:,:,:t+1])))
                #print('t={}, n_iter={} with thetas={}'.format(t, n_iter, np.array_str(self.reward_posterior['theta'])))
        
        # Plotting variational lower bound's evolution   
        if self.reward_prior['variational_plot_save'] is not None: 
            # Plotting    
            plt.figure()
            plt.xlabel('n_iter')
            plt.plot(np.arange(1,n_iter+1),lower_bound[1:n_iter+1], 'b', label='Lower bound')
            plt.ylabel(r'$log lower bound$')
            plt.title('Data log-lowerbound at {}'.format(t))
            legend = plt.legend(loc='upper left', ncol=1, shadow=True)
            if self.reward_prior['variational_plot_save'] == 'show': 
                plt.show()
            else:
                plt.savefig(self.reward_prior['variational_plot_save']+'/{}_loglowerbound_t{}.pdf'.format(self.__class__.__name__,t), format='pdf', bbox_inches='tight')
            plt.close()
                    
        # TODO: Add other reward/prior combinations
        else:
            raise ValueError('Invalid reward_function={} with reward_prior={} combination'.format(self.reward_prior['type'], self.reward_prior['dist']))

    def update_reward_posterior_variational_resp(self, t):
        # Compute rho
        rho=( -0.5*(np.log(self.reward_posterior['beta'])-special.digamma(self.reward_posterior['alpha']))[:,:,None]
            - 0.5*(np.einsum('it,akij,jt->akt', self.context[:,:t+1], self.reward_posterior['Sigma'], self.context[:,:t+1])+np.power(self.returns[:,:t+1].sum(axis=0)-np.einsum('it,aki->akt', self.context[:,:t+1], self.reward_posterior['theta']),2)*self.reward_posterior['alpha'][:,:,None]/self.reward_posterior['beta'][:,:,None])
            + (special.digamma(self.reward_posterior['gamma'])-special.digamma(self.reward_posterior['gamma'].sum(axis=1,keepdims=True)))[:,:,None]
            )
        # Exponentiate (and try to avoid numerical errors)
        r=np.exp(rho-rho.max(axis=1, keepdims=True))
        # And normalize
        self.r[:,:,:t+1]=r/(r.sum(axis=1, keepdims=True))
        
    def update_reward_posterior_variational_params(self):
        # For each arm
        for a in np.arange(self.A):
            # Pick times when arm was played
            this_a=self.actions[a,:]==1
            # For each mixture component            
            for k in np.arange(self.reward_prior['K']):
                # Its responsibilities
                R_ak=np.diag(self.r[a,k,this_a])
                # Update Gamma
                self.reward_posterior['gamma'][a,k]=self.reward_prior['gamma'][a,k]+self.r[a,k,this_a].sum()
                # Update Sigma
                self.reward_posterior['Sigma'][a,k,:,:]=np.linalg.inv(np.linalg.inv(self.reward_prior['Sigma'][a,k,:,:])+np.dot(np.dot(self.context[:,this_a], R_ak), self.context[:,this_a].T))
                # Update and append Theta
                self.reward_posterior['theta'][a,k,:]=np.dot(self.reward_posterior['Sigma'][a,k,:,:], np.dot(np.linalg.inv(self.reward_prior['Sigma'][a,k,:,:]), self.reward_prior['theta'][a,k,:])+np.dot(np.dot(self.context[:,this_a], R_ak), self.returns[a,this_a].T))
                # Update and append alpha
                self.reward_posterior['alpha'][a,k]=self.reward_prior['alpha'][a,k]+(self.r[a,k,this_a].sum())/2
                # Update and append beta
                self.reward_posterior['beta'][a,k]=self.reward_prior['beta'][a,k]+1/2*(
                np.dot(np.dot(self.returns[a,this_a], R_ak), self.returns[a,this_a].T) +
                np.dot(self.reward_prior['theta'][a,k,:].T, np.dot(np.linalg.inv(self.reward_prior['Sigma'][a,k,:,:]), self.reward_prior['theta'][a,k,:])) -
                np.dot(self.reward_posterior['theta'][a,k,:].T, np.dot(np.linalg.inv(self.reward_posterior['Sigma'][a,k,:,:]), self.reward_posterior['theta'][a,k,:]))
                )
        
    def update_reward_posterior_variational_lowerbound(self,t):
        # Indicators for picked arms
        tIdx, aIdx=np.where(self.actions.T)
        # E{ln f(Y|Z,\theta,\sigma)}
        tmp=( (np.log(2*np.pi)+np.log(self.reward_posterior['beta'])-special.digamma(self.reward_posterior['alpha']))[:,:,None] 
                + np.einsum('it,akij,jt->akt', self.context[:,:t+1], self.reward_posterior['Sigma'], self.context[:,:t+1])
                + np.power(self.returns[:,:t+1].sum(axis=0)-np.einsum('it,aki->akt', self.context[:,:t+1], self.reward_posterior['theta']),2) * self.reward_posterior['alpha'][:,:,None]/self.reward_posterior['beta'][:,:,None] )
        # Sum over mixture and time indicators
        E_ln_fY=-0.5*(self.r[aIdx,:,tIdx]*tmp[aIdx,:,tIdx]).sum(axis=(1,0))
        
        # E{ln p(Z|\pi}
        tmp=self.r[:,:,:t+1]*(special.digamma(self.reward_posterior['gamma'])-special.digamma(self.reward_posterior['gamma'].sum(axis=1, keepdims=True)))[:,:,None]
        # Sum over mixture and arm-time indicators
        E_ln_pZ=tmp[aIdx,:,tIdx].sum(axis=(1,0))
    
        # E{ln f(\pi}
        tmp=( special.gammaln(self.reward_prior['gamma'].sum(axis=1, keepdims=True))-special.gammaln(self.reward_prior['gamma']).sum(axis=1,keepdims=True)
                + ((self.reward_prior['gamma']-1)*(special.digamma(self.reward_posterior['gamma'])-special.digamma(self.reward_posterior['gamma'].sum(axis=1, keepdims=True)))).sum(axis=1,keepdims=True) )
        # Sum over all arms
        E_ln_fpi=tmp.sum(axis=0)
        
        # E{ln f(\theta,\sigma}
        tmp=( self.reward_prior['alpha']*np.log(self.reward_prior['beta'])-special.gammaln(self.reward_prior['alpha'])-self.d_context/2*np.log(2*np.pi)-0.5*np.linalg.det(self.reward_prior['Sigma'])
                - (self.d_context/2+self.reward_prior['alpha']+1)*(np.log(self.reward_posterior['beta'])-special.digamma(self.reward_posterior['alpha']))
                - self.reward_prior['beta']*self.reward_posterior['alpha']/self.reward_posterior['beta']
                - 0.5*( np.einsum('akij->ak', np.einsum('akij,akjl->akil',np.linalg.inv(self.reward_prior['Sigma']), self.reward_posterior['Sigma']))
                        + np.einsum('aki,akij,akj->ak', (self.reward_posterior['theta']-self.reward_prior['theta']), np.linalg.inv(self.reward_prior['Sigma']), (self.reward_posterior['theta']-self.reward_prior['theta'])) * self.reward_posterior['alpha']/self.reward_posterior['beta']) )
        # Sum over mixtures and arms
        E_ln_fthetasigma=tmp.sum(axis=(1,0))
        
        # E{ln q(Z|\pi}
        tmp=self.r[:,:,:t+1]*np.log(self.r[:,:,:t+1])
        # Sum over mixture and arm-time indicators
        E_ln_qZ=tmp[aIdx,:,tIdx].sum(axis=(1,0))
        
        # E{ln q(\pi}
        tmp=( special.gammaln(self.reward_posterior['gamma'].sum(axis=1, keepdims=True))-special.gammaln(self.reward_posterior['gamma']).sum(axis=1, keepdims=True)
                + ((self.reward_posterior['gamma']-1)*(special.digamma(self.reward_posterior['gamma'])-special.digamma(self.reward_posterior['gamma'].sum(axis=1, keepdims=True)))).sum(axis=1, keepdims=True) )
        # Sum over all arms
        E_ln_qpi=tmp.sum(axis=0)
        
        # E{ln q(\theta,\sigma}
        tmp=( self.reward_posterior['alpha']*np.log(self.reward_posterior['beta'])-special.gammaln(self.reward_posterior['alpha'])-self.d_context/2*np.log(2*np.pi)-0.5*np.linalg.det(self.reward_posterior['Sigma'])
                - (self.d_context/2+self.reward_posterior['alpha']+1)*(np.log(self.reward_posterior['beta'])-special.digamma(self.reward_posterior['alpha'])) - self.reward_posterior['alpha'] - self.d_context/2 )
        # Sum over mixtures and arms
        E_ln_qthetasigma=tmp.sum(axis=(1,0))
        
        # Return lower bound
        return E_ln_fY + E_ln_pZ + E_ln_fpi + E_ln_fthetasigma - E_ln_qZ - E_ln_qpi - E_ln_qthetasigma
        
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
        self.r=np.zeros((self.A, self.reward_prior['K'], t_max))
                
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
            if self.reward_function['type'] == 'linear_gaussian':
                # Draw from weighted context
                self.returns[action,t]=self.reward_function['dist'].rvs(loc=np.dot(self.context[:,t],self.reward_function['theta'][action,:]), scale=self.reward_function['sigma'][action])
            elif self.reward_function['type'] == 'linear_gaussian_mixture':
                # First, pick mixture
                mixture=np.where(np.random.multinomial(1,self.reward_function['pi'][action]))[0][0]
                self.returns[action,t]=self.reward_function['dist'].rvs(loc=np.dot(self.context[:,t],self.reward_function['theta'][action,mixture,:]), scale=self.reward_function['sigma'][action,mixture])
            else:
                raise ValueError('Invalid reward_function={}'.format(self.reward_function['type']))
            # Update parameter posteriors
            self.update_reward_posterior(t)
        
    def execute_realizations(self, R, context, t_max, exec_type='online', save_bandits=None):
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

            # If required
            if save_bandits is not None:
                # Save bandits info
                with open(save_bandits+'/{}_{}_n{}_M{}_priorK{}_r{}.pickle'.format(self.__class__.__name__, self.sampling['type'], self.sampling['n_samples'], self.M, self.reward_prior['K'], r), 'wb') as f:
                    pickle.dump(self, f)

            # Compute sufficient statistics
            # In online fashion            
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

        # Compute sufficient statistics
        # If in general
        if exec_type == 'general':
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

class VariationalContextualBanditSampling_actionMonteCarlo(VariationalContextualBanditSampling):
    """Class for Variational Contextual Bandits Sampling with Monte Carlo over actions
    
        This class updates the predictive action density of each action using Monte Carlo sampling, based on previous rewards and actions        
        - Draw parameters from the variational posterior
        - Compute the expected return for each parameter sample
        - Decide, for each sample, which action is the best
        - Compute the action predictive density, as a Monte Carlo by averaging best action samples

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
        # TODO: generalize without per arm/K iteration?
        returns_expected_samples=np.zeros((self.A, self.M))
        if self.reward_prior['type'] == 'linear_gaussian_mixture' and self.reward_prior['dist'] == 'NIG':
            for a in np.arange(self.A):
                returns_expected_per_mixture_samples=np.zeros((self.reward_prior['K'], self.M))
                for k in np.arange(self.reward_prior['K']):
                    # Sample reward's parameters (theta), given updated hyperparameters
                    # First sample variance from inverse gamma for each mixture
                    sigma_samples=stats.invgamma.rvs(self.reward_posterior['alpha'][a,k], scale=self.reward_posterior['beta'][a,k], size=(1,self.M))
                    # Then multivariate Gaussian
                    theta_samples=self.reward_posterior['theta'][a,k,:][:,None]+np.sqrt(sigma_samples)*(stats.multivariate_normal.rvs(cov=self.reward_posterior['Sigma'][a,k,:,:], size=self.M).reshape(self.M,self.d_context).T)
                    # Compute expected reward, linearly combining context and sampled parameters per mixture
                    returns_expected_per_mixture_samples[k,:]=np.dot(self.context[:,t], theta_samples)
                
                # Sample mixture proportions
                pi_samples=stats.dirichlet.rvs(self.reward_posterior['gamma'][a], size=self.M).T
                # Compute expected rewards, by averaging over mixture proportions
                returns_expected_samples[a,:]=np.einsum('km,km->m', pi_samples, returns_expected_per_mixture_samples)

        # Expected returns
        self.returns_expected[:,t]=returns_expected_samples.mean(axis=1)
                   
        # Monte Carlo integration for action predictive density
        # Mean times expected reward is maximum
        self.actions_predictive_density['mean'][:,t]=((returns_expected_samples.argmax(axis=0)[None,:]==np.arange(self.A)[:,None]).astype(int)).mean(axis=1)
        # Variance of times expected reward is maximum
        self.actions_predictive_density['var'][:,t]=((returns_expected_samples.argmax(axis=0)[None,:]==np.arange(self.A)[:,None]).astype(int)).var(axis=1)
        
class VariationalContextualBanditSampling_actionMonteCarlo_zSampling(VariationalContextualBanditSampling):
    """Class for Variational Contextual Bandits Sampling with Monte Carlo over actions
    
        This class updates the predictive action density of each action using Monte Carlo sampling, based on previous rewards and actions        
        - Draw parameters from the variational posterior
        - Draw mixture assignment Z with drawn parameters form Dirichlet-Multinomial 
        - Compute the expected return for the selected mixture
        - Decide, for each sample, which action is the best
        - Compute the action predictive density, as a Monte Carlo by averaging best action samples

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
        # TODO: generalize without per arm/K iteration?
        returns_expected_samples=np.zeros((self.A, self.M))
        if self.reward_prior['type'] == 'linear_gaussian_mixture' and self.reward_prior['dist'] == 'NIG':
            for a in np.arange(self.A):
                returns_expected_per_mixture_samples=np.zeros((self.reward_prior['K'], self.M))
                for k in np.arange(self.reward_prior['K']):
                    # Sample reward's parameters (theta), given updated hyperparameters
                    # First sample variance from inverse gamma for each mixture
                    sigma_samples=stats.invgamma.rvs(self.reward_posterior['alpha'][a,k], scale=self.reward_posterior['beta'][a,k], size=(1,self.M))
                    # Then multivariate Gaussian
                    theta_samples=self.reward_posterior['theta'][a,k,:][:,None]+np.sqrt(sigma_samples)*(stats.multivariate_normal.rvs(cov=self.reward_posterior['Sigma'][a,k,:,:], size=self.M).reshape(self.M,self.d_context).T)
                    # Compute expected reward, linearly combining context and sampled parameters per mixture
                    returns_expected_per_mixture_samples[k,:]=np.dot(self.context[:,t], theta_samples)
                
                # Sample mixture proportions, from Dirichlet multinomial
                k_prob=self.reward_posterior['gamma'][a]/(self.reward_posterior['gamma'][a].sum())
                z_samples=np.random.multinomial(1,k_prob, size=self.M).T
                # Compute expected rewards, for each of the picked mixture
                    # Transposed used due to python indexing
                returns_expected_samples[a,:]=returns_expected_per_mixture_samples.T[z_samples.T==1]

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
