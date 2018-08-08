#!/usr/bin/python
    
# Imports: python modules
import numpy as np
import scipy.stats as stats
import pdb

######## Class definition ########
class MonteCarloPosterior(object):
    """ Class for computation of bandit posteriors via Monte Carlo
        - Parameter posterior updates, based on random measure approximations
        - Works with Bandit objects, by inheritance

    Attributes (required and inherited from Bandit):
        reward_function: dictionary with information about the reward distribution and its parameters
        reward_prior: the assumed prior for the multi-armed bandit's reward function
        reward_posterior: the posterior for the learned multi-armed bandit's reward function
        context: d_context by (at_least) t_max array with context for every time instant (None if does not apply)
        actions: the actions that the bandit takes (per realization) as A by t_max array
        rewards: rewards obtained by each arm of the bandit (per realization) as A by t_max array
    """
    def init_reward_posterior(self):
        """ Initialize the posterior of the reward density
            Following random measure approximation to posterior

        Args:
            None
        """
        ### Allocate reward posterior with weights and samples
        if np.all(self.context != None):
            if self.reward_function['type'] == 'linear_gaussian' and ('alpha' in self.reward_prior and 'beta' in self.reward_prior):
                # Reward posterior with theta regressors (size of context) and variance
                self.reward_posterior={'weights': np.ones((self.A, self.reward_prior['M']))/self.reward_prior['M'], 'theta': np.zeros((self.A, self.reward_prior['M'], self.d_context)), 'sigma': np.zeros((self.A, self.reward_prior['M'],1))}
            else:
                # Reward posterior with theta regressors (size of context)
                self.reward_posterior={'weights': np.ones((self.A, self.reward_prior['M']))/self.reward_prior['M'], 'theta': np.zeros((self.A, self.reward_prior['M'], self.d_context))}
        else:
            # Reward posterior with no context
            self.reward_posterior={'weights': np.ones((self.A, self.reward_prior['M']))/self.reward_prior['M'], 'theta': np.zeros((self.A, self.reward_prior['M'],1))}
                    
        ### Initialize from priors
        # Binomial/Bernoulli reward
        if self.reward_function['type'] == 'bernoulli' and self.reward_prior['dist'].name == 'beta':
            self.reward_posterior['theta'][:,:,0]=self.reward_prior['dist'].rvs(self.reward_prior['alpha'],self.reward_prior['beta'],size=(self.A, self.reward_prior['M']))
        # Linear Gaussian reward
        elif self.reward_function['type'] == 'linear_gaussian' and self.reward_prior['dist'] == 'NIG':
            if 'alpha' in self.reward_prior and 'beta' in self.reward_prior:
                # Draw variance samples from inverse gamma
                self.reward_posterior['sigma'][:,:,0]=stats.invgamma.rvs(self.reward_prior['alpha'], scale=self.reward_prior['beta'], size=(self.A,self.reward_prior['M']))

                # Then multivariate Gaussian parameters
                self.reward_posterior['theta']=self.reward_prior['theta'][:,None,:]+np.sqrt(self.reward_posterior['sigma'])*np.einsum('aid,md->ami',np.linalg.cholesky(self.reward_prior['Sigma']), stats.norm.rvs(size=(self.reward_prior['M'],self.reward_prior['theta'].shape[1])))
            else:
                # Multivariate Gaussian parameters, with known sigma
                self.reward_posterior['theta']=self.reward_prior['theta'][:,None,:]+np.einsum('aid,md->ami',np.linalg.cholesky(self.reward_prior['Sigma']), stats.norm.rvs(size=(self.reward_prior['M'],self.reward_prior['theta'].shape[1])))
                
        # Logistic reward
        elif self.reward_function['type'] == 'logistic' and self.reward_prior['dist'] == 'Gaussian':
            # Multivariate Gaussian parameter prior
            self.reward_posterior['theta']=self.reward_prior['theta'][:,None,:]+np.einsum('aid,md->ami',np.linalg.cholesky(self.reward_prior['Sigma']), stats.norm.rvs(size=(self.reward_prior['M'],self.reward_prior['theta'].shape[1])))
        else:
            raise ValueError('Invalid reward_function={} with reward_prior={} combination'.format(self.reward_function['type'], self.reward_prior['dist'].name))

        ### Initialize with equal weights
        self.reward_posterior['weights']=np.ones((self.A,self.reward_prior['M']))/self.reward_prior['M']
        
        ## For unknown dynamics
        if 'dynamics' in self.reward_function and self.reward_function['dynamics']=='linear_mixing_unknown':
            # Allocate space to keep track of parameter evolution
            self.allTheta=np.zeros((self.A, self.reward_prior['M'], self.reward_prior['theta'].shape[1], self.actions.shape[1]))
            
            # Make sure all prior information is provided
            assert ('A_0' in self.reward_prior) and ('Lambda_0' in self.reward_prior) and ('nu_0' in self.reward_prior) and ('C_0' in self.reward_prior), 'Missing prior information for linear_mixing_unknown parameters'
            # Precompute
            self.reward_prior['A_0Lambda_0']=np.einsum('adb,abc->adc', self.reward_prior['A_0'], self.reward_prior['Lambda_0'])
        
    def update_reward_posterior(self, t, update_type='sequential'):
        """ Update the posterior of the reward density, based on available information at time t
            Update the posterior following a Bayesian approach with conjugate priors

        Args:
            t: time of the execution of the bandit
            update='sequential' or 'batch' update for posteriors
        """

        # TODO: should we consider batch updates?
        assert update_type=='sequential', 'Only sequential Monte Carlo posterior updates implemented'
        
        #### SAMPLING
        # Draw M sample indexes from previous posterior random measure for resampling
        # TODO: Add resampling ESS condition
        m_a=(np.random.rand(self.A,self.reward_prior['M'],1)>self.reward_posterior['weights'].cumsum(axis=1)[:,None,:]).sum(axis=2)
        
        ## Only for STATIC parameters (dynamic parameters have already been propagated)
        if 'dynamics' not in self.reward_function:
            # Resampled
            theta_resampled=self.reward_posterior['theta'][np.arange(self.A)[:,None]*np.ones((1,self.reward_prior['M']),dtype=int),m_a]
            
            # Resampling
            if self.reward_prior['sampling']=='resampling':
                # Just resampling
                self.reward_posterior['theta']=theta_resampled
                
            # Random-walk
            elif self.reward_prior['sampling']=='random_walk':
                # Draw from Gaussian centered in resampled 
                self.reward_posterior['theta']=theta_resampled + np.sqrt(self.reward_prior['sampling_sigma'])*stats.norm.rvs(size=self.reward_posterior['theta'].shape)

            # Kernel-based
            elif self.reward_prior['sampling']=='kernel':
                # Compute kernel sufficient statistics
                kernel_theta=self.reward_prior['sampling_alpha'] * self.reward_posterior['theta'] + (1-self.reward_prior['sampling_alpha']) * np.einsum('am,amd->ad', self.reward_posterior['weights'], self.reward_posterior['theta'])[:,None,:]
                theta_diff=self.reward_posterior['theta']-kernel_theta
                kernel_Sigma=np.einsum('am,amdb->adb', self.reward_posterior['weights'], theta_diff[:,:,:,None]*theta_diff[:,:,None,:])
                # TODO: can we vectorize over A?
                for a in np.arange(self.A):
                    kernel_Sigma[a,:,:]+=np.diag(np.diag(kernel_Sigma[a,:,:])<self.reward_prior['sampling_sigma'])*self.reward_prior['sampling_sigma']
                # Double check valid covariance matrix (TODO: better/more elegant checking scheme)
                kernel_Sigma[np.any(np.isinf(kernel_Sigma), axis=(1,2)) | np.any(np.isnan(kernel_Sigma), axis=(1,2))]=self.reward_prior['sampling_sigma']*np.eye(kernel_Sigma.shape[2])
                # positive-definite
                kernel_Sigma[np.any(np.linalg.eigvals(kernel_Sigma)<=0., axis=1)]=self.reward_prior['sampling_sigma']*np.eye(kernel_Sigma.shape[2])
                
                # Draw from kernel centered in resampled
                self.reward_posterior['theta']=theta_resampled + np.einsum('adb,amb->amd', np.linalg.cholesky(kernel_Sigma), stats.norm.rvs(size=self.reward_posterior['theta'].shape))
            
            # Density assisted
            elif self.reward_prior['sampling']=='density':
                # Compute density sufficient statistics
                density_theta=np.einsum('am,amd->ad', self.reward_posterior['weights'], self.reward_posterior['theta'])[:,None,:]
                theta_diff=self.reward_posterior['theta']-density_theta
                density_Sigma=np.einsum('am,amdb->adb', self.reward_posterior['weights'], theta_diff[:,:,:,None]*theta_diff[:,:,None,:])
                # Just to have a minimum variance
                # TODO: can we vectorize over A?
                for a in np.arange(self.A):
                    density_Sigma[a,:,:]+=np.diag(np.diag(density_Sigma[a,:,:])<self.reward_prior['sampling_sigma'])*self.reward_prior['sampling_sigma']
                # Double check valid covariance matrix (TODO: better/more elegant checking scheme)
                density_Sigma[np.any(np.isinf(density_Sigma), axis=(1,2)) | np.any(np.isnan(density_Sigma), axis=(1,2))]=self.reward_prior['sampling_sigma']*np.eye(density_Sigma.shape[2])
                # positive-definite
                density_Sigma[np.any(np.linalg.eigvals(density_Sigma)<=0., axis=1)]=self.reward_prior['sampling_sigma']*np.eye(density_Sigma.shape[2])
                
                # Draw from Gaussian density
                self.reward_posterior['theta']=density_theta + np.einsum('adb,amb->amd', np.linalg.cholesky(density_Sigma), stats.norm.rvs(size=self.reward_posterior['theta'].shape))

            else:
                raise ValueError('Invalid reward_posterior regressor sampling={}'.format(self.reward_prior['sampling']))
        
        # If reward variance is unknown
        # Different propagation to account for support (0,\infty)
        if self.reward_function['type'] == 'linear_gaussian' and ('alpha' in self.reward_prior and 'beta' in self.reward_prior):
            # Resample variance
            sigma_resampled=self.reward_posterior['sigma'][np.arange(self.A)[:,None]*np.ones((1,self.reward_prior['M']),dtype=int),m_a]
            
            # Resampling
            if self.reward_prior['sampling']=='resampling':
                # Just resampling
                self.reward_posterior['sigma']=sigma_resampled
            
            # Density assisted
            elif self.reward_prior['sampling']=='density':
                # Compute sufficient statistics of samples
                sigma_mean=np.einsum('am,amd->ad', self.reward_posterior['weights'], self.reward_posterior['sigma'])
                sigma_diff=self.reward_posterior['sigma']-sigma_mean[:,None,:]
                sigma_var=np.einsum('am,amd->ad', self.reward_posterior['weights'], sigma_diff*sigma_diff)
                # Just to have a minimum variance
                sigma_var+=(sigma_var<self.reward_prior['sampling_sigma'])*self.reward_prior['sampling_sigma']
                # Draw from inverse gamma, by matching sample suff stats to params
                tmp=np.power(sigma_mean,2)/np.power(sigma_var,2)
                self.reward_posterior['sigma'][:,:,0]=stats.invgamma.rvs(tmp+2, scale=(tmp+1)*sigma_mean, size=(self.A,self.reward_prior['M']))
            else:
                raise ValueError('Invalid reward_posterior variance sampling={}'.format(self.reward_prior['sampling']))

            assert np.all(self.reward_posterior['sigma']>=0), 'Can not have negative reward variance'

        #### WEIGHTING
        # Equal weights due to resampling
        self.reward_posterior['weights']=np.ones((self.A,self.reward_prior['M']))
        # Zero weight for non-sense
        self.reward_posterior['weights'][np.any(np.isinf(self.reward_posterior['theta']),axis=2) | np.any(np.isnan(self.reward_posterior['theta']),axis=2)]=0.
        # Based on likelihood of reward for played arm
        a_played=self.actions[:,t]==1
        # Binomial/Bernoulli reward
        if self.reward_function['type'] == 'bernoulli':
            # TODO: rethink this hack
            self.reward_posterior['theta'][self.reward_posterior['theta']<0]=0.
            self.reward_posterior['theta'][self.reward_posterior['theta']>1]=1.
            # Weights are p if y=1, (1-p) if y=0
            self.reward_posterior['weights'][a_played]=self.reward_posterior['theta'][a_played,:,0] if self.rewards[a_played,t] else (1-self.reward_posterior['theta'][a_played,:,0])

        # Linear Gaussian reward
        elif self.reward_function['type'] == 'linear_gaussian':
            # Mean diff
            y_mean=np.einsum('d,amd->am', self.context[:,t], self.reward_posterior['theta'][a_played])
            # Compute loglikelihood
            if 'alpha' in self.reward_prior and 'beta' in self.reward_prior:
                log_weights=-0.5*(np.log(self.reward_posterior['sigma'][a_played,:,0]**2)+(self.rewards[a_played,t]-y_mean)**2/self.reward_posterior['sigma'][a_played,:,0]**2)
            else:
                # Compute loglikelihood
                log_weights=-0.5*(np.log(self.reward_function['sigma'][a_played]**2)+(self.rewards[a_played,t]-y_mean)**2/self.reward_function['sigma'][a_played]**2)
            # Double-check
            log_weights[np.any(np.isinf(self.reward_posterior['theta'][a_played]),axis=2) | np.any(np.isnan(self.reward_posterior['theta'][a_played]),axis=2)]=-np.inf
            log_weights[np.isnan(log_weights)]=-np.inf
            # Linear weights
            self.reward_posterior['weights'][a_played]=np.exp(log_weights-np.max(log_weights))
            self.reward_posterior['weights'][a_played][np.isnan(self.reward_posterior['weights'][a_played])]=np.finfo(float).eps
            
        # Logistic reward
        elif self.reward_function['type'] == 'logistic':
            xTheta=np.einsum('d,amd->am', self.context[:,t], self.reward_posterior['theta'][a_played])
            log_weights=self.rewards[a_played,t]*xTheta-np.log(1+np.exp(xTheta))
            # Double-check
            log_weights[np.any(np.isinf(self.reward_posterior['theta'][a_played]),axis=2) | np.any(np.isnan(self.reward_posterior['theta'][a_played]),axis=2)]=-np.inf
            log_weights[np.isnan(log_weights)]=-np.inf
            # Linear weights
            self.reward_posterior['weights'][a_played]=np.exp(log_weights-np.max(log_weights))
            self.reward_posterior['weights'][a_played][np.isnan(self.reward_posterior['weights'][a_played])]=np.finfo(float).eps
        else:
            raise ValueError('Invalid reward_function={}'.format(self.reward_function['type']))
            
        # Normalize weights
        self.reward_posterior['weights']/=self.reward_posterior['weights'].sum(axis=1,keepdims=True)
        self.reward_posterior['weights'][a_played][np.isnan(self.reward_posterior['weights'][a_played])]=np.finfo(float).eps
        
        ### Validating Random Measure
        # TODO: reevaluate this approach
        if not (np.all(self.reward_posterior['weights']>=0) and np.allclose(self.reward_posterior['weights'].sum(axis=1), np.ones(self.A))):
            # PF has diverged/numerical issues
            print('Invalid weights, reinitialize')
            # Reinit with priors and equal weights
            self.reward_posterior['weights']=np.ones((self.A,self.reward_prior['M']))/self.reward_prior['M']
            if self.reward_function['type'] == 'bernoulli' and self.reward_prior['dist'].name == 'beta':
                self.reward_posterior['theta'][:,:,0]=self.reward_prior['dist'].rvs(self.reward_prior['alpha'],self.reward_prior['beta'],size=(self.A, self.reward_prior['M']))
            # Linear Gaussian reward
            elif self.reward_function['type'] == 'linear_gaussian' and self.reward_prior['dist'] == 'NIG':
                if 'alpha' in self.reward_prior and 'beta' in self.reward_prior:
                    # Draw variance samples from inverse gamma
                    self.reward_posterior['sigma'][:,:,0]=stats.invgamma.rvs(self.reward_prior['alpha'], scale=self.reward_prior['beta'], size=(self.A,self.reward_prior['M']))
                    # Then multivariate Gaussian parameters
                    self.reward_posterior['theta']=self.reward_prior['theta'][:,None,:]+np.sqrt(self.reward_posterior['sigma'])*np.einsum('aid,md->ami',np.linalg.cholesky(self.reward_prior['Sigma']), stats.norm.rvs(size=(self.reward_prior['M'],self.reward_prior['theta'].shape[1])))
                else:
                    # Multivariate Gaussian parameters, with known sigma
                    self.reward_posterior['theta']=self.reward_prior['theta'][:,None,:]+np.einsum('aid,md->ami',np.linalg.cholesky(self.reward_prior['Sigma']), stats.norm.rvs(size=(self.reward_prior['M'],self.reward_prior['theta'].shape[1])))
            # Logistic reward
            elif self.reward_function['type'] == 'logistic' and self.reward_prior['dist'] == 'Gaussian':
                # Multivariate Gaussian parameter prior
                self.reward_posterior['theta']=self.reward_prior['theta'][:,None,:]+np.einsum('aid,md->ami',np.linalg.cholesky(self.reward_prior['Sigma']), stats.norm.rvs(size=(self.reward_prior['M'],self.reward_prior['theta'].shape[1])))
            else:
                raise ValueError('Invalid reward_function={} with reward_prior={} combination'.format(self.reward_function['type'], self.reward_prior['dist'].name))
            

# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
