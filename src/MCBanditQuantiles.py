#!/usr/bin/python

# Imports: python modules
import copy
import pdb
# Imports: other modules
from BanditQuantiles import * 
from MonteCarloPosterior import *

######## Class definition ########
class MCBanditQuantiles(BanditQuantiles, MonteCarloPosterior):
    """ Class for bandits with policies based on quantiles (upper-confidence bounds)
        These bandits decide which arm to play by picking the arm with the highest expected reward quantile
            - Quantiles are computed based on random measure (MC) approximations to parameter posteriors
            - Only empirical quantile computation approach is considered

    Attributes (besides inherited):
        reward_prior: the assumed prior for the multi-armed bandit's reward function
        reward_posterior: the posterior for the learned multi-armed bandit's reward function
        quantile_info: information on how to compute the quantile and which parameters to use
            quantile_info['alpha']: information on quantile parameter alpha to use (might depend on t)
            quantile_info['type'] == empirical (based on MC) or sampling based
                quantile_info['n_samples']: number of samples for sampling-based quantile estimation
        arm_quantile: quantile values for each arm at each time instant
    """
    
    def __init__(self, A, reward_function, reward_prior, quantile_info):
        """ Initialize the Bandit object and its attributes
        
        Args:
            A: the size of the bandit
            reward_function: the reward function of the bandit
            reward_prior: the assumed prior for the multi-armed bandit's reward function
            quantile_info: information on how to compute the quantile and which parameters to use
            quantile_info['alpha']: information on quantile parameter alpha to use (might depend on t)
            quantile_info['type'] == empirical (based on MC) or sampling based
                quantile_info['n_samples']: number of samples for sampling-based quantile estimation
        """
        
        # Initialize
        super().__init__(A, reward_function, reward_prior, quantile_info)
            
    def compute_arm_quantile(self, t):
        """ Method to compute the quantile values for each arm, based on available information at time t, which depends on posterior update type
                Monte Carlo posterior updates

            Different alternatives on computing the quantiles are considered
                quantile_info['type']='empirical': use the MC posterior to compute the quantile (inverse CDF) function of the expected rewards 
                quantile_info['type']='sampling': estimate the quantile (inverse CDF) function by sampling expected reward samples
        Args:
            t: time of the execution of the bandit
        """
        ### Data preallocation
        # Posterior theta
        theta_loc=copy.deepcopy(self.reward_posterior['theta'])
        # Posterior weights
        posterior_weights=copy.deepcopy(self.reward_posterior['weights'])
        
        ### Propagation of dynamic parameters for arm quantile computation
        # Update particles with parameter dynamics
        # TODO: oversampling?
        if 'dynamics' in self.reward_function:
            
            # TODO: Add resampling ESS condition
            # Draw M resampling indexes from previous posterior random measure
            m_a=(np.random.rand(self.A,self.reward_prior['M'],1)>self.reward_posterior['weights'].cumsum(axis=1)[:,None,:]).sum(axis=2)

            # Linear mixing dynamics, with known parameters
            if self.reward_function['dynamics']=='linear_mixing_known':
                # Draw from transition density: Gaussian with known parameters
                # Propagated mean
                theta_loc=np.einsum('adb,amb->amd', self.reward_function['dynamics_A'], self.reward_posterior['theta'])
                # Draw from Gaussian (with resampled mean)
                self.reward_posterior['theta']=theta_loc[np.arange(self.A)[:,None]*np.ones((1,self.reward_prior['M']),dtype=int),m_a]+ np.einsum('adb,amb->amd', np.linalg.cholesky(self.reward_function['dynamics_C']), stats.norm.rvs(size=self.reward_posterior['theta'].shape))
            
            # Linear mixing dynamics, with unknown parameters
            elif self.reward_function['dynamics']=='linear_mixing_unknown':
                # Add latest particles to whole stream
                self.allTheta[:,:,:,t]=self.reward_posterior['theta']
                # Degrees of freedom for transition density
                nu=(self.reward_prior['nu_0']+t-self.reward_posterior['theta'].shape[2]+1)
                
                # Only after observing enough data
                if t>=2*self.reward_posterior['theta'].shape[2] and nu>0:
                    # We compute all sufficient statistics after resampling, to avoid computation of negligible (small weighted) streams
                    # Keep track of whole stream after resampling
                    self.allTheta[:,:,:,:t+1]=self.allTheta[np.arange(self.A)[:,None]*np.ones((1,self.reward_prior['M']),dtype=int),m_a,:,:t+1]
      
                    # Data products
                    ZZtop=np.einsum('amdt,ambt->amdb', self.allTheta[:,:,:,0:t-1], self.allTheta[:,:,:,0:t-1])
                    XZtop=np.einsum('amdt,ambt->amdb', self.allTheta[:,:,:,1:t], self.allTheta[:,:,:,0:t-1])
                    # Updated Lambda
                    Lambda_N=ZZtop+self.reward_prior['Lambda_0'][:,None,:,:]
                    # Double check invertibility
                    Lambda_N[np.any(np.isinf(Lambda_N), axis=(2,3)) | np.any(np.isnan(Lambda_N), axis=(2,3))]=self.reward_prior['sampling_sigma']*np.eye(self.reward_posterior['theta'].shape[2])
                    Lambda_N[np.linalg.det(Lambda_N)==0]=self.reward_prior['sampling_sigma']*np.eye(self.reward_posterior['theta'].shape[2])

                    # Estimated A
                    A_hat=np.einsum('amdb,ambc->amdc', XZtop + self.reward_prior['A_0Lambda_0'][:,None,:,:], np.linalg.inv(Lambda_N))
                    # Propagated mean (already resampled)
                    theta_loc=np.einsum('amdb,amb->amd', A_hat, self.allTheta[:,:,:,t])
                    
                    # Auxiliary data for correlation matrix
                    diff_1=self.allTheta[:,:,:,1:t]-np.einsum('amdb,ambt->amdt', A_hat, self.allTheta[:,:,:,0:t-1])
                    tmp_1=np.einsum('amdt,ambt->amdb', diff_1, diff_1)
                    diff_2=A_hat-self.reward_prior['A_0'][:,None,:,:]
                    tmp_2=np.einsum('amdb,abc,amec->amde', diff_2, self.reward_prior['Lambda_0'], diff_2)
                    UUtop=Lambda_N+np.einsum('amd,amb->amdb',self.allTheta[:,:,:,t],self.allTheta[:,:,:,t])
                    # Double check invertibility
                    UUtop[np.any(np.isinf(UUtop), axis=(2,3)) | np.any(np.isnan(UUtop), axis=(2,3))]=self.reward_prior['sampling_sigma']*np.eye(self.reward_posterior['theta'].shape[2])
                    UUtop[np.linalg.det(UUtop)==0]=self.reward_prior['sampling_sigma']*np.eye(self.reward_posterior['theta'].shape[2])
                    
                    # Estimated covariance
                    C_N=self.reward_prior['C_0'][:,None,:,:]+tmp_1+tmp_2
                    den=1-np.einsum('amd,amdb,amb->am', self.allTheta[:,:,:,t], np.linalg.inv(UUtop), self.allTheta[:,:,:,t])
                    # Correlation matrix
                    t_cov=C_N/(nu*den)[:,:,None,None]
                    # Double check valid covariance matrix (TODO: better/more elegant checking scheme)
                    t_cov[np.any(np.isinf(t_cov), axis=(2,3)) | np.any(np.isnan(t_cov), axis=(2,3))]=self.reward_prior['sampling_sigma']*np.eye(self.reward_posterior['theta'].shape[2])
                    # Guarantee sym
                    triu=np.zeros((self.reward_posterior['theta'].shape[2],self.reward_posterior['theta'].shape[2]),dtype=bool)
                    triu[np.triu_indices(triu.shape[0])]=True
                    t_cov[np.tile(triu.T, (self.A,self.reward_prior['M'],1,1))]=t_cov[np.tile(triu, (self.A, self.reward_prior['M'],1,1))]
                    # positive-definite
                    t_cov[np.any(np.linalg.eigvals(t_cov)<=0., axis=2)]=self.reward_prior['sampling_sigma']*np.eye(self.reward_posterior['theta'].shape[2])
                    #t_cov[np.any(t_cov != t_cov.transpose(0,1,3,2), axis=(2,3))]=self.reward_prior['sampling_sigma']*np.eye(self.reward_posterior['theta'].shape[2])
                    # Draw from transition density: multivariate t-distribution (with resampled sufficient statistics) via multivariate normal and chi-square
                    self.reward_posterior['theta'] = theta_loc + np.einsum('amdb,amb->amd', np.linalg.cholesky(t_cov), stats.norm.rvs(size=self.reward_posterior['theta'].shape))/np.sqrt(stats.chi2.rvs(nu, size=self.reward_posterior['theta'].shape)/nu)
                else:
                    # Propagate with priors over dynamics
                    # Propagated mean (resampled)
                    theta_loc=np.einsum('adb,amb->amd', self.reward_prior['A_0'], self.reward_posterior['theta'][np.arange(self.A)[:,None]*np.ones((1,self.reward_prior['M']),dtype=int),m_a])
                    # Draw from Gaussian
                    self.reward_posterior['theta']=theta_loc + np.einsum('adb,amb->amd', np.linalg.cholesky(self.reward_prior['C_0']), stats.norm.rvs(size=self.reward_posterior['theta'].shape))
            else:
                raise ValueError('Invalid reward function dynamics={}'.format(self.reward_function['dynamics']))
                
        ### Quantile type
        if self.quantile_info['type'] == 'empirical':
            ### Compute the quantiles, using the MC posterior
            ## True posterior expected_reward_samples
            # Bernoulli bandits
            if self.reward_function['type'] == 'bernoulli':
                # Double-check we have valid parameters 
                theta_loc[theta_loc<0]=0.
                theta_loc[theta_loc>1]=1.
                # Draw Bernoulli parameters, which match expected rewards
                expected_reward_samples=theta_loc[:,:,0]
        
            # Contextual Linear Gaussian bandits
            elif self.reward_function['type'] == 'linear_gaussian':
                # Expected rewards are linear combination of context and parameters
                expected_reward_samples=np.einsum('d, amd->am',self.context[:,t],theta_loc)
                
            # Logistic bandits
            elif self.reward_function['type'] == 'logistic':                
                # Expected rewards are given by the logistic function of context and parameters
                xTheta=np.einsum('d,amd->am', self.context[:,t], theta_loc)
                expected_reward_samples=np.exp(xTheta)/(1+np.exp(xTheta))
                
            else:
                raise ValueError('reward_function={} not implemented yet'.format(self.reward_function['type']))

            ### Empirically compute the quantiles, given expected reward parameters
            # First, compute expected reward
            self.rewards_expected[:,t]=np.einsum('am, am->a', posterior_weights, expected_reward_samples)
            # Order samples
            sorted_idx=[(np.arange(self.A)[:,None]*np.ones((1,self.reward_prior['M']),dtype=int)),np.argsort(expected_reward_samples, axis=1)]
            
            # Alpha computation
            if self.quantile_info['MC_alpha']=='alpha':
                alpha=np.maximum(np.minimum(self.quantile_info['alpha'][t],1),0)
            elif self.quantile_info['MC_alpha']=='alpha_plus_mcsigma':
                rewards_expected_var=np.einsum('am,am->a', posterior_weights, (self.rewards_expected[:,t][:,None]-expected_reward_samples)**2)[:,None]
                alpha=np.maximum(np.minimum(self.quantile_info['alpha'][t]+rewards_expected_var,1),0)
            elif self.quantile_info['MC_alpha']=='alpha_times_mcsigma':
                rewards_expected_var=np.einsum('am,am->a', posterior_weights, (self.rewards_expected[:,t][:,None]-expected_reward_samples)**2)[:,None]
                alpha=np.maximum(np.minimum(self.quantile_info['alpha'][t]*rewards_expected_var,1),0)
                
            # Quantile Idx
            quantile_idx=(posterior_weights[sorted_idx].cumsum(axis=1)<=(1-alpha)).sum(axis=1)
            # Fix limitations of empirical quantile approach
            if np.any(quantile_idx==expected_reward_samples.shape[1]):
                # Just last sample available
                quantile_idx[quantile_idx==expected_reward_samples.shape[1]]-=1
                
            # Quantile value
            self.arm_quantile[:,t]=expected_reward_samples[sorted_idx][np.arange(self.A),quantile_idx]
        
        elif self.quantile_info['type'] == 'sampling':
            ### Sample reward's expected returns, given updated posterior            
            # Expected reward sample data preallocation
            expected_reward_samples=np.zeros((self.A, self.quantile_info['n_samples']))
            # Sample indexes from multinomial given by posterior weights
            n_a=(np.random.rand(self.A,self.quantile_info['n_samples'],1)>posterior_weights.cumsum(axis=1)[:,None,:]).sum(axis=2)
            
            # Bernoulli bandits
            if self.reward_function['type'] == 'bernoulli':
                # Draw Bernoulli parameters, which match expected rewards
                expected_reward_samples=self.reward_posterior['theta'][np.arange(self.A)[:,None]*np.ones((1,self.quantile_info['n_samples']),dtype=int), n_a,0]
        
            # Contextual Linear Gaussian bandits
            elif self.reward_function['type'] == 'linear_gaussian':
                # Draw theta parameters
                theta_samples=self.reward_posterior['theta'][np.arange(self.A)[:,None]*np.ones((1,self.quantile_info['n_samples']),dtype=int), n_a]
                # Expected rewards are linear combination of context and parameters
                expected_reward_samples=np.einsum('d, amd->am',self.context[:,t],theta_samples)
                
            # Logistic bandits
            elif self.reward_function['type'] == 'logistic':
                # Draw theta parameters
                theta_samples=self.reward_posterior['theta'][np.arange(self.A)[:,None]*np.ones((1,self.quantile_info['n_samples']),dtype=int), n_a]
                
                # Expected rewards are given by the logistic function of context and parameters
                xTheta=np.einsum('d,amd->am', self.context[:,t], theta_samples)
                expected_reward_samples=np.exp(xTheta)/(1+np.exp(xTheta))
                
            else:
                raise ValueError('reward_function={} not implemented yet'.format(self.reward_function['type']))
                      
            ### Compute the quantile, given expected reward samples
            # First, compute expected reward
            self.rewards_expected[:,t]=expected_reward_samples.mean(axis=1)
            # Alpha computation
            if self.quantile_info['MC_alpha']=='alpha':
                alpha=np.maximum(np.minimum(self.quantile_info['alpha'][t],1),0)
            elif self.quantile_info['MC_alpha']=='alpha_plus_mcsigma':
                rewards_expected_var=np.einsum('am,am->a', posterior_weights, (self.rewards_expected[:,t][:,None]-expected_reward_samples)**2)[:,None]
                alpha=np.maximum(np.minimum(self.quantile_info['alpha'][t]+rewards_expected_var,1),0)
            elif self.quantile_info['MC_alpha']=='alpha_times_mcsigma':
                rewards_expected_var=np.einsum('am,am->a', posterior_weights, (self.rewards_expected[:,t][:,None]-expected_reward_samples)**2)[:,None]
                alpha=np.maximum(np.minimum(self.quantile_info['alpha'][t]*rewards_expected_var,1),0)

            # Quantile Idx
            quantile_idx=np.floor(self.quantile_info['n_samples']*(1-alpha)).astype(int)
            # Fix limitations of empirical quantile approach
            if np.any(quantile_idx==expected_reward_samples.shape[1]):
                # Just last sample available
                quantile_idx[quantile_idx==expected_reward_samples.shape[1]]-=1

            # Quantile value
            self.arm_quantile[:,t]=np.sort(expected_reward_samples, axis=1)[:,quantile_idx]
            
        else:
            raise ValueError('Quantile computation type={} not implemented yet'.format(self.quantile_info['type']))
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
