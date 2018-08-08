#!/usr/bin/python

# Imports: python modules
# Imports: other modules
from BanditQuantiles import * 
from BayesianAnalyticalPosterior import *

######## Class definition ########
class BayesianBanditQuantiles(BanditQuantiles, BayesianAnalyticalPosterior):
    """ Class for bandits with policies based on quantiles (upper-confidence bounds)
        These bandits decide which arm to play by picking the arm with the highest expected reward quantile
            - Quantiles are computed for Bayesian parameter posteriors, based on conjugacy
            - Analytical and empirical quantile computation approaches are considered

    Attributes (besides inherited):
        reward_prior: the assumed prior for the multi-armed bandit's reward function
        reward_posterior: the posterior for the learned multi-armed bandit's reward function
        quantile_info: information on how to compute the quantile and which parameters to use
            quantile_info['alpha']: information on quantile parameter alpha to use (might depend on t)
            quantile_info['type']: analytical or sampling-based
                if quantile_info['type'] == empirical
                    quantile_info['n_samples']: number of samples for sampling based quantile estimation
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
            quantile_info['type']: analytical or sampling-based
                if quantile_info['type'] == sampling
                    quantile_info['n_samples']: number of samples for sampling-based quantile estimation
        """
        
        # Initialize
        super().__init__(A, reward_function, reward_prior, quantile_info)
            
    def compute_arm_quantile(self, t):
        """ Method to compute the quantile values for each arm, based on available information at time t, which depends on posterior update type
                Bayesian posterior updates

            Different alternatives on computing the quantiles are considered
                quantile_info['type']='analytical': use the exact analytical quantile (inverse CDF) function of the expected rewards 
                quantile_info['type']='sampling': estimate the quantile (inverse CDF) function by sampling expected reward samples
        Args:
            t: time of the execution of the bandit
        """
        ### Propagation of dynamic parameters for arm quantile computation
        # Analytically update dynamic sufficient statistics
        if 'dynamics' in self.reward_function:
            # Linear mixing dynamics, with known parameters
            if self.reward_function['dynamics']=='linear_mixing_known':
                self.reward_posterior['theta']=np.einsum('adb, ab->ad', self.reward_function['dynamics_A'], self.reward_posterior['theta'])
                self.reward_posterior['Sigma']=np.einsum('abc,acd,ade->abe', self.reward_function['dynamics_A'], self.reward_posterior['Sigma'],self.reward_function['dynamics_A'])
                if 'alpha' in self.reward_prior and 'beta' in self.reward_prior:
                    # Unknown scale
                    self.reward_posterior['Sigma']+=self.reward_function['dynamics_C']*self.reward_posterior['alpha']/self.reward_posterior['beta']
                else:
                    # Known scale
                    self.reward_posterior['Sigma']+=self.reward_function['dynamics_C']/self.reward_function['sigma']**2
            else:
                raise ValueError('Invalid reward function dynamics={}'.format(self.reward_function['dynamics']))

        ### Quantile type
        if self.quantile_info['type'] == 'analytical':
            ### Compute the quantiles, using the analytical quantile function
            # Bernoulli bandits with beta prior
            if self.reward_function['type'] == 'bernoulli' and self.reward_prior['dist'].name == 'beta':
                # Quantile of updated beta
                self.arm_quantile[:,[t]]=self.reward_posterior['dist'].ppf(1-self.quantile_info['alpha'][t], self.reward_posterior['alpha'], self.reward_posterior['beta'])
            
                # Also, compute expected reward, which matches expected value of theta, as given by its Beta prior
                self.rewards_expected[:,t]=(self.reward_posterior['alpha']/(self.reward_posterior['alpha']+self.reward_posterior['beta']))[:,0]
                
            # Contextual Linear Gaussian bandits with NIG prior
            elif self.reward_function['type'] == 'linear_gaussian' and self.reward_prior['dist'] == 'NIG':                
                # Updated sufficient statistics
                mu_loc=np.einsum('d, ad->a',self.context[:,t], self.reward_posterior['theta'])[:,None]
                if 'alpha' in self.reward_prior and 'beta' in self.reward_prior:
                    # Scale unknown
                    mu_var=self.reward_posterior['beta']/self.reward_posterior['alpha']*np.einsum('d,adb,b->a', self.context[:,t], self.reward_posterior['Sigma'], self.context[:,t])[:,None]
                    # Quantile of updated t
                    self.arm_quantile[:,[t]]=stats.t.ppf(1-self.quantile_info['alpha'][t], 2*self.reward_posterior['alpha'], loc=mu_loc, scale=np.sqrt(mu_var))
                else:
                    # Scale is known
                    mu_var=(self.reward_function['sigma']**2*np.einsum('d,adb,b->a', self.context[:,t], self.reward_posterior['Sigma'], self.context[:,t]))[:,None]
                    # Quantile of updated t
                    self.arm_quantile[:,[t]]=stats.norm.ppf(1-self.quantile_info['alpha'][t], loc=mu_loc, scale=np.sqrt(mu_var))
                
                # Also, compute expected reward, which matches expected mu
                self.rewards_expected[:,t]=mu_loc[:,0]  
            else:
                raise ValueError('reward_function={} with reward_prior={} not implemented yet'.format(self.reward_function['type'], self.reward_prior['dist'].name))
                
        elif self.quantile_info['type'] == 'sampling':
            ### Sample reward's parameters and expected returns, given updated hyperparameters
            # Expected reward sample data preallocation
            expected_reward_samples=np.zeros((self.A, self.quantile_info['n_samples']))
            
            # Bernoulli bandits with beta prior
            if self.reward_function['type'] == 'bernoulli' and self.reward_prior['dist'].name == 'beta':
                # Draw Bernoulli parameters, which match expected rewards
                expected_reward_samples=self.reward_posterior['dist'].rvs(self.reward_posterior['alpha'], self.reward_posterior['beta'], size=(self.A,self.quantile_info['n_samples']))
                        
            # Contextual Linear Gaussian bandits with NIG prior
            elif self.reward_function['type'] == 'linear_gaussian' and self.reward_prior['dist'] == 'NIG':                        
                # Updated sufficient statistics
                mu_loc=np.einsum('d, ad->a',self.context[:,t], self.reward_posterior['theta'])[:,None]
                if 'alpha' in self.reward_prior and 'beta' in self.reward_prior:
                    # Scale is unknown
                    mu_var=self.reward_posterior['beta']/self.reward_posterior['alpha']*np.einsum('d,adb,b->a', self.context[:,t], self.reward_posterior['Sigma'], self.context[:,t])[:,None]
                    # Draw expected rewards from updated Student-t
                    expected_reward_samples=stats.t.rvs(2*self.reward_posterior['alpha'], loc=mu_loc, scale=np.sqrt(mu_var), size=(self.A,self.quantile_info['n_samples']))
                else:
                    # Scale is known
                    mu_var=(self.reward_function['sigma']**2*np.einsum('d,adb,b->a', self.context[:,t], self.reward_posterior['Sigma'], self.context[:,t]))[:,None]
                    # Draw expected rewards from updated Gaussian
                    expected_reward_samples=stats.norm.rvs(loc=mu_loc, scale=np.sqrt(mu_var), size=(self.A,self.quantile_info['n_samples']))
                    
            else:
                raise ValueError('reward_function={} with reward_prior={} not implemented yet'.format(self.reward_function['type'], self.reward_prior['dist'].name))
            
            ### Compute the quantile, given expected reward samples
            # Quantile Idx
            quantile_idx=np.floor(self.quantile_info['n_samples']*(1-self.quantile_info['alpha'][t])).astype(int)
            # Fix limitations of empirical quantile approach
            if np.any(quantile_idx==expected_reward_samples.shape[1]):
                # Just last sample available
                quantile_idx[quantile_idx==expected_reward_samples.shape[1]]-=1
            
            # Quantile value
            self.arm_quantile[:,t]=np.sort(expected_reward_samples, axis=1)[:,quantile_idx]
            
            # Also, compute expected reward
            self.rewards_expected[:,t]=expected_reward_samples.mean(axis=1)
            
        else:
            raise ValueError('Quantile computation type={} not implemented yet'.format(self.quantile_info['type']))
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
