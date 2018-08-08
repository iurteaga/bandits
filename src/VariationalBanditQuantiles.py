#!/usr/bin/python

# Imports: python modules
# Imports: other modules
from BanditQuantiles import * 
from VariationalPosterior import *

######## Class definition ########
class VariationalBanditQuantiles(BanditQuantiles, VariationalPosterior):
    """ Class for bandits with policies based on quantiles (upper-confidence bounds)
        These bandits decide which arm to play by picking the arm with the highest expected reward quantile
            - Quantiles are computed based or mixture model variational approximation to posterior
            - Only empirical quantile computation approach is considered

    Attributes (besides inherited):
        reward_prior: the assumed prior for the multi-armed bandit's reward function
        reward_posterior: the posterior for the learned multi-armed bandit's reward function
        quantile_info: information on how to compute the quantile and which parameters to use
            quantile_info['alpha']: information on quantile parameter alpha to use (might depend on t)
            quantile_info['type']: analytical or empirical
                if quantile_info['type'] == empirical
                    quantile_info['n_samples']: number of samples for empirical quantile estimation
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
            quantile_info['type']: analytical or empirical
                if quantile_info['type'] == empirical
                    quantile_info['n_samples']: number of samples for empirical quantile estimation
        """
        
        # Initialize
        super().__init__(A, reward_function, reward_prior, quantile_info)
            
    def compute_arm_quantile(self, t):
        """ Method to compute the quantile values for each arm, based on available information at time t, which depends on posterior update type
                Variational parameter posterior updates

            Only possible alternative to computing the quantile
        Args:
            t: time of the execution of the bandit
        """
        
        ### Quantile type                
        if self.quantile_info['type'] == 'empirical':
            ### Sample reward's parameters and expected returns, given updated hyperparameters
            # Expected reward sample data preallocation
            expected_reward_samples=np.zeros((self.A, self.quantile_info['n_samples']))

            # Contextual Linear Gaussian mixture bandits with NIG prior
            if self.reward_function['type'] == 'linear_gaussian_mixture' and self.reward_prior['dist'] == 'NIG':
                # For each arm
                for a in np.arange(self.A):
                    # Data for each mixture
                    rewards_expected_per_mixture_samples=np.zeros((self.reward_prior['K'], self.quantile_info['n_samples']))
                        
                    # Compute for each mixture
                    for k in np.arange(self.reward_prior['K']):
                        # First sample variance from inverse gamma for each mixture
                        sigma_samples=stats.invgamma.rvs(self.reward_posterior['alpha'][a,k], scale=self.reward_posterior['beta'][a,k], size=(1,self.quantile_info['n_samples']))
                        # Then multivariate Gaussian parameters
                        theta_samples=self.reward_posterior['theta'][a,k,:][:,None]+np.sqrt(sigma_samples)*(stats.multivariate_normal.rvs(cov=self.reward_posterior['Sigma'][a,k,:,:], size=self.quantile_info['n_samples']).reshape(self.quantile_info['n_samples'],self.d_context).T)
                        # Compute expected reward per mixture, linearly combining context and sampled parameters
                        rewards_expected_per_mixture_samples[k,:]=np.dot(self.context[:,t], theta_samples)

                    ## How to compute (expected) rewards over mixtures
                    # Sample Z
                    if self.arm_predictive_policy['mixture_expectation'] == 'z_sampling':
                        # Draw Z from mixture proportions as determined by Dirichlet multinomial
                        k_prob=self.reward_posterior['gamma'][a]/(self.reward_posterior['gamma'][a].sum())
                        z_samples=np.random.multinomial(1,k_prob, size=self.quantile_info['n_samples']).T
                        # Compute expected rewards, for each of the picked mixture
                        # Note: transposed used due to python indexing
                        rewards_expected_samples[a,:]=rewards_expected_per_mixture_samples.T[z_samples.T==1]
                    
                    # Sample pi
                    elif self.arm_predictive_policy['mixture_expectation'] == 'pi_sampling':
                        # Draw mixture proportions as determined by Dirichlet multinomial
                        pi_samples=stats.dirichlet.rvs(self.reward_posterior['gamma'][a], size=self.quantile_info['n_samples']).T
                        # Compute expected rewards, by averaging over sampled mixture proportions
                        rewards_expected_samples[a,:]=np.einsum('km,km->m', pi_samples, rewards_expected_per_mixture_samples)
                            
                    # Expected pi
                    elif self.arm_predictive_policy['mixture_expectation'] == 'pi_expected':
                        # Computed expected mixture proportions as determined by Dirichlet multinomial
                        pi=self.reward_posterior['gamma'][a]/(self.reward_posterior['gamma'][a].sum())
                        # Compute expected rewards, by averaging over expected mixture proportions
                        rewards_expected_samples[a,:]=np.einsum('k,km->m', pi, rewards_expected_per_mixture_samples)
                        
                    else:
                        raise ValueError('Arm predictive mixture expectation computation type={} not implemented yet'.format(self.arm_predictive_policy['mixture_expectation']))

            # TODO: Add other reward function/prior combinations
            else:
                raise ValueError('reward_function={} with reward_prior={} not implemented yet'.format(self.reward_function['dist'].nameself.reward_prior['dist'].name))
            
            ### Empirically compute the quantiles, given expected reward parameters
            # TODO: doublecheck
            self.arm_quantile[:,t]=np.sort(expected_reward_samples, axis=1)[:,np.floor(self.quantile_info['n_samples']*(1-self.quantile_info['alpha'][t])).astype(int)]
            
            # Also, compute expected reward
            self.rewards_expected[:,t]=expected_reward_samples.mean(axis=1)
        else:
            raise ValueError('Quantile computation type={} not implemented yet'.format(self.quantile_info['type']))
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
