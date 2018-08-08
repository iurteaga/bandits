#!/usr/bin/python

# Imports: python modules
# Imports: other modules
from BanditSampling import * 
from BayesianAnalyticalPosterior import *

######## Class definition ########
class BayesianBanditSampling(BanditSampling, BayesianAnalyticalPosterior):
    """ Class for bandits with sampling policies
            - Bayesian parameter posterior updates, based on conjugacy
            - Bandits decide which arm to play by drawing arm candidates from a predictive arm posterior
            - Different arm-sampling approaches are considered
            - Different arm predictive density computation approaches are considered

    Attributes (besides inherited):
        reward_prior: the assumed prior for the multi-armed bandit's reward function
        reward_posterior: the posterior for the learned multi-armed bandit's reward function
        arm_predictive_policy: how to compute arm predictive density and sampling policy
        arm_predictive_density: predictive density of each arm
        arm_N_samples: number of candidate arm samples to draw at each time instant
    """
    
    def __init__(self, A, reward_function, reward_prior, arm_predictive_policy):
        """ Initialize the Bandit object and its attributes
        
        Args:
            A: the size of the bandit
            reward_function: the reward function of the bandit
            reward_prior: the assumed prior for the multi-armed bandit's reward function
            arm_predictive_policy: how to compute arm predictive density and sampling policy
        """
        
        # Initialize
        super().__init__(A, reward_function, reward_prior, arm_predictive_policy)
            
    def compute_arm_predictive_density(self, t):
        """ Method to compute the predictive density of each arm, based on available information at time t:
                Bayesian posterior updates

            Different alternatives on computing the arm predictive density are considered
            - Integration: Monte Carlo
                Due to analitical intractability, resort to MC
                MC over rewards:
                    - Draw parameters from the posterior
                    - Draw rewards, for each parameter sample
                    - Decide, for each drawn reward sample, which arm is the best
                    - Compute the arm predictive density, as a Monte Carlo by averaging best arm samples
                MC over Expectedrewards
                    - Draw parameters from the posterior
                    - Compute the expected reward for each parameter sample
                    - Compute the overall expected reward estimate, as a Monte Carlo by averaging over per-sample expected rewards
                    - Decide, given the MC expected reward, which arm is the best
                MC over Arms
                    - Draw parameters from the posterior
                    - Compute the expected reward for each parameter sample
                    - Decide, for each sample, which arm is the best
                    - Compute the arm predictive density, as a Monte Carlo over best arm samples
        Args:
            t: time of the execution of the bandit
        """
        ### Data preallocation
        if self.arm_predictive_policy['MC_type'] == 'MC_expectedRewards' or self.arm_predictive_policy['MC_type'] == 'MC_arms':
            rewards_expected_samples=np.zeros((self.A, self.arm_predictive_policy['M']))
        elif self.arm_predictive_policy['MC_type'] == 'MC_rewards':
            rewards_samples=np.zeros((self.A, self.arm_predictive_policy['M']))
        else:
            raise ValueError('Arm predictive density computation type={} not implemented yet'.format(self.arm_predictive_policy['MC_type']))
        
        ### Propagation of dynamic parameters for predictive density computation
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
       
        ### Sample reward's parameters, given updated hyperparameters
        # Bernoulli bandits with beta prior
        if self.reward_function['type'] == 'bernoulli' and self.reward_prior['dist'].name == 'beta':
            # Draw Bernoulli parameters
            reward_params_samples=self.reward_posterior['dist'].rvs(self.reward_posterior['alpha'], self.reward_posterior['beta'], size=(self.A,self.arm_predictive_policy['M']))
            
            if self.arm_predictive_policy['MC_type'] == 'MC_expectedRewards' or self.arm_predictive_policy['MC_type'] == 'MC_arms':
                # Compute expected rewards of sampled parameters
                rewards_expected_samples=reward_params_samples
            elif self.arm_predictive_policy['MC_type'] == 'MC_rewards':
                # Draw rewards given sampled parameters
                rewards_samples=self.reward_function['dist'].rvs(reward_params_samples)
                
        # Contextual Linear Gaussian bandits with NIG prior
        elif self.reward_function['type'] == 'linear_gaussian' and self.reward_prior['dist'] == 'NIG':        
            # For each arm
            # TODO: can we vectorize over A?
            for a in np.arange(self.A):
                if 'alpha' in self.reward_prior and 'beta' in self.reward_prior:
                    # Draw variance samples from inverse gamma
                    sigma_samples=stats.invgamma.rvs(self.reward_posterior['alpha'][a], scale=self.reward_posterior['beta'][a], size=(1,self.arm_predictive_policy['M']))
                else:
                    # Variance is known
                    sigma_samples=self.reward_function['sigma'][a]**2*np.ones((1,self.arm_predictive_policy['M']))

                # Then multivariate Gaussian parameters
                reward_params_samples=self.reward_posterior['theta'][a,:][:,None]+np.sqrt(sigma_samples)*(stats.multivariate_normal.rvs(cov=self.reward_posterior['Sigma'][a,:,:], size=self.arm_predictive_policy['M']).reshape(self.arm_predictive_policy['M'],self.d_context).T)
            
                if self.arm_predictive_policy['MC_type'] == 'MC_expectedRewards' or self.arm_predictive_policy['MC_type'] == 'MC_arms':
                    # Compute expected rewards, linearly combining context and sampled parameters
                    rewards_expected_samples[a,:]=np.dot(self.context[:,t], reward_params_samples)
                elif self.arm_predictive_policy['MC_type'] == 'MC_rewards':
                    # Draw rewards given sampled parameters and context
                    rewards_samples[a,:]=self.reward_function['dist'].rvs(loc=np.einsum('d,dm->m', self.context[:,t], reward_params_samples), scale=np.sqrt(sigma_samples))

        # TODO: Add other reward function/prior combinations
        else:
            raise ValueError('reward_function={} with reward_prior={} not implemented yet'.format(self.reward_function['type'], self.reward_prior['dist'].name))
        
        ### Compute arm predictive density
        if self.arm_predictive_policy['MC_type'] == 'MC_rewards':
            # Monte Carlo integration over reward samples
            # Mean times reward is maximum
            self.arm_predictive_density['mean'][:,t]=((rewards_samples.argmax(axis=0)[None,:]==np.arange(self.A)[:,None]).astype(int)).mean(axis=1)
            # Variance of times reward is maximum
            self.arm_predictive_density['var'][:,t]=((rewards_samples.argmax(axis=0)[None,:]==np.arange(self.A)[:,None]).astype(int)).var(axis=1)
            # Also, compute expected rewards
            self.rewards_expected[:,t]=rewards_samples.mean(axis=1)
            
        elif self.arm_predictive_policy['MC_type'] == 'MC_expectedRewards':
            # First, compute expectation over rewards
            self.rewards_expected[:,t]=rewards_expected_samples.mean(axis=1)
            
            # Then, Monte Carlo integration over expected reward
            # Arm for which expected reward is maximum
            self.arm_predictive_density['mean'][:,t]=(self.rewards_expected[:,t].argmax(axis=0)==np.arange(self.A)).astype(int)
            # No variance 
            self.arm_predictive_density['var'][:,t]=0
        elif self.arm_predictive_policy['MC_type'] == 'MC_arms':
            # Monte Carlo integration over arm samples
            # Mean times expected reward is maximum
            self.arm_predictive_density['mean'][:,t]=((rewards_expected_samples.argmax(axis=0)[None,:]==np.arange(self.A)[:,None]).astype(int)).mean(axis=1)
            # Variance of times expected reward is maximum
            self.arm_predictive_density['var'][:,t]=((rewards_expected_samples.argmax(axis=0)[None,:]==np.arange(self.A)[:,None]).astype(int)).var(axis=1)
            # Also, compute expected rewards            
            self.rewards_expected[:,t]=rewards_expected_samples.mean(axis=1)
        else:
            raise ValueError('Arm predictive density computation type={} not implemented yet'.format(self.arm_predictive_policy['MC_type']))
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
