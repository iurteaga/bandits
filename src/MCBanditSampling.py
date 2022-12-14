#!/usr/bin/python

# Imports: python modules
import copy
import pdb
# Imports: other modules
from BanditSampling import * 
from MonteCarloPosterior import *
# My auxiliary functions
from my_functions import *

######## Class definition ########
class MCBanditSampling(BanditSampling, MonteCarloPosterior):
    """ Class for bandits with sampling policies
            - Monte Carlo parameter posterior updates, based on random measure approximations
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
                Monte Carlo posterior updates
                
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
        # Posterior weights
        posterior_weights=copy.deepcopy(self.reward_posterior['weights'])
        # Rewards
        if self.arm_predictive_policy['MC_type'] == 'MC_expectedRewards' or self.arm_predictive_policy['MC_type'] == 'MC_arms':
            rewards_expected_samples=np.zeros((self.A, self.arm_predictive_policy['M']))
        elif self.arm_predictive_policy['MC_type'] == 'MC_rewards':
            rewards_samples=np.zeros((self.A, self.arm_predictive_policy['M']))
        else:
            raise ValueError('Arm predictive density computation type={} not implemented yet'.format(self.arm_predictive_policy['MC_type']))
        
        
        ### Propagation of dynamic parameters for predictive density computation
        # Update particles with parameter dynamics
        # TODO: oversampling?
        if 'dynamics' in self.reward_function:
            
            # TODO: Add resampling ESS condition
            # Draw M resampling indexes from previous posterior random measure
            m_a=(np.random.rand(self.A,self.reward_prior['M'],1)>self.reward_posterior['weights'].cumsum(axis=1)[:,None,:]).sum(axis=2)
            # Weights for resampled parameters are all equal
            posterior_weights=np.ones(posterior_weights.shape)

            # Linear mixing dynamics, with known parameters
            if self.reward_function['dynamics']=='linear_mixing_known':
                # Draw from transition density: Gaussian with known parameters
                # Propagated mean (resampled)
                theta_loc=np.einsum('a...db,am...b->am...d', self.reward_function['dynamics_A'], self.reward_posterior['theta'])
                # Draw from Gaussian
                self.reward_posterior['theta']=theta_loc[np.arange(self.A)[:,None]*np.ones((1,self.reward_prior['M']),dtype=int),m_a]+ np.einsum('a...db,am...b->am...d', my_cholesky(self.reward_function['dynamics_C']), stats.norm.rvs(size=self.reward_posterior['theta'].shape))
            
            # Linear mixing dynamics, with unknown parameters
            elif self.reward_function['dynamics']=='linear_mixing_unknown':
                # Add latest particles to whole stream
                self.allTheta[...,t]=self.reward_posterior['theta']
                # Degrees of freedom for transition density
                nu=(self.reward_prior['nu_0']+t-self.reward_posterior['theta'].shape[-1]+1)
                
                # Only after observing enough data
                if t>=2*self.reward_posterior['theta'].shape[-1] and nu>0:
                    # We compute all sufficient statistics after resampling, to avoid computation of negligible (small weighted) streams
                    # Keep track of whole stream after resampling
                    self.allTheta[...,:t+1]=self.allTheta[np.arange(self.A)[:,None]*np.ones((1,self.reward_prior['M']),dtype=int),m_a,...,:t+1]
      
                    # Data products
                    ZZtop=np.einsum('am...dt,am...bt->am...db', self.allTheta[...,0:t-1], self.allTheta[...,0:t-1])
                    XZtop=np.einsum('am...dt,am...bt->am...db', self.allTheta[...,1:t], self.allTheta[...,0:t-1])
                    # Updated Lambda
                    Lambda_N=ZZtop+self.reward_prior['Lambda_0'][:,None,...]
                    # Double check invertibility
                    Lambda_N[np.any(np.isinf(Lambda_N), axis=(-2,-1)) | np.any(np.isnan(Lambda_N), axis=(-2,-1))]=self.reward_prior['sampling_sigma']*np.eye(self.reward_posterior['theta'].shape[-1])
                    Lambda_N[np.linalg.det(Lambda_N)==0]=self.reward_prior['sampling_sigma']*np.eye(self.reward_posterior['theta'].shape[-1])
                    
                    # Estimated A
                    A_hat=np.einsum('am...db,am...bc->am...dc', XZtop + self.reward_prior['A_0Lambda_0'][:,None,...], np.linalg.inv(Lambda_N))
                    # Propagated mean (already resampled)
                    theta_loc=np.einsum('am...db,am...b->am...d', A_hat, self.allTheta[...,t])
                    
                    # Auxiliary data for correlation matrix
                    diff_1=self.allTheta[...,1:t]-np.einsum('am...db,am...bt->am...dt', A_hat, self.allTheta[...,0:t-1])
                    tmp_1=np.einsum('am...dt,am...bt->am...db', diff_1, diff_1)
                    diff_2=A_hat-self.reward_prior['A_0'][:,None,...]
                    tmp_2=np.einsum('am...db,a...bc,am...ec->am...de', diff_2, self.reward_prior['Lambda_0'], diff_2)
                    UUtop=Lambda_N+np.einsum('am...d,am...b->am...db',self.allTheta[...,t],self.allTheta[...,t])
                    # Double check invertibility
                    UUtop[np.any(np.isinf(UUtop), axis=(-2,-1)) | np.any(np.isnan(UUtop), axis=(-2,-1))]=self.reward_prior['sampling_sigma']*np.eye(self.reward_posterior['theta'].shape[-1])
                    UUtop[np.linalg.det(UUtop)==0]=self.reward_prior['sampling_sigma']*np.eye(self.reward_posterior['theta'].shape[-1])
                    
                    # Estimated covariance
                    C_N=self.reward_prior['C_0'][:,None,...]+tmp_1+tmp_2
                    den=1-np.einsum('am...d,am...db,am...b->am...', self.allTheta[...,t], np.linalg.inv(UUtop), self.allTheta[...,t])
                    # Correlation matrix
                    t_cov=C_N/(nu*den)[...,None,None]
                    # Double check valid covariance matrix (TODO: better/more elegant checking scheme)
                    t_cov[np.any(np.isinf(t_cov), axis=(-2,-1)) | np.any(np.isnan(t_cov), axis=(-2,-1))]=self.reward_prior['sampling_sigma']*np.eye(self.reward_posterior['theta'].shape[-1])
                    # Guarantee sym
                    triu=np.zeros((self.reward_posterior['theta'].shape[-1],self.reward_posterior['theta'].shape[-1]),dtype=bool)
                    triu[np.triu_indices(triu.shape[0])]=True
                    t_cov[np.tile(triu.T, self.reward_posterior['theta'].shape[:-1]+(1,1))]=t_cov[np.tile(triu, self.reward_posterior['theta'].shape[:-1]+(1,1))]
                    # Draw from transition density: multivariate t-distribution (with resampled sufficient statistics) via multivariate normal and chi-square
                    self.reward_posterior['theta'] = theta_loc + np.einsum('am...db,am...b->am...d', my_cholesky(t_cov,precision=self.reward_prior['sampling_sigma'], fix_type='strict_replace'), stats.norm.rvs(size=self.reward_posterior['theta'].shape))/np.sqrt(stats.chi2.rvs(nu, size=self.reward_posterior['theta'].shape)/nu)
                else:
                    # Propagate with priors over dynamics
                    # Propagated mean (resampled)
                    theta_loc=np.einsum('a...db,am...b->am...d', self.reward_prior['A_0'], self.reward_posterior['theta'][np.arange(self.A)[:,None]*np.ones((1,self.reward_prior['M']),dtype=int),m_a])
                    # Draw from Gaussian
                    self.reward_posterior['theta']=theta_loc + np.einsum('a...db,am...b->am...d', my_cholesky(self.reward_prior['C_0']), stats.norm.rvs(size=self.reward_posterior['theta'].shape))
            else:
                raise ValueError('Invalid reward function dynamics={}'.format(self.reward_function['dynamics']))
       
        ### Sample reward's parameters, given updated approximating posterior random measure
        # Sample indexes
        m_a=(np.random.rand(self.A,self.arm_predictive_policy['M'],1)>posterior_weights.cumsum(axis=1)[:,None,:]).sum(axis=2)
        
        # Bernoulli bandits with beta prior
        if self.reward_function['type'] == 'bernoulli':
            # Double-check we have valid parameters 
            self.reward_posterior['theta'][self.reward_posterior['theta']<0]=0.
            self.reward_posterior['theta'][self.reward_posterior['theta']>1]=1.
            # Draw Bernoulli parameters
            reward_params_samples=self.reward_posterior['theta'][np.arange(self.A)[:,None]*np.ones((1,self.arm_predictive_policy['M']),dtype=int), m_a,0]
            
            if self.arm_predictive_policy['MC_type'] == 'MC_expectedRewards' or self.arm_predictive_policy['MC_type'] == 'MC_arms':
                # Compute expected rewards of sampled parameters
                rewards_expected_samples=reward_params_samples
            elif self.arm_predictive_policy['MC_type'] == 'MC_rewards':
                # Draw rewards given sampled parameters
                rewards_samples=self.reward_function['dist'].rvs(reward_params_samples)
                
        # Contextual Linear Gaussian bandits with NIG prior
        elif self.reward_function['type'] == 'linear_gaussian' and self.reward_prior['dist'] == 'NIG':
            # Draw theta parameters
            reward_params_samples=self.reward_posterior['theta'][np.arange(self.A)[:,None]*np.ones((1,self.arm_predictive_policy['M']),dtype=int), m_a]

            if 'alpha' in self.reward_prior and 'beta' in self.reward_prior:
                # Draw variance samples too
                sigma_samples=self.reward_posterior['sigma'][np.arange(self.A)[:,None]*np.ones((1,self.arm_predictive_policy['M']),dtype=int), m_a,0]
            else:
                # Variance is known
                sigma_samples=self.reward_function['sigma'][:,None]**2*np.ones((self.A,self.arm_predictive_policy['M']))

            if self.arm_predictive_policy['MC_type'] == 'MC_expectedRewards' or self.arm_predictive_policy['MC_type'] == 'MC_arms':
                # Compute expected rewards, linearly combining context and sampled parameters
                rewards_expected_samples=np.einsum('d, amd->am',self.context[:,t],reward_params_samples)
            elif self.arm_predictive_policy['MC_type'] == 'MC_rewards':
                # Draw rewards given sampled parameters and context
                rewards_samples=np.einsum('d, amd->am',self.context[:,t],reward_params_samples)+np.sqrt(sigma_samples)*self.reward_function['dist'].rvs(size=(self.A,self.arm_predictive_policy['M']))

        # Logistic bandits
        elif self.reward_function['type'] == 'logistic':
            # Draw theta parameters
            reward_params_samples=self.reward_posterior['theta'][np.arange(self.A)[:,None]*np.ones((1,self.arm_predictive_policy['M']),dtype=int), m_a]
            # Compute linear combination of context and parameters
            xTheta=np.einsum('d,amd->am', self.context[:,t], reward_params_samples)
            
            if self.arm_predictive_policy['MC_type'] == 'MC_expectedRewards' or self.arm_predictive_policy['MC_type'] == 'MC_arms':
                # Expected rewards are given by the logistic function of context and parameters
                rewards_expected_samples=np.exp(xTheta)/(1+np.exp(xTheta))
            elif self.arm_predictive_policy['MC_type'] == 'MC_rewards':
                # Draw rewards given sampled logistic function of context and parameters
                rewards_samples=stats.bernoulli.rvs(np.exp(xTheta)/(1+np.exp(xTheta)))
                
        # Softmax bandits
        elif self.reward_function['type'] == 'softmax':
            # Draw theta parameters
            reward_params_samples=self.reward_posterior['theta'][np.arange(self.A)[:,None]*np.ones((1,self.arm_predictive_policy['M']),dtype=int), m_a]
            # Compute linear combination of context and parameters per category
            xTheta=np.einsum('d,amcd->amc', self.context[:,t], reward_params_samples)
            # Compute categorical probabilities
            w=np.exp(xTheta)/(np.exp(xTheta).sum(axis=2,keepdims=True))
            
            if self.arm_predictive_policy['MC_type'] == 'MC_expectedRewards' or self.arm_predictive_policy['MC_type'] == 'MC_arms':
                # Expected rewards are given by the logistic function of context and parameters
                rewards_expected_samples=np.sum(np.arange(self.reward_function['C'])[None,None,:]*w, axis=2)
            elif self.arm_predictive_policy['MC_type'] == 'MC_rewards':
                # Draw rewards given sampled logistic function of context and parameters
                rewards_samples=(np.random.rand(self.A,self.arm_predictive_policy['M'],1)>w.cumsum(axis=2)).sum(axis=2)
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
