#!/usr/bin/python

# Imports: python modules
# Imports: other modules
from BanditSampling import * 

######## Class definition ########
class BayesianBanditSampling(BanditSampling):
    """ Class for bandits with
        - Bayesian parameter posterior updates, based on conjugacy
        - Arm sampling based policies

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
            
    def update_reward_posterior(self, t, update_type='sequential'):
        """ Update the posterior of the reward density, based on available information at time t
            Update the posterior following a Bayesian approach with conjugate priors

        Args:
            t: time of the execution of the bandit
            update='sequential' or 'batch' update for posteriors
        """
        
        # Binomial/Bernoulli reward with beta conjugate prior
        if self.reward_function['dist'].name == 'bernoulli' and self.reward_prior['dist'].name == 'beta':
            if update_type=='sequential':
                self.reward_posterior['alpha']+=self.rewards[:,t]
                self.reward_posterior['beta']+=1-self.rewards[:,t]
            elif update_type=='batch':
                # Number of successes up to t (included)        
                s_t=self.rewards[:,:t+1].sum(axis=1, keepdims=True)
                # Number of trials up to t (included)
                n_t=self.actions[:,:t+1].sum(axis=1, keepdims=True)
                self.reward_posterior['alpha']=self.reward_prior['alpha']+s_t
                self.reward_posterior['beta']=self.reward_prior['beta']+(n_t-s_t)
            else:
                raise ValueError('Invalid update computation type={}'.format(update_type))
        # Linear Gaussian reward with Normal Inverse Gamma conjugate prior
        elif self.reward_function['type'] == 'linear_gaussian' and self.reward_prior['dist'] == 'NIG':
            if update_type=='sequential':
                # Played action
                a = np.where(self.actions[:,t]==1)[0][0]
                # Update alpha
                self.reward_posterior['alpha'][a]+=1/2
                # Update beta
                self.reward_posterior['beta'][a]+=1/2*np.power(self.rewards[a,t]-np.einsum('i,i->', self.context[:,t], self.reward_posterior['theta'][a,:]),2)/(1+np.einsum('i,ij,j->', self.context[:,t], self.reward_posterior['Sigma'][a,:,:], self.context[:,t]))
                # Update Sigma
                prev_Sigma=copy.deepcopy(self.reward_posterior['Sigma'][a,:,:])
                self.reward_posterior['Sigma'][a,:,:]=np.linalg.inv(np.linalg.inv(prev_Sigma)+np.einsum('i,j->ij',self.context[:,t], self.context[:,t]))
                # Update Theta
                self.reward_posterior['theta'][a,:]=np.einsum('ij,j->i', self.reward_posterior['Sigma'][a,:,:], np.einsum('ij,j->i',np.linalg.inv(prev_Sigma), self.reward_posterior['theta'][a,:])+self.context[:,t]*self.rewards[a,t])
            elif update_type=='batch':
                for a in np.arange(self.A):
                    this_a=self.actions[a,:]==1
                    # Update Sigma
                    self.reward_posterior['Sigma'][a,:,:]=np.linalg.inv(np.linalg.inv(self.reward_prior['Sigma'][a,:,:])+np.dot(self.context[:,this_a], self.context[:,this_a].T))
                    # Update Theta
                    self.reward_posterior['theta'][a,:]=np.dot(self.reward_posterior['Sigma'][a,:,:], np.dot(np.linalg.inv(self.reward_prior['Sigma'][a,:,:]), self.reward_prior['theta'][a,:])+np.dot(self.context[:,this_a], self.rewards[a,this_a].T))
                    # Update alpha
                    self.reward_posterior['alpha'][a]=self.reward_prior['alpha'][a]+this_a.size/2
                    # Update beta
                    self.reward_posterior['beta'][a]=self.reward_prior['beta'][a]+1/2*(
                        np.dot(self.rewards[a,this_a], self.rewards[a,this_a].T) +
                        np.dot(self.reward_prior['theta'][a,:].T, np.dot(np.linalg.inv(self.reward_prior['Sigma'][a,:,:]), self.reward_prior['theta'][a,:])) -
                        np.dot(self.reward_posterior['theta'][a,:].T, np.dot(np.linalg.inv(self.reward_posterior['Sigma'][a,:,:]), self.reward_posterior['theta'][a,:]))
                        )
            else:
                raise ValueError('Invalid update computation type={}'.format(update_type))
        # TODO: Add other reward/prior combinations
        else:
            raise ValueError('Invalid reward_function={} with reward_prior={} combination'.format(self.reward_function['dist'].name, self.reward_prior['dist'].name))
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
