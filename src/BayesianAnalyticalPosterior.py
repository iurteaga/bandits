#!/usr/bin/python

# Imports: python modules
import numpy as np
import copy

######## Class definition ########
class BayesianAnalyticalPosterior(object):
    """ Class for computation of bandits Bayesian posteriors analytically
        - Bayesian parameter posterior updates, based on conjugacy
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
            Following Bayesian approach with conjugate priors

        Args:
            None
        """
        # Initialize reward posterior with prior
        self.reward_posterior=copy.deepcopy(self.reward_prior)
        
    def update_reward_posterior(self, t, update_type='sequential'):
        """ Update the posterior of the reward density, based on available information at time t
            Update the posterior following a Bayesian approach with conjugate priors

        Args:
            t: time of the execution of the bandit
            update='sequential' or 'batch' update for posteriors
        """
        
        # Binomial/Bernoulli reward with beta conjugate prior
        if self.reward_function['type'] == 'bernoulli' and self.reward_prior['dist'].name == 'beta':
            if update_type=='sequential':
                a = np.where(self.actions[:,t]==1)[0][0]
                self.reward_posterior['alpha'][a]+=self.rewards[a,t]
                self.reward_posterior['beta'][a]+=(1-self.rewards[a,t])
            elif update_type=='batch':
                # Number of successes up to t (included)        
                s_t=np.nansum(self.rewards[:,:t+1], axis=1, keepdims=True)
                # Number of trials up to t (included)
                n_t=self.actions[:,:t+1].sum(axis=1, keepdims=True)
                self.reward_posterior['alpha']=self.reward_prior['alpha']+s_t
                self.reward_posterior['beta']=self.reward_prior['beta']+(n_t-s_t)
            else:
                raise ValueError('Invalid update computation type={}'.format(update_type))
        # Linear Gaussian reward with Normal Inverse Gamma conjugate prior
        elif self.reward_function['type'] == 'linear_gaussian' and self.reward_prior['dist'] == 'NIG':
            # Update parameter posterior based on observed data (if dynamic parameters, they have already been propagated)
            if update_type=='sequential':
                # Played action
                a = np.where(self.actions[:,t]==1)[0][0]
                
                # If unknown scale
                if 'alpha' in self.reward_prior and 'beta' in self.reward_prior:
                    # Update alpha
                    self.reward_posterior['alpha'][a]+=1/2
                    # Update beta
                    self.reward_posterior['beta'][a]+=1/2*np.power(self.rewards[a,t]-np.einsum('i,i->', self.context[:,t], self.reward_posterior['theta'][a,:]),2)/(1+np.einsum('i,ij,j->', self.context[:,t], self.reward_posterior['Sigma'][a,:,:], self.context[:,t]))
                
                #### Always, update based on observed action/reward
                # Update Sigma
                prev_Sigma=copy.deepcopy(self.reward_posterior['Sigma'][a,:,:])
                self.reward_posterior['Sigma'][a,:,:]=np.linalg.inv(np.linalg.inv(prev_Sigma)+np.einsum('i,j->ij',self.context[:,t], self.context[:,t]))
                # Update Theta
                self.reward_posterior['theta'][a,:]=np.einsum('ij,j->i', self.reward_posterior['Sigma'][a,:,:], np.einsum('ij,j->i',np.linalg.inv(prev_Sigma), self.reward_posterior['theta'][a,:])+self.context[:,t]*self.rewards[a,t])
            
            elif update_type=='batch':
                #### If not dynamic parameters
                if 'dynamics' not in self.reward_function:
                    for a in np.arange(self.A):
                        this_a=self.actions[a,:]==1
                        # Update Sigma
                        self.reward_posterior['Sigma'][a,:,:]=np.linalg.inv(np.linalg.inv(self.reward_prior['Sigma'][a,:,:])+np.dot(self.context[:,this_a], self.context[:,this_a].T))
                        # Update Theta
                        self.reward_posterior['theta'][a,:]=np.dot(self.reward_posterior['Sigma'][a,:,:], np.dot(np.linalg.inv(self.reward_prior['Sigma'][a,:,:]), self.reward_prior['theta'][a,:])+np.dot(self.context[:,this_a], self.rewards[a,this_a].T))

                        # Unknown scale
                        if 'alpha' in self.reward_prior and 'beta' in self.reward_prior:
                            # Update alpha
                            self.reward_posterior['alpha'][a]=self.reward_prior['alpha'][a]+this_a.size/2
                            # Update beta
                            self.reward_posterior['beta'][a]=self.reward_prior['beta'][a]+1/2*(
                                np.dot(self.rewards[a,this_a], self.rewards[a,this_a].T) +
                                np.dot(self.reward_prior['theta'][a,:].T, np.dot(np.linalg.inv(self.reward_prior['Sigma'][a,:,:]), self.reward_prior['theta'][a,:])) -
                                np.dot(self.reward_posterior['theta'][a,:].T, np.dot(np.linalg.inv(self.reward_posterior['Sigma'][a,:,:]), self.reward_posterior['theta'][a,:]))
                                )
                else:
                    raise ValueError('Cannot update in batch form for dynamic reward function {}'.format(self.reward_function['dynamics']))
            else:
                raise ValueError('Invalid update computation type={}'.format(update_type))
        # TODO: Add other reward/prior combinations
        else:
            raise ValueError('Invalid reward_function={} with reward_prior={} combination'.format(self.reward_function['type'], self.reward_prior['dist'].name))
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
