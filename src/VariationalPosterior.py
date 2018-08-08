#!/usr/bin/python

# Imports: python modules
import numpy as np
import copy
import scipy.special as special
import matplotlib.pyplot as plt
import pdb

######## Class definition ########
class VariationalPosterior(object):
    """ Class for computation of bandit posteriors via Variational inference
        - Mixture model approximation to reward function
        - Parameter posterior updates, based on variational inference
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
            Following mixture model variational approximation to posterior

        Args:
            None
        """
        # Initialize reward posterior with prior
        self.reward_posterior=copy.deepcopy(self.reward_prior)
                   
    def update_reward_posterior(self, t):
        """ Update the posterior of the reward density, based on available information at time t
            Update the posterior following a Variational approximation approach

        Args:
            t: time of the execution of the bandit
        """

        # Linear Gaussian Mixture with Normal Inverse Gamma conjugate prior
        if self.reward_prior['type'] == 'linear_gaussian_mixture' and self.reward_prior['dist'] == 'NIG':
            # Variational update
            # Requires responsibilities of mixtures per arm
            self.r=np.zeros((self.A, self.reward_prior['K'], self.rewards.shape[1]))
            # Lower bound
            lower_bound=np.zeros(self.reward_prior['variational_max_iter']+1)
            lower_bound[0]=np.finfo(float).min
            # First iteration
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
                    
        # TODO: Add other reward/prior combinations
        else:
            raise ValueError('Invalid reward_function={} with reward_prior={} combination'.format(self.reward_prior['type'], self.reward_prior['dist']))
            
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

    def update_reward_posterior_variational_resp(self, t):
    
        # Variational responsibilities for linear Gaussian Mixture with Normal Inverse Gamma conjugate prior
        if self.reward_prior['type'] == 'linear_gaussian_mixture' and self.reward_prior['dist'] == 'NIG':
            # Indicators for picked arms
            tIdx, aIdx=np.where(self.actions.T)
            # Compute rho
            rho=( -0.5*(np.log(self.reward_posterior['beta'])-special.digamma(self.reward_posterior['alpha']))[:,:,None]
                - 0.5*(np.einsum('it,akij,jt->akt', self.context[:,tIdx], self.reward_posterior['Sigma'], self.context[:,tIdx])+np.power(self.rewards[:,tIdx].sum(axis=0)-np.einsum('it,aki->akt', self.context[:,tIdx], self.reward_posterior['theta']),2)*self.reward_posterior['alpha'][:,:,None]/self.reward_posterior['beta'][:,:,None])
                + (special.digamma(self.reward_posterior['gamma'])-special.digamma(self.reward_posterior['gamma'].sum(axis=1,keepdims=True)))[:,:,None]
                )
            # Exponentiate (and try to avoid numerical errors)
            r=np.exp(rho-rho.max(axis=1, keepdims=True))
            # And normalize
            self.r[:,:,tIdx]=r/(r.sum(axis=1, keepdims=True))
        
    def update_reward_posterior_variational_params(self):
    
        # Variational parameters for linear Gaussian Mixture with Normal Inverse Gamma conjugate prior
        if self.reward_prior['type'] == 'linear_gaussian_mixture' and self.reward_prior['dist'] == 'NIG':
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
                    self.reward_posterior['theta'][a,k,:]=np.dot(self.reward_posterior['Sigma'][a,k,:,:], np.dot(np.linalg.inv(self.reward_prior['Sigma'][a,k,:,:]), self.reward_prior['theta'][a,k,:])+np.dot(np.dot(self.context[:,this_a], R_ak), self.rewards[a,this_a].T))
                    # Update and append alpha
                    self.reward_posterior['alpha'][a,k]=self.reward_prior['alpha'][a,k]+(self.r[a,k,this_a].sum())/2
                    # Update and append beta
                    self.reward_posterior['beta'][a,k]=self.reward_prior['beta'][a,k]+1/2*(
                        np.dot(np.dot(self.rewards[a,this_a], R_ak), self.rewards[a,this_a].T) +
                        np.dot(self.reward_prior['theta'][a,k,:].T, np.dot(np.linalg.inv(self.reward_prior['Sigma'][a,k,:,:]), self.reward_prior['theta'][a,k,:])) -
                        np.dot(self.reward_posterior['theta'][a,k,:].T, np.dot(np.linalg.inv(self.reward_posterior['Sigma'][a,k,:,:]), self.reward_posterior['theta'][a,k,:]))
                        )
        
    def update_reward_posterior_variational_lowerbound(self,t):
        # Indicators for picked arms
        tIdx, aIdx=np.where(self.actions.T)
        atIdx=np.arange(aIdx.size)
        # E{ln f(Y|Z,\theta,\sigma)}
        tmp=( (np.log(2*np.pi)+np.log(self.reward_posterior['beta'])-special.digamma(self.reward_posterior['alpha']))[:,:,None] 
                + np.einsum('it,akij,jt->akt', self.context[:,tIdx], self.reward_posterior['Sigma'], self.context[:,tIdx])
                + np.power(self.rewards[:,tIdx].sum(axis=0)-np.einsum('it,aki->akt', self.context[:,tIdx], self.reward_posterior['theta']),2) * self.reward_posterior['alpha'][:,:,None]/self.reward_posterior['beta'][:,:,None] )
        # Sum over mixture and time indicators
        E_ln_fY=-0.5*(self.r[aIdx,:,tIdx]*tmp[aIdx,:,atIdx]).sum(axis=(1,0))
        
        # E{ln p(Z|\pi}
        tmp=self.r[:,:,tIdx]*(special.digamma(self.reward_posterior['gamma'])-special.digamma(self.reward_posterior['gamma'].sum(axis=1, keepdims=True)))[:,:,None]
        # Sum over mixture and arm-time indicators
        E_ln_pZ=tmp[aIdx,:,atIdx].sum(axis=(1,0))
    
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
        tmp=self.r[:,:,tIdx]*np.log(self.r[:,:,tIdx])
        # Sum over mixture and arm-time indicators
        E_ln_qZ=tmp[aIdx,:,atIdx].sum(axis=(1,0))
        
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
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
