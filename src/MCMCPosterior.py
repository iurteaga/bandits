#!/usr/bin/python

# Imports: python modules
import numpy as np
import copy
import scipy.special as special
import scipy.stats as stats
import time
import matplotlib.pyplot as plt
import pdb

######## Class definition ########
class MCMCPosterior(object):
    """ Class for computation of bandit posteriors via MCMC inference
        - Mixture model approximation to reward function
        - Parameter posterior updates, based on MCMC inference
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
            Following mixture model approximation to posterior

        Args:
            None
        """       
        if self.reward_prior['K'] != 'nonparametric':
            # Initialize reward posterior with prior
            self.reward_posterior=copy.deepcopy(self.reward_prior)
            if self.reward_prior['K'].size==1:
                self.reward_posterior['K']=np.ones(self.A, dtype=int)*self.reward_prior['K']
                
        elif self.reward_prior['K'] == 'nonparametric':
            # Initialize posterior
            self.reward_posterior={'alpha': copy.deepcopy(self.reward_prior['alpha'][:,None]), 'beta': copy.deepcopy(self.reward_prior['beta'][:,None]), 'theta': copy.deepcopy(self.reward_prior['theta'][:,None,:]),'Sigma': copy.deepcopy(self.reward_prior['Sigma'][:,None,::]), 'K':np.zeros(self.A, dtype=int)}
        else:
            raise ValueError('Invalid reward_prior K={}'.format(self.reward_prior['K']))

        # Assignments to mixtures matrix
        self.reward_posterior['Z']=np.nan*np.ones((self.A, self.rewards.shape[1]))
                 
    def update_reward_posterior(self, t):
        """ Update the posterior of the reward density, based on available information at time t
            Update the posterior following a MCMC approximation approach

        Args:
            t: time of the execution of the bandit
        """
        
        # Linear Gaussian Mixture with Normal Inverse Gamma conjugate prior
        if self.reward_prior['type'] == 'linear_gaussian_mixture' and self.reward_prior['dist'] == 'NIG':
            t_init=time.process_time()
            # Time-indexes for this arm
            # Played action
            a = np.where(self.actions[:,t]==1)[0][0]
            t_a=self.actions[a,:]==1
            # Relevant data for this arm
            y_a=self.rewards[a,t_a]
            x_a=self.context[:,t_a]
            
            # If first observation
            if t_a.sum()==1:
                n=0
                # Fixed number of mixtures 
                if self.reward_prior['K'] != 'nonparametric':
                    # Likelihood of x under each mixture component
                    xlik_k=self.compute_ylikelihood_per_mixture(a, x_a[:,n], y_a[n])
                    # Assignment will be the mixture with max likelihood
                    k_new=np.argmax(xlik_k)
                    self.reward_posterior['Z'][a,t_a]=k_new

                # Nonparametric number of mixtures
                elif self.reward_prior['K'] == 'nonparametric':
                    # Assign to first mixture
                    k_new=0
                    self.reward_posterior['Z'][a,t_a]=k_new
                    # Update mixture count
                    self.reward_posterior['K'][a]+=1
                    
                else:
                    raise ValueError('Invalid reward_prior K={}'.format(self.reward_prior['K']))

                # Update posterior with this new assignment
                self.update_reward_posterior_params('add', a, k_new, x_a[:,n], y_a[n])
                    
            else:
                # Previous mixture assignments for this arm
                z_a=self.reward_posterior['Z'][a,t_a]

                # Suff statistics
                N_ak=np.zeros(self.reward_posterior['K'][a])
                k, k_count=np.unique(z_a[~np.isnan(z_a)], return_counts=True)
                N_ak[k.astype(int)]=k_count

                # Loglikelihood variables
                XcondZ_loglik=np.nan*np.ones(self.reward_prior['gibbs_max_iter']+1)
                XcondZ_loglik[0]=np.finfo(float).min
                Z_loglik=np.nan*np.ones(self.reward_prior['gibbs_max_iter']+1)
                Z_loglik[0]=np.finfo(float).min
            
                # For new observation, 
                if self.reward_prior['K'] != 'nonparametric':
                    # Likelihood of last observation for each mixture component
                    xlik_k=self.compute_ylikelihood_per_mixture(a, x_a[:,-1], y_a[-1])
                    # Probability of mixtures
                    p_k=(self.reward_prior['gamma'][a]+N_ak[:self.reward_posterior['K'][a]])/(self.reward_prior['gamma'][a].sum()+N_ak[:self.reward_posterior['K'][a]].sum())*xlik_k
                    
                elif self.reward_prior['K'] == 'nonparametric':
                    # Likelihood of last observation for each mixture component
                    xlik_k=self.compute_ylikelihood_per_mixture(a, x_a[:,-1], y_a[-1])
                    # Likelihood of last observation for unseen mixture component
                    xlik_new_k=self.compute_ylikelihood_for_new_mixture(a, x_a[:,-1], y_a[-1])
                    # Probability of mixtures (seen and new)
                    p_k=np.concatenate(((N_ak[:self.reward_posterior['K'][a]]-self.reward_prior['d'][a])*xlik_k/(N_ak[:self.reward_posterior['K'][a]].sum()+self.reward_prior['gamma'][a]), (self.reward_prior['gamma'][a]+self.reward_posterior['K'][a]*self.reward_prior['d'][a])*xlik_new_k/(N_ak[:self.reward_posterior['K'][a]].sum()+self.reward_prior['gamma'][a])), axis=0)
                else:
                    raise ValueError('Invalid reward_prior K={}'.format(self.reward_prior['K']))
                    
                # Normalize mixture probabilities
                p_k=(p_k/p_k.max())/((p_k/p_k.max()).sum())
                # And draw mixture assignment
                k_new=np.where(stats.multinomial.rvs(1,p_k))[0][0]

                # If new (nonparametric) mixture
                if self.reward_prior['K'] == 'nonparametric' and k_new==self.reward_posterior['K'][a]:
                    # Allocate space if needed
                    if k_new==N_ak.size:
                        # Increase (double) posteriors and mixture assignment sizes
                        N_ak=np.concatenate((N_ak, np.zeros(N_ak.size)), axis=0)
                        self.reward_posterior['alpha']=np.concatenate((self.reward_posterior['alpha'], self.reward_prior['alpha'][:,None]*np.ones((1,self.reward_posterior['K'][a]))), axis=1)
                        self.reward_posterior['beta']=np.concatenate((self.reward_posterior['beta'], self.reward_prior['beta'][:,None]*np.ones((1,self.reward_posterior['K'][a]))), axis=1)
                        self.reward_posterior['theta']=np.concatenate((self.reward_posterior['theta'], self.reward_prior['theta'][:,None,:]*np.ones((1,self.reward_posterior['K'][a],1))), axis=1)
                        self.reward_posterior['Sigma']=np.concatenate((self.reward_posterior['Sigma'], self.reward_prior['Sigma'][:,None,:,:]*np.ones((1,self.reward_posterior['K'][a],1,1))), axis=1)
                    # Increase mixture number count
                    self.reward_posterior['K'][a]+=1

                # Update with new observation assignment
                z_a[-1]=k_new
                N_ak[k_new]+=1
                self.update_reward_posterior_params('add', a, k_new, x_a[:,-1], y_a[-1])

                # Ready to start Gibbs
                n_iter=1
                (XcondZ_loglik[n_iter], Z_loglik[n_iter])=self.compute_loglikelihood(a, z_a, N_ak, x_a, y_a)
                print('t={}, n_iter={}, {} observations for arm {} with loglikelihood={}'.format(t, n_iter, t_a.sum(), a, XcondZ_loglik[n_iter]+Z_loglik[n_iter]))
            
                # Iterate while not converged or not max iterations
                while (n_iter < self.reward_prior['gibbs_max_iter'] and abs((XcondZ_loglik[n_iter]+Z_loglik[n_iter]) - (XcondZ_loglik[n_iter-1]+Z_loglik[n_iter-1])) >= (self.reward_prior['gibbs_loglik_eps']*abs((XcondZ_loglik[n_iter-1]+Z_loglik[n_iter-1])))):
                    n_iter+=1
                    # For all rewards of this arm
                    for n in np.random.permutation(t_a.sum()):
                        # Mixture assignment for this observation
                        k_old=int(z_a[n])
                        # Update posterior: "unsee" datapoint
                        N_ak[k_old]-=1
                        self.update_reward_posterior_params('del', a, k_old, x_a[:,n], y_a[n])
                        
                        # Housekeeping for nonparametric case
                        if self.reward_prior['K'] == 'nonparametric' and N_ak[k_old]==0:
                            # TODO: We are not emptying memory, just initializing with prior
                            # Empty observation counts
                            N_ak[k_old:self.reward_posterior['K'][a]-1]=N_ak[k_old+1:self.reward_posterior['K'][a]]; N_ak[self.reward_posterior['K'][a]-1:]=0
                            # Empty mixture posteriors
                            self.reward_posterior['alpha'][a,k_old:self.reward_posterior['K'][a]-1]=self.reward_posterior['alpha'][a,k_old+1:self.reward_posterior['K'][a]]; self.reward_posterior['alpha'][a,self.reward_posterior['K'][a]-1:]=self.reward_prior['alpha'][a]
                            self.reward_posterior['beta'][a,k_old:self.reward_posterior['K'][a]-1]=self.reward_posterior['beta'][a,k_old+1:self.reward_posterior['K'][a]]; self.reward_posterior['beta'][a,self.reward_posterior['K'][a]-1:]=self.reward_prior['beta'][a]
                            self.reward_posterior['theta'][a,k_old:self.reward_posterior['K'][a]-1,:]=self.reward_posterior['theta'][a,k_old+1:self.reward_posterior['K'][a],:]; self.reward_posterior['theta'][a,self.reward_posterior['K'][a]-1:,:]=self.reward_prior['theta'][a]
                            self.reward_posterior['Sigma'][a,k_old:self.reward_posterior['K'][a]-1,:,:]=self.reward_posterior['Sigma'][a,k_old+1:self.reward_posterior['K'][a],:,:]; self.reward_posterior['Sigma'][a,self.reward_posterior['K'][a]-1:,:,:]=self.reward_prior['Sigma'][a]
                            # Reduce mixture assignments impacted by emptied one
                            z_a[z_a>k_old]-=1
                            # Reduce number of mixtures
                            self.reward_posterior['K'][a]-=1
                        
                        # Probabilities per mixture
                        if self.reward_prior['K'] != 'nonparametric':
                            # Likelihood of datapoint for each mixture component
                            xlik_k=self.compute_ylikelihood_per_mixture(a, x_a[:,n], y_a[n])
                            # Probability of mixtures
                            p_k=(self.reward_prior['gamma'][a]+N_ak[:self.reward_posterior['K'][a]])/(self.reward_prior['gamma'][a].sum()+N_ak[:self.reward_posterior['K'][a]].sum())*xlik_k
                            
                        elif self.reward_prior['K'] == 'nonparametric':
                            # Likelihood of datapoint for each mixture component
                            xlik_k=self.compute_ylikelihood_per_mixture(a, x_a[:,n], y_a[n])
                            # Likelihood of datapoint for unseen mixture component
                            xlik_new_k=self.compute_ylikelihood_for_new_mixture(a, x_a[:,n], y_a[n])
                            # Probability of mixtures (seen and new)
                            p_k=np.concatenate(((N_ak[:self.reward_posterior['K'][a]]-self.reward_prior['d'][a])*xlik_k/(N_ak[:self.reward_posterior['K'][a]].sum()+self.reward_prior['gamma'][a]), (self.reward_prior['gamma'][a]+self.reward_posterior['K'][a]*self.reward_prior['d'][a])*xlik_new_k/(N_ak[:self.reward_posterior['K'][a]].sum()+self.reward_prior['gamma'][a])), axis=0)
                        else:
                            raise ValueError('Invalid reward_prior K={}'.format(self.reward_prior['K']))
                            
                        # Normalize probabilities
                        p_k=(p_k/p_k.max())/((p_k/p_k.max()).sum())
                        # and draw mixture assignment
                        k_new=np.where(stats.multinomial.rvs(1,p_k))[0][0]

                        # If new (nonparametric) mixture
                        if self.reward_prior['K'] == 'nonparametric' and k_new==self.reward_posterior['K'][a]:
                            # Allocate space if needed
                            if k_new==N_ak.size:
                                # Increase (double) posteriors and mixture assignment sizes
                                N_ak=np.concatenate((N_ak, np.zeros(N_ak.size)), axis=0)
                                self.reward_posterior['alpha']=np.concatenate((self.reward_posterior['alpha'], self.reward_prior['alpha'][:,None]*np.ones((1,self.reward_posterior['K'][a]))), axis=1)
                                self.reward_posterior['beta']=np.concatenate((self.reward_posterior['beta'], self.reward_prior['beta'][:,None]*np.ones((1,self.reward_posterior['K'][a]))), axis=1)
                                self.reward_posterior['theta']=np.concatenate((self.reward_posterior['theta'], self.reward_prior['theta'][:,None,:]*np.ones((1,self.reward_posterior['K'][a],1))), axis=1)
                                self.reward_posterior['Sigma']=np.concatenate((self.reward_posterior['Sigma'], self.reward_prior['Sigma'][:,None,:,:]*np.ones((1,self.reward_posterior['K'][a],1,1))), axis=1)
                            # Increase mixture number count
                            self.reward_posterior['K'][a]+=1

                        # Update assignment and posterior: "See" data point
                        z_a[n]=k_new
                        N_ak[k_new]+=1
                        self.update_reward_posterior_params('add', a, k_new, x_a[:,n], y_a[n])

                    # Compute loglikelihood
                    (XcondZ_loglik[n_iter], Z_loglik[n_iter])=self.compute_loglikelihood(a, z_a, N_ak, x_a, y_a)
                    print('t={}, n_iter={}, {} observations for arm {} with loglikelihood={}'.format(t, n_iter, t_a.sum(), a, XcondZ_loglik[n_iter]+Z_loglik[n_iter]))
                    
                # Final assignments
                self.reward_posterior['Z'][a,t_a]=z_a
                
                # Plotting Gibbs loglikelihood evolution
                if self.reward_prior['gibbs_plot_save'] is not None: 
                    # Plotting    
                    plt.figure()
                    plt.xlabel('n_iter')
                    plt.plot(np.arange(len(Z_loglik)),Z_loglik, 'b', label='log p(Z)')
                    plt.plot(np.arange(len(XcondZ_loglik)),XcondZ_loglik, 'g', label='log p(X|Z)')
                    plt.plot(np.arange(len(XcondZ_loglik)),XcondZ_loglik+Z_loglik, 'r', label='log p(X,Z)')
                    plt.xlim([1,1+(~np.isnan(XcondZ_loglik)).sum()])
                    plt.xlabel('n_iter')
                    plt.ylabel(r'$log p( )$')
                    plt.title('Data log-likelihoods')
                    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
                    if self.reward_prior['gibbs_plot_save'] == 'show': 
                        plt.show()
                    else:
                        plt.savefig(self.reward_prior['gibbs_plot_save']+'/{}_loglik_t{}.pdf'.format(self.__class__.__name__,t), format='pdf', bbox_inches='tight')
                    plt.close()
                    
        # TODO: Add other reward/prior combinations
        else:
            raise ValueError('Invalid reward_function={} with reward_prior={} combination'.format(self.reward_prior['type'], self.reward_prior['dist']))

        print('update_reward_posterior at t={} in {}'.format(t,time.process_time()-t_init))
            
    def update_reward_posterior_params(self, how, a, k, x, y):
    
        # Parameters for linear Gaussian Mixture with Normal Inverse Gamma conjugate prior
        if self.reward_prior['type'] == 'linear_gaussian_mixture' and self.reward_prior['dist'] == 'NIG':
            # For arm and mixture of interest
            if how == 'add':
                # Update alpha
                self.reward_posterior['alpha'][a,k]+=1/2
                # Update beta
                self.reward_posterior['beta'][a,k]+=np.power(y-np.einsum('d,d->', x, self.reward_posterior['theta'][a,k]),2)/(2*(1+np.einsum('d,da,a->',x,self.reward_posterior['Sigma'][a,k],x)))
                # Sigma inverse
                sigma_inv=np.linalg.inv(self.reward_posterior['Sigma'][a,k])
                # Update regressor covariance (V)
                self.reward_posterior['Sigma'][a,k]=np.linalg.inv(sigma_inv+x[:,None]*x[None,:])
                # Update regressors (u)
                self.reward_posterior['theta'][a,k]=np.einsum('ab,b->a', self.reward_posterior['Sigma'][a,k],(x*y+np.einsum('ab,b->a',sigma_inv, self.reward_posterior['theta'][a,k])))
            
            elif how == 'del':
                # Sigma inverse
                sigma_inv=np.linalg.inv(self.reward_posterior['Sigma'][a,k])
                # Update regressor covariance (V)
                self.reward_posterior['Sigma'][a,k]=np.linalg.inv(sigma_inv-x[:,None]*x[None,:])
                # Update regressors (u)
                self.reward_posterior['theta'][a,k]=np.einsum('ab,b->a', self.reward_posterior['Sigma'][a,k],(np.einsum('ab,b->a',sigma_inv, self.reward_posterior['theta'][a,k])-x*y))
                # Update alpha
                self.reward_posterior['alpha'][a,k]-=1/2
                # Update beta
                self.reward_posterior['beta'][a,k]-=np.power(y-np.einsum('d,d->', x, self.reward_posterior['theta'][a,k]),2)/(2*(1+np.einsum('d,da,a->',x,self.reward_posterior['Sigma'][a,k],x)))
            else:
                raise ValueError('Unknown posterior parameter update type={}'.format(how))
            
            # Doublecheck posterior update makes sense
            # Positive inverse gamma parameters
            assert np.all(self.reward_posterior['alpha'][a]>0.)
            assert np.all(self.reward_posterior['beta'][a]>0.)
            # Positive definite Sigma
            assert np.all(np.linalg.eigvals(self.reward_posterior['Sigma'][a])>.0)
            
    def compute_ylikelihood_per_mixture(self, a, x, y):
        # Sufficient statistics of posterior
        nu_y=2*self.reward_posterior['alpha'][a,:self.reward_posterior['K'][a]]
        m_y=np.einsum('d,kd->k', x, self.reward_posterior['theta'][a,:self.reward_posterior['K'][a]])
        r_y=self.reward_posterior['beta'][a,:self.reward_posterior['K'][a]]/self.reward_posterior['alpha'][a,:self.reward_posterior['K'][a]]*(1+np.einsum('d,kdd,d->k', x, self.reward_posterior['Sigma'][a,:self.reward_posterior['K'][a]], x))
        # Likelihood of predictive distribution
        f_y_k=stats.t.pdf(y, nu_y, loc=m_y , scale=np.sqrt(r_y))
        
        # Doublechecking
        # Zeros
        if np.any(f_y_k==0):
            f_y_k[f_y_k==0]=np.finfo(float).eps
        # Infs
        if np.any(np.isinf(f_y_k)):
            f_y_k[np.isinf(f_y_k)]=1.0
        # NaNs
        assert not np.any(np.isnan(f_y_k)), 'Nan in ylikelihood_per_mixture f_y_k={}'.format(f_y_k)
        return f_y_k
        
    def compute_ylikelihood_for_new_mixture(self, a, x, y):
        # Sufficient statistics of prior
        nu_y=2*self.reward_prior['alpha'][a]
        m_y=np.einsum('d,d->', x, self.reward_prior['theta'][a])
        r_y=self.reward_prior['beta'][a]/self.reward_prior['alpha'][a]*(1+np.einsum('d,dd,d->', x, self.reward_prior['Sigma'][a], x))
        # Likelihood of predictive distribution
        f_y_newk=(stats.t.pdf(y, nu_y, loc=m_y , scale=np.sqrt(r_y)))[None]
        
        # Doublechecking
        # Zeros
        if np.any(f_y_newk==0):
            f_y_newk[f_y_newk==0]=np.finfo(float).eps
        # Infs
        if np.any(np.isinf(f_y_newk)):
            f_y_newk[np.isinf(f_y_newk)]=1.0
        # NaNs
        assert not np.any(np.isnan(f_y_newk)), 'Nan in likelihood_for_new_mixture f_y_newk={}'.format(f_y_newk)
        return f_y_newk

    def compute_loglikelihood(self, a, z_a, N_ak, x_a, y_a):
        """ Compute the log-likelihoods
            X given Z
            Z        
        Args:
            a: arm played
            z_a: assignments
            N_ak: assignment sufficient statistics
            x_a: relevant context
            y_a: relevant rewards
        Returns:
            (XcondZ, Z) = Tuple with loglikelihood of X given Z and Z
        """
        
        # Compute logp(X|Z)
        XcondZ_loglik=self.__compute_loglikelihood_XcondZ(a, z_a, N_ak, x_a, y_a)
        
        # Compute logp(Z)
        Z_loglik=self.__compute_loglikelihood_Z(a, N_ak)
        
        return XcondZ_loglik, Z_loglik
        
    def __compute_loglikelihood_XcondZ(self, a, z_a, N_ak, x_a, y_a):
        """ Compute the log-likelihood of X, given assignments Z
        Args:
            a: arm played
            z_a: assignments
            N_ak: assignment sufficient statistics
            x_a: relevant context
            y_a: relevant rewards
        """

        if self.reward_prior['type'] == 'linear_gaussian_mixture' and self.reward_prior['dist'] == 'NIG':
            # Sufficient statistics of posterior
            nu_Y=2*self.reward_posterior['alpha'][a]
            Omega_Y=2*self.reward_posterior['beta'][a]
            XcondZ_loglik=0.
            # For each valid k
            for k in np.arange(self.reward_posterior['K'][a]):
                # Find and count
                k_idx=(z_a==k)
                n_Y=k_idx.sum()
                # Suff stats
                M_Y=np.einsum('dn,d->n', x_a[:,k_idx], self.reward_posterior['theta'][a,k])
                Psi_Y=np.eye(n_Y)+np.einsum('an,ab,bt->nt', x_a[:,k_idx], self.reward_posterior['Sigma'][a,k], x_a[:,k_idx])
                # Inside det
                tmp=np.eye(n_Y)+np.einsum('ab, bc-> ac', np.linalg.inv(Psi_Y), (y_a[k_idx]-M_Y)[:,None] * (y_a[k_idx]-M_Y)[None,:]/Omega_Y[k])
                # Add this k loglikelihood
                XcondZ_loglik+=special.gammaln((nu_Y[k]+n_Y)/2) - special.gammaln((nu_Y[k])/2) -n_Y/2*(np.log(np.pi)+np.log(Omega_Y[k]))-1/2*np.log(np.linalg.det(Psi_Y))-((nu_Y[k]+n_Y)/2)*np.log(np.linalg.det(tmp))
                
                if np.isinf(XcondZ_loglik):
                    pdb.set_trace()

        else:
            raise ValueError('reward_prior type {} not implemented yet'.format(self.reward_prior['type']))

        return XcondZ_loglik

    def __compute_loglikelihood_Z(self, a, N_ak):
        """ Compute the log-likelihood of Z
        Args:
            a: arm played
            N_ak: assignment sufficient statistics
        """

        if self.reward_prior['K'] != 'nonparametric':
            Z_loglik=special.gammaln(self.reward_prior['gamma'][a].sum())-special.gammaln(self.reward_prior['gamma'][a].sum()+N_ak.sum())+special.gammaln(self.reward_prior['gamma'][a]+N_ak).sum()-special.gammaln(self.reward_prior['gamma'][a]).sum()
            
        elif self.reward_prior['K'] == 'nonparametric':
            Z_loglik=special.gammaln(self.reward_prior['gamma'][a])-special.gammaln(self.reward_prior['gamma'][a] + N_ak[:self.reward_posterior['K'][a]].sum()) + self.reward_posterior['K'][a] * np.log(self.reward_prior['gamma'][a]) + special.gammaln(N_ak[:self.reward_posterior['K'][a]]).sum()
            
        else:
            raise ValueError('Mixture prior {} not implemented yet'.format(self.reward_prior['K']))

        if np.isinf(Z_loglik):
            pdb.set_trace()

        return Z_loglik
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
