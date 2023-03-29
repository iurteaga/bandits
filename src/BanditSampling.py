#!/usr/bin/python

# Imports: python modules
# Imports: other modules
from Bandit import * 

######## Class definition ########
class BanditSampling(Bandit):
    """ Abstract class for bandits with sampling policies
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
        super().__init__(A, reward_function)
        
        # Reward prior
        self.reward_prior=reward_prior
        # Arm predictive computation strategy
        self.arm_predictive_policy=arm_predictive_policy
        
    def execute_realizations(self, R, t_max, context=None, exec_type='sequential'):
        """ Execute R realizations of the bandit
        Args:
            R: number of realizations to run
            t_max: maximum execution time for the bandit
            context: d_context by (at_least) t_max array with context for every time instant (None if does not apply)
            exec_type: batch (keep data from all realizations) or sequential (update mean and variance of realizations data)
        """

        # Allocate overall variables
        if exec_type == 'sequential':
            self.rewards_R={'mean':np.zeros((1,t_max)), 'm2':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.regrets_R={'mean':np.zeros((1,t_max)), 'm2':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.cumregrets_R={'mean':np.zeros((1,t_max)), 'm2':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.rewards_expected_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.actions_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.arm_predictive_density_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.arm_N_samples_R={'mean':np.zeros(t_max), 'm2':np.zeros(t_max), 'var':np.zeros(t_max)}
        elif exec_type =='batch':
            self.rewards_R={'all':np.zeros((R,1,t_max)), 'mean':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.regrets_R={'all':np.zeros((R,1,t_max)), 'mean':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.cumregrets_R={'all':np.zeros((R,1,t_max)), 'mean':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.rewards_expected_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.actions_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.arm_predictive_density_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.arm_predictive_density_var_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.arm_N_samples_R={'all':np.zeros((R,t_max)),'mean':np.zeros(t_max), 'var':np.zeros(t_max)}            
        else:
            raise ValueError('Execution type={} not implemented'.format(exec_type))
            
        # Execute all
        for r in np.arange(R):
            # Run one realization
            print('Executing realization {}'.format(r))
            self.execute(t_max, context)

            if exec_type == 'sequential':
                # Update overall mean and variance sequentially
                self.rewards_R['mean'], self.rewards_R['m2'], self.rewards_R['var']=online_update_mean_var(r+1, self.rewards.sum(axis=0), self.rewards_R['mean'], self.rewards_R['m2'])
                self.regrets_R['mean'], self.regrets_R['m2'], self.regrets_R['var']=online_update_mean_var(r+1, self.regrets, self.regrets_R['mean'], self.regrets_R['m2'])
                self.cumregrets_R['mean'], self.cumregrets_R['m2'], self.cumregrets_R['var']=online_update_mean_var(r+1, self.cumregrets, self.cumregrets_R['mean'], self.cumregrets_R['m2'])
                self.rewards_expected_R['mean'], self.rewards_expected_R['m2'], self.rewards_expected_R['var']=online_update_mean_var(r+1, self.rewards_expected, self.rewards_expected_R['mean'], self.rewards_expected_R['m2'])
                self.actions_R['mean'], self.actions_R['m2'], self.actions_R['var']=online_update_mean_var(r+1, self.actions, self.actions_R['mean'], self.actions_R['m2'])
                self.arm_predictive_density_R['mean'], self.arm_predictive_density_R['m2'], self.arm_predictive_density_R['var']=online_update_mean_var(r+1, self.arm_predictive_density['mean'], self.arm_predictive_density_R['mean'], self.arm_predictive_density_R['m2'])
                self.arm_N_samples_R['mean'], self.arm_N_samples_R['m2'], self.arm_N_samples_R['var']=online_update_mean_var(r+1, self.arm_N_samples, self.arm_N_samples_R['mean'], self.arm_N_samples_R['m2'])
            else:
                self.rewards_R['all'][r,0,:]=self.rewards.sum(axis=0)
                self.regrets_R['all'][r,0,:]=self.regrets
                self.cumregrets_R['all'][r,0,:]=self.cumregrets
                self.rewards_expected_R['all'][r,:,:]=self.rewards_expected
                self.actions_R['all'][r,:,:]=self.actions
                self.arm_predictive_density_R['all'][r,:,:]=self.arm_predictive_density['mean']
                self.arm_predictive_density_var_R['all'][r,:,:]=self.arm_predictive_density['var']
                self.arm_N_samples_R['all'][r,:]=self.arm_N_samples
                
        if exec_type == 'batch':
            # Compute sufficient statistics
            self.rewards_R['mean']=self.rewards_R['all'].mean(axis=0)
            self.rewards_R['var']=self.rewards_R['all'].var(axis=0)
            self.regrets_R['mean']=self.regrets_R['all'].mean(axis=0)
            self.regrets_R['var']=self.regrets_R['all'].var(axis=0)
            self.cumregrets_R['mean']=self.cumregrets_R['all'].mean(axis=0)
            self.cumregrets_R['var']=self.cumregrets_R['all'].var(axis=0)
            self.rewards_expected_R['mean']=self.rewards_expected_R['all'].mean(axis=0)
            self.rewards_expected_R['var']=self.rewards_expected_R['all'].var(axis=0)
            self.actions_R['mean']=self.actions_R['all'].mean(axis=0)
            self.actions_R['var']=self.actions_R['all'].var(axis=0)
            self.arm_predictive_density_R['mean']=self.arm_predictive_density_R['all'].mean(axis=0)
            self.arm_predictive_density_R['var']=self.arm_predictive_density_R['all'].var(axis=0)
            self.arm_N_samples_R['mean']=self.arm_N_samples_R['all'].mean(axis=0)
            self.arm_N_samples_R['var']=self.arm_N_samples_R['all'].var(axis=0)
                
    def execute(self, t_max, context=None):
        """ Execute the Bayesian bandit
        Args:
            t_max: maximum execution time for the bandit
            context: d_context by (at_least) t_max array with context for every time instant (None if does not apply)
        """

        if np.all(context != None):
            # Contextual bandit
            self.d_context=context.shape[0]
            assert context.shape[1]>=t_max, 'Not enough context provided: context.shape[1]={} while t_max={}'.format(context.shape[1],t_max)
            self.context=context
        
        # Initialize attributes
        self.actions=np.zeros((self.A,t_max))
        self.rewards=np.zeros((self.A,t_max))
        self.rewards_expected=np.zeros((self.A,t_max))
        self.arm_predictive_density={'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
        self.arm_N_samples=np.ones(t_max)

        # Initialize reward posterior
        self.init_reward_posterior()
        
        # Execute the bandit for each time instant
        print('Start running bandit')
        for t in np.arange(t_max):
            #print('Running time instant {}'.format(t))
            
            # Compute predictive density for each arm
            self.compute_arm_predictive_density(t)

            # Compute number of candidate arm samples, based on sampling strategy
            self.arm_N_samples[t]=self.compute_arm_N_samples(t)
            
            # Pick next action
            if self.arm_N_samples[t] == np.inf:
                # Pick maximum 
                action = self.arm_predictive_density[:,t].argmax()
                self.actions[action,t]=1.
            else:
                # SAMPLE arm_N_samples and pick the most likely action                
                self.actions[np.random.multinomial(1,self.arm_predictive_density['mean'][:,t], size=int(self.arm_N_samples[t])).sum(axis=0).argmax(),t]=1
                action = np.where(self.actions[:,t]==1)[0][0]

            # Play selected arm
            self.play_arm(action, t)

            if np.isnan(self.rewards[action,t]):
                # This instance has not been played, and no parameter update (e.g. for logged data)
                self.actions[action,t]=0.
            else:
                # Update parameter posterior
                self.update_reward_posterior(t)

        print('Finished running bandit at {}'.format(t))
        # Compute expected rewards with true function
        self.compute_true_expected_rewards()
        # Compute regret
        self.regrets=self.true_expected_rewards.max(axis=0) - self.rewards.sum(axis=0)
        self.cumregrets=self.regrets.cumsum()
    
    def execute_interactive(self, t, y=None, context=None, t_max=100):
        """ Execute the Bayesian bandit interactively
                That is, return the selected action, based on history up to t
        Args:
            t: time/interaction for the bandit
            y: last observed reward (i.e., at time t)
            context: d_context by 1 array with context for this time instant (None if does not apply)
        Returns:
            a: next action to execute (a_{t+1} after observing y_t)
        """
        ######### Internal attributes 
        if t==0:
            # Initialize attributes
            self.actions=np.zeros((self.A,t_max))
            self.rewards=np.zeros((self.A,t_max))
            self.rewards_expected=np.zeros((self.A,t_max))
            if context is not None:
                # Contextual bandit
                self.d_context=context.shape[0]
                self.context=np.zeros((self.d_context,t_max))
            self.arm_predictive_density={'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.arm_N_samples=np.ones(t_max)        
        
        ######### Execute bandit
        # Capture context if needed
        if context is not None:
            # Contextual bandit
            self.d_context=context.shape[0]
            self.context[:,t]=context
            
        if t==0:
            # Initialize reward posterior
            self.init_reward_posterior()
        else:
            # Update history of rewards with latest value, for previously played action
            previous_action=np.where(self.actions[:,t-1])
            
            assert y is not None, 'Need observed reward at time {}'.format(t)
            self.rewards[previous_action,t-1]=y
            
            # Update parameter posterior given full evidence
            self.update_reward_posterior(t-1)
        
        # Compute predictive density for each arm
        self.compute_arm_predictive_density(t)

        # Compute number of candidate arm samples, based on sampling strategy
        self.arm_N_samples[t]=self.compute_arm_N_samples(t)
        
        # Pick next action
        if self.arm_N_samples[t] == np.inf:
            # Pick maximum 
            action = self.arm_predictive_density[:,t].argmax()
            self.actions[action,t]=1.
        else:
            # SAMPLE arm_N_samples and pick the most likely action                
            self.actions[np.random.multinomial(1,self.arm_predictive_density['mean'][:,t], size=int(self.arm_N_samples[t])).sum(axis=0).argmax(),t]=1
            action = np.where(self.actions[:,t]==1)[0][0]
        
        # Return action index
        return int(action)
        
    def compute_arm_N_samples(self,t):
        """ Determine the number of arm samples to draw, based on policy and information available at time t
        Args:
            t: time of the execution of the bandit
        """
        
        n_samples=0
        if self.arm_predictive_policy['sampling_type'] == 'static':
            # Static number of samples
            n_samples=self.arm_predictive_policy['arm_N_samples']
        elif self.arm_predictive_policy['sampling_type'] == 'infPfa':
            # Function of inverse of probability of other arms being optimal (prob false alarm)
            # Optimal action estimate and its "weight"
            a_opt=self.arm_predictive_density['mean'][:,t].argmax()
            w_opt=self.arm_predictive_density['mean'][a_opt,t]

            # Probability of "false alarm" computation:
            if self.arm_predictive_policy['Pfa'] == 'tGaussian':
                # Sufficient statistics for truncated Gaussian approximation
                xi=(w_opt-self.arm_predictive_density['mean'][np.arange(self.A)!=a_opt,t])/np.sqrt(self.arm_predictive_density['var'][np.arange(self.A)!=a_opt,t])
                alpha=(-self.arm_predictive_density['mean'][np.arange(self.A)!=a_opt,t])/np.sqrt(self.arm_predictive_density['var'][np.arange(self.A)!=a_opt,t])
                beta=(1-self.arm_predictive_density['mean'][np.arange(self.A)!=a_opt,t])/np.sqrt(self.arm_predictive_density['var'][np.arange(self.A)!=a_opt,t])
                # Compute average of per (suboptimal) arm false alarm probability
                p_fa=(1-(stats.norm.cdf(xi)-stats.norm.cdf(alpha))/(stats.norm.cdf(beta)-stats.norm.cdf(alpha))).sum()/(self.A-1)
            elif self.arm_predictive_policy['Pfa'] == 'Markov':
                # Markov's inequality
                # Compute average of per (suboptimal) arm false alarm probability
                p_fa=(self.arm_predictive_density['mean'][np.arange(self.A)!=a_opt,t]/w_opt).sum()/(self.A-1)
            elif self.arm_predictive_policy['Pfa'] == 'Chebyshev':
                # Chebyshev's inequality
                delta=w_opt - self.arm_predictive_density['mean'][np.arange(self.A)!=a_opt,t]
                # Compute average of per (suboptimal) arm false alarm probability
                p_fa=(self.arm_predictive_density['var'][np.arange(self.A)!=a_opt,t]/(2*np.power(delta,2))).sum()/(self.A-1)
            else:
                raise ValueError('Invalid Pfa computation type={}'.format(self.arm_predictive_policy['Pfa']))
            
            # Decide number of candidate samples, enforce at least 1 and limit max
            n_samples=np.fmin(np.maximum(self.arm_predictive_policy['f(1/Pfa)'](1/p_fa),1), self.arm_predictive_policy['N_max'])
        elif self.arm_predictive_policy['MC_type'] == 'argMax':
            # Infinite samples are equivalent to picking maximum
            n_samples=np.inf
        else:
            raise ValueError('Invalid arm predictive computation sampling type={}'.format(self.arm_predictive_policy['sampling_type']))
            
        # return number of samples
        return n_samples

    @abc.abstractmethod            
    def compute_arm_predictive_density(self, t):
        """ Abstract method to compute the predictive density of each arm
            It is based on available information at time t, which depends on posterior update type
            
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
	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
