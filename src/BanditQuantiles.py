#!/usr/bin/python

# Imports: python modules
# Imports: other modules
from Bandit import *

######## Class definition ########
class BanditQuantiles(Bandit):
    """ Abstract class for bandits with policies based on quantiles (upper-confidence bounds)
        These bandits decide which arm to play by picking the arm with the highest expected reward quantile

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
        super().__init__(A, reward_function)
        
        # Reward prior
        self.reward_prior=reward_prior
        # Quantile computation strategy
        self.quantile_info=quantile_info
        
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
            self.arm_quantile_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
        elif exec_type =='batch':
            self.rewards_R={'all':np.zeros((R,1,t_max)), 'mean':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.regrets_R={'all':np.zeros((R,1,t_max)), 'mean':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.cumregrets_R={'all':np.zeros((R,1,t_max)), 'mean':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.rewards_expected_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.actions_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.arm_quantile_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
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
                self.arm_quantile_R['mean'], self.arm_quantile_R['m2'], self.arm_quantile_R['var']=online_update_mean_var(r+1, self.arm_quantile, self.arm_quantile_R['mean'], self.arm_quantile_R['m2'])
            else:
                self.rewards_R['all'][r,0,:]=self.rewards.sum(axis=0)
                self.regrets_R['all'][r,0,:]=self.regrets
                self.cumregrets_R['all'][r,0,:]=self.cumregrets
                self.rewards_expected_R['all'][r,:,:]=self.rewards_expected
                self.actions_R['all'][r,:,:]=self.actions
                self.arm_quantile_R['all'][r,:,:]=self.arm_quantile
                
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
            self.arm_quantile_R['mean']=self.arm_quantile_R['all'].mean(axis=0)
            self.arm_quantile_R['var']=self.arm_quantile_R['all'].var(axis=0)
                
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
        self.arm_quantile=np.zeros((self.A,t_max))

        # Initialize reward posterior
        self.init_reward_posterior()

        # Execute the bandit for each time instant
        print('Start running bandit')
        for t in np.arange(t_max):
            #print('Running time instant {}'.format(t))
            
            # Compute quantile values for each arm
            self.compute_arm_quantile(t)
            
            # Pick next action, by maximum of quantiles 
            action = self.arm_quantile[:,t].argmax()
            self.actions[action,t]=1.

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

    @abc.abstractmethod            
    def compute_arm_quantile(self, t):
        """ Abstract method to compute the quantile values for each arm
            It is based on available information at time t, which depends on posterior update type
            
            Different alternatives on computing the quantiles are considered
                quantile_info['type']='analytical': use the exact analytical quantile (inverse CDF) function of the expected rewards 
                quantile_info['type']='empirical': estimate the quantile (inverse CDF) function from expected reward samples
        Args:
            t: time of the execution of the bandit
        """
        
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
