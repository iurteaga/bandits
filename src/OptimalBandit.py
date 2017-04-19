#!/usr/bin/python

# Imports: python modules
# Imports: other modules
from Bandit import * 

######## Class definition ########
class OptimalBandit(Bandit):
    """Class for bandits with optimal policy
        This Bandit always picks the action with the highest expected reward, given true parameters
    Attributes:
        (all inherited)
    """
    
    def __init__(self, A, reward_function):
        """ Initialize the Bandit object and its attributes
        
        Args:
            A: the size of the bandit
            reward_function: the reward function of the bandit
        """
        
        # Initialize
        super().__init__(A, reward_function)
        
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
            self.rewards_expected_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.actions_R={'mean':np.zeros((self.A,t_max)), 'm2':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
        elif exec_type =='batch':
            self.rewards_R={'all':np.zeros((R,1,t_max)), 'mean':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.regrets_R={'all':np.zeros((R,1,t_max)), 'mean':np.zeros((1,t_max)), 'var':np.zeros((1,t_max))}
            self.rewards_expected_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
            self.actions_R={'all':np.zeros((R,self.A,t_max)), 'mean':np.zeros((self.A,t_max)), 'var':np.zeros((self.A,t_max))}
        else:
            raise ValueError('Execution type={} not implemented'.format(exec_type))
            
        # Execute all realizations
        for r in np.arange(R):
            # Run one realization
            print('Executing realization {}'.format(r))
            self.execute(t_max, context)

            if exec_type == 'sequential':
                # Update overall mean and variance sequentially
                self.rewards_R['mean'], self.rewards_R['m2'], self.rewards_R['var']=online_update_mean_var(r+1, self.rewards.sum(axis=0), self.rewards_R['mean'], self.rewards_R['m2'])
                self.regrets_R['mean'], self.regrets_R['m2'], self.regrets_R['var']=online_update_mean_var(r+1, self.regrets, self.regrets_R['mean'], self.regrets_R['m2'])
                self.rewards_expected_R['mean'], self.rewards_expected_R['m2'], self.rewards_expected_R['var']=online_update_mean_var(r+1, self.rewards_expected, self.rewards_expected_R['mean'], self.rewards_expected_R['m2'])
                self.actions_R['mean'], self.actions_R['m2'], self.actions_R['var']=online_update_mean_var(r+1, self.actions, self.actions_R['mean'], self.actions_R['m2'])
            else:
                self.rewards_R['all'][r,0,:]=self.rewards.sum(axis=0)
                self.regrets_R['all'][r,0,:]=self.regrets
                self.rewards_expected_R['all'][r,:,:]=self.rewards_expected
                self.actions_R['all'][r,:,:]=self.actions
                
        if exec_type == 'batch':
            # Compute sufficient statistics
            self.rewards_R['mean']=self.rewards_R['all'].mean(axis=0)
            self.rewards_R['var']=self.rewards_R['all'].var(axis=0)
            self.regrets_R['mean']=self.regrets_R['all'].mean(axis=0)
            self.regrets_R['var']=self.regrets_R['all'].var(axis=0)
            self.rewards_expected_R['mean']=self.rewards_expected_R['all'].mean(axis=0)
            self.rewards_expected_R['var']=self.rewards_expected_R['all'].var(axis=0)
            self.actions_R['mean']=self.actions_R['all'].mean(axis=0)
            self.actions_R['var']=self.actions_R['all'].var(axis=0)
        
    def execute(self, t_max, context=None):
        """ Execute the Bayesian bandit
        Args:
            t_max: maximum execution time for the bandit
            context: d_context by (at_least) t_max array with context for every time instant (None if does not apply)
        """

        if context != None:
            # Contextual bandit
            self.d_context=context.shape[0]
            assert context.shape[1]>=t_max, 'Not enough context provided: context.shape[1]={} while t_max={}'.format(context.shape[1],t_max)
            self.context=context
        
        # Initialize
        self.actions=np.zeros((self.A,t_max))
        self.rewards=np.zeros((self.A,t_max))
        # Compute expected rewards with true function
        self.compute_true_expected_rewards()
        # Optimal bandit expects true
        self.rewards_expected=self.true_expected_rewards
        # And decide optimal action that maximizes expected reward
        self.actions[self.true_expected_rewards.argmax(axis=0),np.arange(t_max)]=1
        # Simply draw from optimal action as many times as indicated
        self.play_arm(self.true_expected_rewards.argmax(axis=0), np.arange(t_max))
        # Compute regret
        self.regrets=self.true_expected_rewards.max(axis=0) - self.rewards.sum(axis=0)
        
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
