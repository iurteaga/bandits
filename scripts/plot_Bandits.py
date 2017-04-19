#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from Bandit import * 
from OptimalBandit import * 
from BanditSampling import * 
from BayesianBanditSampling import * 
from VariationalBanditSampling import * 

################################
# Bandit plotting functions
################################

# Bandit plotting function: rewards 
def bandits_plot_rewards(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot rewards for a set of bandits
        
        Args:
            bandits: bandit list
        Rets:
    """
       
    # rewards over time
    plt.figure()
    plt.plot(np.arange(t_plot), bandits[0].true_expected_rewards.max(axis=0)[0:t_plot], 'k', label='Expected')
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot), bandit.rewards_R['mean'][0,0:t_plot], colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit.rewards_R['mean'][0,0:t_plot]-np.sqrt(bandit.rewards_R['var'][0,0:t_plot]), bandit.rewards_R['mean'][0,0:t_plot]+np.sqrt(bandit.rewards_R['var'][0,0:t_plot]),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$y_t$')
    plt.title('rewards over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/rewards_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    # Cumulative rewards over time
    plt.figure()
    plt.plot(np.arange(t_plot), bandits[0].true_expected_rewards.max(axis=0)[0:t_plot], 'k', label='Expected')
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot),bandit.rewards_R['mean'][0,0:t_plot].cumsum(axis=1), colors[n], label=labels[n])
    plt.xlabel('t')
    plt.ylabel(r'$\sum_{t=0}^Ty_t$')
    plt.title('Cumulative rewards over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/rewards_cumulative_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Bandit plotting function: rewards expected
def bandits_plot_rewards_expected(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot rewards for a set of bandits
        
        Args:
            bandits: bandit list
        Rets:
    """
    
    # Expected rewards per arm, over time
    for a in np.arange(0,bandits[0].A):
        plt.figure()
        plt.plot(np.arange(t_plot), bandits[0].true_expected_rewards[a,0:t_plot], 'k', label='Expected')
        for (n,bandit) in enumerate(bandits):
            plt.plot(np.arange(t_plot), bandit.rewards_expected_R['mean'][a,0:t_plot], colors[n], label=labels[n])
            if plot_std:
                plt.fill_between(np.arange(t_plot), bandit.rewards_expected_R['mean'][a,0:t_plot]-np.sqrt(bandit.rewards_expected_R['var'][a,0:t_plot]), bandit.rewards_expected_R['mean'][a,0:t_plot]+np.sqrt(bandit.rewards_expected_R['var'][a,0:t_plot]),alpha=0.5, facecolor=colors[n])
        plt.ylabel(r'$E\{\mu_{a,t}\}$')
        plt.xlabel('t')
        plt.title('Expected rewards over time for arm {}'.format(a))
        plt.xlim([0, t_plot-1])
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(plot_save+'/rewards_expected_{}_std{}.pdf'.format(a,str(plot_std)), format='pdf', bbox_inches='tight')
            plt.close()

# Bandit plotting function: regrets 
def bandits_plot_regret(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot regrets for a set of bandits
        
        Args:
            bandits: bandit list
        Rets:
    """
    
    # Regret over time
    plt.figure()
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot), bandit.regrets_R['mean'][0,0:t_plot], colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit.regrets_R['mean'][0,0:t_plot]-np.sqrt(bandit.regrets_R['var'][0,0:t_plot]), bandit.regrets_R['mean'][0,0:t_plot]+np.sqrt(bandit.regrets_R['var'][0,0:t_plot]),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$l_t=y_t^*-y_t$')
    plt.title('Regret over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/regret_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    # Cumulative regret over time
    plt.figure()
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot), bandit.regrets_R['mean'][0,0:t_plot].cumsum(), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit.regrets_R['mean'][0,0:t_plot].cumsum()-np.sqrt(bandit.regrets_R['var'][0,0:t_plot]), bandit.regrets_R['mean'][0,0:t_plot].cumsum()+np.sqrt(bandit.regrets_R['var'][0,0:t_plot]),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$L_t=\sum_{t=0}^T y_t^*-y_t$')
    plt.title('Cumulative regret over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/regret_cumulative_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()


# Bandit plotting function: arm predictive density
def bandits_plot_arm_density(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot the computed predictive arm density for a set of bandits
        
        Args:
            bandits: bandit list
        Rets:
    """
    
    # arm predictive density probabilities over time
    for a in np.arange(0,bandits[0].A):
        plt.figure()
        for (n,bandit) in enumerate(bandits):
            if isinstance(bandit,BanditSampling):
                plt.plot(np.arange(t_plot), bandit.arm_predictive_density_R['mean'][a,0:t_plot], colors[n], label=labels[n])
                if plot_std:
                    plt.fill_between(np.arange(t_plot), bandit.arm_predictive_density_R['mean'][a,0:t_plot]-np.sqrt(bandit.arm_predictive_density_R['var'][a,0:t_plot]), bandit.arm_predictive_density_R['mean'][a,0:t_plot]+np.sqrt(bandit.arm_predictive_density_R['var'][a,0:t_plot]),alpha=0.5, facecolor=colors[n])
        plt.ylabel(r'$f(a_{t+1}=a|a_{1:t}, y_{1:t})$')
        plt.xlabel('t')
        plt.title('Averaged Action Predictive density probabilities for arm {}'.format(a))
        plt.xlim([0, t_plot-1])
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(plot_save+'/action_density_{}_std{}.pdf'.format(a,str(plot_std)), format='pdf', bbox_inches='tight')
            plt.close()

# Bandit plotting function: correct action predictive density percentages
def bandits_plot_action_density_correct(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot the probability of the predictive action density being correct for a set of bandits
        
        Args:
            bandits: bandit list
        Rets:
    """
    plt.figure()
    for (n,bandit) in enumerate(bandits):
        if isinstance(bandit,BanditSampling):
            plt.plot(np.arange(t_plot), (bandit.arms_predictive_density_R['mean'].argmax(axis=0)==(bandit.A-1)).astype(int), colors[n], label=labels[n])
    plt.xlabel('t')
    plt.ylabel('% Correct')
    plt.title('Correct action predictive density percentage')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/action_density_correct_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Bandit plotting function: actions
def bandits_plot_actions(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot the actions selected for a set of bandits
        
        Args:
            bandits: bandit list
        Rets:
    """
      
    # Action probabilities over time
    for a in np.arange(0,bandits[0].A):
        plt.figure()
        for (n,bandit) in enumerate(bandits):
            plt.plot(np.arange(t_plot), bandit.actions_R['mean'][a,0:t_plot], colors[n], label=labels[n]+' actions')
            if plot_std:
                plt.fill_between(np.arange(t_plot), bandit.actions_R['mean'][a,0:t_plot]-np.sqrt(bandit.actions_R['var'][a,0:t_plot]), bandit.actions_R['mean'][a,0:t_plot]+np.sqrt(bandit.actions_R['var'][a,0:t_plot]),alpha=0.5, facecolor=colors[n])
        plt.ylabel(r'$f(a_{t+1}=a|a_{1:t}, y_{1:t})$')
        plt.xlabel('t')
        plt.title('Averaged Action probabilities for arm {}'.format(a))
        plt.xlim([0, t_plot-1])
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(plot_save+'/actions_{}_std{}.pdf'.format(a,str(plot_std)), format='pdf', bbox_inches='tight')
            plt.close()

# Bandit plotting function: correct actions
def bandits_plot_actions_correct(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot the probability of selecting correct actions for a set of bandits
        
        Args:
            bandits: bandit list
        Rets:
    """
   
	# Correct arm selection probability
    plt.figure()
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot), bandit.actions_R['mean'][bandit.true_expected_rewards.argmax(axis=0)[0:t_plot],np.arange(t_plot)], colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit.actions_R['mean'][bandit.true_expected_rewards.argmax(axis=0),np.arange(t_plot)]-np.sqrt(bandit.actions_R['var'][bandit.true_expected_rewards.argmax(axis=0),np.arange(t_plot)]), bandit.actions_R['mean'][bandit.true_expected_rewards.argmax(axis=0),np.arange(t_plot)]+np.sqrt(bandit.actions_R['var'][bandit.true_expected_rewards.argmax(axis=0),np.arange(t_plot)]),alpha=0.5, facecolor=colors[n])
    plt.ylabel(r'$f(a_{t+1}=a^*|a_{1:t}, y_{1:t})$')
    plt.xlabel('t')
    plt.title('Averaged Correct Action probabilities')
    plt.axis([0, t_plot-1, 0, 1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/correct_actions_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Bandit plotting function: n_samples 
def bandits_plot_n_samples(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot number of samples for a set of sampling bandits
        
        Args:
            bandits: bandit list
        Rets:
    """
    
    # Regret over time
    plt.figure()
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot), bandit.n_samples_R['mean'][0:t_plot], colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit.n_samples_R['mean'][0:t_plot]-np.sqrt(bandit.n_samples_R['var'][0:t_plot]), bandit.n_samples_R['mean'][0:t_plot]+np.sqrt(bandit.n_samples_R['var'][0:t_plot]),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$M_t$')
    plt.title('n_samples over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/n_samples_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Bandit plotting function: all
def bandits_plot_all(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot all results for a set of bandits
        
        Args:
            bandits: bandit list
        Rets:
    """
        
    # rewards over time
    bandits_plot_rewards(bandits, colors, labels, t_plot, plot_std, plot_save)

    # Regret over time
    bandits_plot_regret(bandits, colors, labels, t_plot, plot_std, plot_save)

    # Arm predictive density probabilities over time
    bandits_plot_arm_density(bandits, colors, labels, t_plot, plot_std, plot_save)
    
    # Correct arm predictive density percentages
    bandits_plot_arm_density_correct(bandits, colors, labels, t_plot, plot_std, plot_save)
   
    # Actions over time
    bandits_plot_actions(bandits, colors, labels, t_plot, plot_std, plot_save)

    # Correct actions over time
    bandits_plot_actions_correct(bandits, colors, labels, t_plot, plot_std, plot_save)
    
    # Number of samples over time
    bandits_plot_n_samples(bandits, colors, labels, t_plot, plot_std, plot_save)
    	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
