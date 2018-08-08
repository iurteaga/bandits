#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Bandits
from Bandit import * 
# Optimal
from OptimalBandit import * 
# Sampling based
from BayesianBanditSampling import * 
from MCBanditSampling import * 
from VariationalBanditSampling import * 
# Quantile based
from BayesianBanditQuantiles import * 
from MCBanditQuantiles import * 
from VariationalBanditQuantiles import * 
 
################################
# Bandit plotting functions
################################

### GENERAL bandits
# Bandit plotting function: rewards 
def bandits_plot_rewards(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot rewards for a set of bandits
        
        Args:
            bandits: bandit list
            colors: color list for each bandit
            labels: label list for each bandit
            t_plot: max time to plot
            plot_std: whether to plot standard deviations or not
            plot_save: whether to save (in given dir) or not plots
        Rets:
            None
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

# Bandit plotting function: cumulative rewards 
def bandits_plot_cumrewards(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot cumulative rewards for a set of bandits
        
        Args:
            bandits: bandit list
            colors: color list for each bandit
            labels: label list for each bandit
            t_plot: max time to plot
            plot_std: whether to plot standard deviations or not
            plot_save: whether to save (in given dir) or not plots
        Rets:
            None
    """
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

# Bandit plotting function: expected rewards
def bandits_plot_rewards_expected(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot rewards for a set of bandits
        
        Args:
            bandits: bandit list
            colors: color list for each bandit
            labels: label list for each bandit
            t_plot: max time to plot
            plot_std: whether to plot standard deviations or not
            plot_save: whether to save (in given dir) or not plots
        Rets:
            None
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
            colors: color list for each bandit
            labels: label list for each bandit
            t_plot: max time to plot
            plot_std: whether to plot standard deviations or not
            plot_save: whether to save (in given dir) or not plots
        Rets:
            None
    """
    # Regret over time
    plt.figure()
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot), bandit.regrets_R['mean'][0,0:t_plot], colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit.regrets_R['mean'][0,0:t_plot]-np.sqrt(bandit.regrets_R['var'][0,0:t_plot]), bandit.regrets_R['mean'][0,0:t_plot]+np.sqrt(bandit.regrets_R['var'][0,0:t_plot]),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$r_t=y_t^*-y_t$')
    plt.title('Regret over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/regret_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Bandit plotting function: regrets 
def bandits_plot_cumregret(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot cumulative regrest for a set of bandits
        
        Args:
            bandits: bandit list
            colors: color list for each bandit
            labels: label list for each bandit
            t_plot: max time to plot
            plot_std: whether to plot standard deviations or not
            plot_save: whether to save (in given dir) or not plots
        Rets:
            None
    """
    # Cumulative regret over time
    plt.figure()
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot), bandit.cumregrets_R['mean'][0,0:t_plot], colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit.cumregrets_R['mean'][0,0:t_plot]-np.sqrt(bandit.cumregrets_R['var'][0,0:t_plot]), bandit.cumregrets_R['mean'][0,0:t_plot]+np.sqrt(bandit.cumregrets_R['var'][0,0:t_plot]),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$R_t=\sum_{t=0}^T y_t^*-y_t$')
    plt.title('Cumulative regret over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/cumregret_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Bandit plotting function: actions
def bandits_plot_actions(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot the played (averaged) actions for a set of bandits
        
        Args:
            bandits: bandit list
            colors: color list for each bandit
            labels: label list for each bandit
            t_plot: max time to plot
            plot_std: whether to plot standard deviations or not
            plot_save: whether to save (in given dir) or not plots
        Rets:
            None
    """
    # Action (average probabilities) over time
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
            colors: color list for each bandit
            labels: label list for each bandit
            t_plot: max time to plot
            plot_std: whether to plot standard deviations or not
            plot_save: whether to save (in given dir) or not plots
        Rets:
            None
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

## SAMPLING bandits
# Bandit Sampling plotting function: arm predictive density
def bandits_plot_arm_density(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot the computed predictive arm density for a set of bandits
        
        Args:
            bandits: bandit list
            colors: color list for each bandit
            labels: label list for each bandit
            t_plot: max time to plot
            plot_std: whether to plot standard deviations or not
            plot_save: whether to save (in given dir) or not plots
        Rets:
            None
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

# Bandit Sampling plotting function: correct action predictive density percentages
def bandits_plot_action_density_correct(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot the probability of the predictive action density being correct for a set of bandits
        
        Args:
            bandits: bandit list
            colors: color list for each bandit
            labels: label list for each bandit
            t_plot: max time to plot
            plot_std: whether to plot standard deviations or not
            plot_save: whether to save (in given dir) or not plots
        Rets:
            None
    """
    # Correct argmax(action_density)
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

# Bandit Sampling plotting function: arm_N_samples 
def bandits_plot_arm_n_samples(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot number of arm samples for a set of sampling bandits
        
        Args:
            bandits: bandit list
            colors: color list for each bandit
            labels: label list for each bandit
            t_plot: max time to plot
            plot_std: whether to plot standard deviations or not
            plot_save: whether to save (in given dir) or not plots
        Rets:
            None
    """
    # Arm N samples over time
    plt.figure()
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot), bandit.arm_N_samples_R['mean'][0:t_plot], colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit.arm_N_samples_R['mean'][0:t_plot]-np.sqrt(bandit.arm_N_samples_R['var'][0:t_plot]), bandit.arm_N_samples_R['mean'][0:t_plot]+np.sqrt(bandit.arm_N_samples_R['var'][0:t_plot]),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$M_t$')
    plt.title('arm_N_samples over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/arm_N_samples_R_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

## QUANTILE bandits
# Bandit Quantiles plotting function: arm quantiles
def bandits_plot_arm_quantile(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot the computed arm quantile for a set of bandits
        
        Args:
            bandits: bandit list
            colors: color list for each bandit
            labels: label list for each bandit
            t_plot: max time to plot
            plot_std: whether to plot standard deviations or not
            plot_save: whether to save (in given dir) or not plots
        Rets:
            None
    """
    # arm quantiles over time
    for a in np.arange(0,bandits[0].A):
        plt.figure()
        for (n,bandit) in enumerate(bandits):
            if isinstance(bandit,BanditQuantiles):
                plt.plot(np.arange(t_plot), bandit.arm_quantile_R['mean'][a,0:t_plot], colors[n], label=labels[n])
                if plot_std:
                    plt.fill_between(np.arange(t_plot), bandit.arm_quantile_R['mean'][a,0:t_plot]-np.sqrt(bandit.arm_quantile_R['var'][a,0:t_plot]), bandit.arm_quantile_R['mean'][a,0:t_plot]+np.sqrt(bandit.arm_quantile_R['var'][a,0:t_plot]),alpha=0.5, facecolor=colors[n])
        plt.ylabel(r'$P(\mu_a<x)\leq \alpha $')
        plt.xlabel('t')
        plt.title('Averaged Action Quantiles for arm {}'.format(a))
        plt.xlim([0, t_plot-1])
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(plot_save+'/action_quantile_{}_std{}.pdf'.format(a,str(plot_std)), format='pdf', bbox_inches='tight')
            plt.close()

# Bandit Quantiles plotting function: correct action quantile percentages
def bandits_plot_action_quantile_correct(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot the probability of the quantile based action being correct for a set of bandits
        
        Args:
            bandits: bandit list
            colors: color list for each bandit
            labels: label list for each bandit
            t_plot: max time to plot
            plot_std: whether to plot standard deviations or not
            plot_save: whether to save (in given dir) or not plots
        Rets:
            None
    """
    # Correct argmax(action_quantile)
    plt.figure()
    for (n,bandit) in enumerate(bandits):
        if isinstance(bandit,BanditQuantiles):
            plt.plot(np.arange(t_plot), (bandit.arm_quantile_R['mean'].argmax(axis=0)==(bandit.A-1)).astype(int), colors[n], label=labels[n])
    plt.xlabel('t')
    plt.ylabel('% Correct')
    plt.title('Correct action predictive density percentage')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/action_quantile_correct_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()        

# Bandit plotting function: all
def bandits_plot_all(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot all results for a set of bandits
        
        Args:
            bandits: bandit list
        Rets:
    """

    ## General bandits
    # rewards over time
    bandits_plot_rewards(bandits, colors, labels, t_plot, plot_std, plot_save)

    # Regret over time
    bandits_plot_regret(bandits, colors, labels, t_plot, plot_std, plot_save)
   
    # Actions over time
    bandits_plot_actions(bandits, colors, labels, t_plot, plot_std, plot_save)

    # Correct actions over time
    bandits_plot_actions_correct(bandits, colors, labels, t_plot, plot_std, plot_save)
    
    ## Sampling Bandits
    # Arm predictive density probabilities over time
    bandits_plot_arm_density(bandits, colors, labels, t_plot, plot_std, plot_save)
    
    # Correct arm predictive density percentages
    bandits_plot_arm_density_correct(bandits, colors, labels, t_plot, plot_std, plot_save)
    
    # Number of arm samples over time
    bandits_plot_arm_n_samples(bandits, colors, labels, t_plot, plot_std, plot_save)
    	
    ## Quantile Bandits
    # Arm quantiles over time
    bandits_plot_arm_quantile(bandits, colors, labels, t_plot, plot_std, plot_save)
    
    # Correct arm quantile percentages
    bandits_plot_arm_quantile_correct(bandits, colors, labels, t_plot, plot_std, plot_save)
    
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
