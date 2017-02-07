#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from Bandits import * 
from BayesianBandits import * 
from BayesianBanditsSampling import * 
from BayesianContextualBandits import * 

# Bandit execution function
def execute_bandits(K, bandits, R, t_max, plot_colors=False):
    """ Execute a set of bandits for t_max time instants over R realization 
        
        Args:
            bandits: list of bandits to execute
            R: number of realizations to run
            t_max: maximum time for the bandits' execution
            plot_colors: colors to use for plotting each realization (default is False, no plotting)
        Rets:
            returns: array with returns per realization (bandit idx by R by K by t_max)
            returns_expected: array with expected returns per realization (bandit idx by R by K by t_max)
            actions: array with actions per realization (bandit idx by R by K by t_max)
            predictive: dictionary-array with action predictive density per realization (bandit idx by R by K by t_max)
            n_samples: array with used action samples per realization (bandit idx by R by t_max)
    """
    
    n_bandits=len(bandits)
    
    # Allocate space
    returns=np.zeros((n_bandits, R, bandits[0].K, t_max))
    returns_expected=np.zeros((n_bandits, R, bandits[0].K, t_max))
    actions=np.zeros((n_bandits, R, bandits[0].K, t_max))
    predictive=np.zeros((n_bandits, R, bandits[0].K, t_max))
    n_samples=np.zeros((n_bandits, R, t_max))    

    # Realizations
    for r in np.arange(R):
        
        # Execute each bandit
        for n in np.arange(n_bandits):
            bandits[n].execute(t_max)
            
            if isinstance(bandits[n], OptimalBandit):
                returns[n,r,bandits[n].actions,:]=bandits[n].returns
                returns_expected[n,r,:,:]=bandits[n].returns_expected*np.ones((1,t_max))
                actions[n,r,bandits[n].actions,:]=np.ones(t_max)
                predictive[n,r,bandits[n].actions,:]=np.ones(t_max)
            elif isinstance(bandits[n], ProbabilisticBandit):
                returns[n,r,:,:]=bandits[n].returns
                actions[n,r,:,:]=bandits[n].actions
                predictive[n,r,:,:]=bandits[n].returns_expected*np.ones((1,t_max))
            elif isinstance(bandits[n], BayesianBandit):
                returns[n,r,:,:]=bandits[n].returns
                returns_expected[n,r,:,:]=bandits[n].returns_expected
                actions[n,r,:,:]=bandits[n].actions
                predictive[n,r,:,:]=bandits[n].actions_predictive_density
            elif isinstance(bandits[n], BayesianBanditSampling):
                returns[n,r,:,:]=bandits[n].returns
                returns_expected[n,r,:,:]=bandits[n].returns_expected
                actions[n,r,:,:]=bandits[n].actions
                predictive[n,r,:,:]=bandits[n].actions_predictive_density['mean']
                n_samples[n,r,:]=bandits[n].n_samples
            else:
                raise ValueError('Invalid bandit number {} class type={}'.format(n, bandits[n]))            
        
    # Return per realization matrices
    return (returns, returns_expected, actions, predictive, n_samples)

################################
# Bandit plotting functions
################################

# Bandit plotting function: returns 
def bandits_plot_returns(returns_expected, bandit_returns, colors, labels, t_plot=None, plot_std=True, plot_save=None):
    """ Plot returns for a set of bandits
        
        Args:
            returns_expected: true expected returns
            bandit_returns: bandits' returns
        Rets:
    """
    # Dimensionalities
    n_bandits, R, K, t_max = bandit_returns.shape

    if t_plot is None:
        t_plot=t_max
        
    # Returns over time
    plt.figure()
    plt.plot(np.arange(t_plot), returns_expected.max()*np.ones(t_plot), 'k', label='Expected')
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_plot), bandit_returns[n,:,:,0:t_plot].sum(axis=1).mean(axis=0), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit_returns[n,:,:,0:t_plot].sum(axis=1).mean(axis=0)-bandit_returns[n,:,:,0:t_plot].sum(axis=1).std(axis=0), bandit_returns[n,:,:,0:t_plot].sum(axis=1).mean(axis=0)+bandit_returns[n,:,:,0:t_plot].sum(axis=1).std(axis=0),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$y_t$')
    plt.title('Returns over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/returns.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    # Cumulative returns over time
    plt.figure()
    plt.plot(np.arange(t_plot), returns_expected.max()*np.arange(t_plot), 'k', label='Expected')
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_plot), bandit_returns[n,:,:,0:t_plot].sum(axis=1).cumsum(axis=1).mean(axis=0), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit_returns[n,:,:,0:t_plot].sum(axis=1).cumsum(axis=1).mean(axis=0)-bandit_returns[n,:,:,0:t_plot].sum(axis=1).cumsum(axis=1).std(axis=0), bandit_returns[n,:,:,0:t_plot].sum(axis=1).cumsum(axis=1).mean(axis=0)+bandit_returns[n,:,:,0:t_plot].sum(axis=1).cumsum(axis=1).std(axis=0),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$\sum_{t=0}^Ty_t$')
    plt.title('Cumulative returns over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/returns_cumulative.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Bandit plotting function: returns expected
def bandits_plot_returns_expected(returns_expected, bandit_returns_expected, colors, labels, t_plot=None, plot_std=True, plot_save=None):
    """ Plot returns for a set of bandits
        
        Args:
            returns_expected: true expected returns
            bandit_returns_expected: bandits' expected returns
        Rets:
    """
    # Dimensionalities
    n_bandits, R, K, t_max = bandit_returns_expected.shape

    if t_plot is None:
        t_plot=t_max
    
    # Expected returns per arm, over time
    for k in np.arange(0,K):
        plt.figure()
        plt.plot(np.arange(t_plot), returns_expected[k]*np.ones(t_plot), 'k', label='Expected')
        for n in np.arange(n_bandits):
            plt.plot(np.arange(t_plot), bandit_returns_expected[n,:,k,0:t_plot].mean(axis=0), colors[n], label=labels[n]+' actions')
            if plot_std:
                plt.fill_between(np.arange(t_plot), bandit_returns_expected[n,:,k,0:t_plot].mean(axis=0)-bandit_returns_expected[n,:,k,0:t_plot].std(axis=0), bandit_returns_expected[n,:,k,0:t_plot].mean(axis=0)+bandit_returns_expected[n,:,k,0:t_plot].std(axis=0),alpha=0.5, facecolor=colors[n])
        plt.ylabel(r'$E\{\mu_{a,t}\}$')
        plt.xlabel('t')
        plt.title('Expected returns over time for arm {}'.format(k))
        plt.xlim([0, t_plot-1])
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(plot_save+'/returns_expected_{}.pdf'.format(k), format='pdf', bbox_inches='tight')
            plt.close()

# Bandit plotting function: regrets 
def bandits_plot_regret(returns_expected, bandit_returns, colors, labels, t_plot=None, plot_std=True, plot_save=None):
    """ Plot regrets for a set of bandits
        
        Args:
            returns_expected: true expected returns
            bandit_returns: bandits' returns
        Rets:
    """
    # Dimensionalities
    n_bandits, R, K, t_max = bandit_returns.shape
    
    # Regret over time
    plt.figure()
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_plot), returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1).mean(axis=0), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), (returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1).mean(axis=0))-bandit_returns[n,:,:,0:t_plot].sum(axis=1).std(axis=0), (returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1).mean(axis=0))+bandit_returns[n,:,:,0:t_plot].sum(axis=1).std(axis=0),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$l_t=y_t^*-y_t$')
    plt.title('Regret over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/regret.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    # Cumulative regret over time
    plt.figure()
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_plot), (returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1)).cumsum(axis=1).mean(axis=0), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), (returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1)).cumsum(axis=1).mean(axis=0)-(returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1)).cumsum(axis=1).std(axis=0), (returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1)).cumsum(axis=1).mean(axis=0)+(returns_expected.max()-bandit_returns[n,:,:,0:t_plot].sum(axis=1)).cumsum(axis=1).std(axis=0),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$L_t=\sum_{t=0}^T y_t^*-y_t$')
    plt.title('Cumulative regret over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/regret_cumulative.pdf', format='pdf', bbox_inches='tight')
        plt.close()


# Bandit plotting function: action predictive density
def bandits_plot_action_density(bandit_predictive, colors, labels, t_plot=None, plot_std=True, plot_save=None):
    """ Plot the computed predictive action density for a set of bandits
        
        Args:
            bandit_predictive: bandits' action predictive density
        Rets:
    """
    # Dimensionalities
    n_bandits, R, K, t_max = bandit_predictive.shape
    
    # Action predictive density probabilities over time
    for k in np.arange(0,K):
        plt.figure()
        for n in np.arange(n_bandits):
            plt.plot(np.arange(t_plot), bandit_predictive[n,:,k,0:t_plot].mean(axis=0), colors[n], label=labels[n])
            if plot_std:
                plt.fill_between(np.arange(t_plot), bandit_predictive[n,:,k,0:t_plot].mean(axis=0)-bandit_predictive[n,:,k,0:t_plot].std(axis=0), bandit_predictive[n,:,k,0:t_plot].mean(axis=0)+bandit_predictive[n,:,k,0:t_plot].std(axis=0),alpha=0.5, facecolor=colors[n])
        plt.ylabel(r'$f(a_{t+1}=a|a_{1:t}, y_{1:t})$')
        plt.xlabel('t')
        plt.title('Averaged Action Predictive density probabilities for arm {}'.format(k))
        plt.xlim([0, t_plot-1])
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(plot_save+'/action_density_{}.pdf'.format(k), format='pdf', bbox_inches='tight')
            plt.close()

# Bandit plotting function: correct action predictive density percentages
def bandits_plot_action_density_correct(bandit_predictive, colors, labels, t_plot=None, plot_std=True, plot_save=None):
    """ Plot the probability of the predictive action density being correct for a set of bandits
        
        Args:
            bandit_predictive: bandits' action predictive density
        Rets:
    """
    
    # Dimensionalities
    n_bandits, R, K, t_max = bandit_predictive.shape

    plt.figure()
    for n in np.arange(bandit_predictive.shape[0]):
        plt.plot(np.arange(t_plot), (bandit_predictive[n,:,:,:].argmax(axis=1)==(K-1)).astype(int).mean(axis=0), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), (bandit_predictive[n,:,:,:].argmax(axis=1)==(K-1)).astype(int).mean(axis=0)-(bandit_predictive[n,:,:,:].argmax(axis=1)==(K-1)).astype(int).std(axis=0), (bandit_predictive[n,:,:,:].argmax(axis=1)==(K-1)).astype(int).mean(axis=0)+(bandit_predictive[n,:,:,:].argmax(axis=1)==(K-1)).astype(int).std(axis=0),alpha=0.5, facecolor=bandits_colors[n])
    plt.xlabel('t')
    plt.ylabel('% Correct')
    plt.title('Correct action predictive density percentage')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/action_density_correct.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Bandit plotting function: actions
def bandits_plot_actions(bandit_actions, colors, labels, t_plot=None, plot_std=True, plot_save=None):
    """ Plot the actions selected for a set of bandits
        
        Args:
            bandit_actions: bandits' actions
        Rets:
    """
    # Dimensionalities
    n_bandits, R, K, t_max = bandit_actions.shape
       
    # Action probabilities over time
    for k in np.arange(0,K):
        plt.figure()
        for n in np.arange(n_bandits):
            plt.plot(np.arange(t_plot), bandit_actions[n,:,k,0:t_plot].mean(axis=0), colors[n], label=labels[n]+' actions')
            if plot_std:
                plt.fill_between(np.arange(t_plot), bandit_actions[n,:,k,0:t_plot].mean(axis=0)-bandit_actions[n,:,k,0:t_plot].std(axis=0), bandit_actions[n,:,k,0:t_plot].mean(axis=0)+bandit_actions[n,:,k,0:t_plot].std(axis=0),alpha=0.5, facecolor=colors[n])
        plt.ylabel(r'$f(a_{t+1}=a|a_{1:t}, y_{1:t})$')
        plt.xlabel('t')
        plt.title('Averaged Action probabilities for arm {}'.format(k))
        plt.xlim([0, t_plot-1])
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(plot_save+'/actions_{}.pdf'.format(k), format='pdf', bbox_inches='tight')
            plt.close()

# Bandit plotting function: correct actions
def bandits_plot_actions_correct(returns_expected,bandit_actions, colors, labels, t_plot=None, plot_std=True, plot_save=None):
    """ Plot the probability of selecting correct actions for a set of bandits
        
        Args:
            returns_expected: true expected returns        
            bandit_actions: bandits' actions
        Rets:
    """
    
    # Dimensionalities
    n_bandits, R, K, t_max = bandit_actions.shape
    
	# Correct arm selection probability
    plt.figure()
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_plot), bandit_actions[n,:,returns_expected.argmax(),0:t_plot].mean(axis=0), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit_actions[n,:,returns_expected.argmax(),0:t_plot].mean(axis=0)-bandit_actions[n,:,returns_expected.argmax(),0:t_plot].std(axis=0), bandit_actions[n,:,returns_expected.argmax(),0:t_plot].mean(axis=0)+bandit_actions[n,:,returns_expected.argmax(),0:t_plot].std(axis=0),alpha=0.5, facecolor=colors[n])
    plt.ylabel(r'$f(a_{t+1}=a^*|a_{1:t}, y_{1:t})$')
    plt.xlabel('t')
    plt.title('Averaged Correct Action probabilities')
    plt.axis([0, t_plot-1, 0, 1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/correct_actions.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Bandit plotting function: n_samples 
def bandits_plot_n_samples(n_samples, colors, labels, t_plot=None, plot_std=True, plot_save=None):
    """ Plot number of samples for a set of sampling bandits
        
        Args:
            n_samples: number of samples used over time in SAMPLING based bandits
        Rets:
    """
    # Dimensionalities
    n_bandits, R, t_max = n_samples.shape
    
    # Regret over time
    plt.figure()
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_plot), n_samples[n,:,0:t_plot].mean(axis=0), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), n_samples[n,:,0:t_plot].mean(axis=0)-n_samples[n,:,0:t_plot].std(axis=0), n_samples[n,:,0:t_plot].mean(axis=0)+n_samples[n,:,0:t_plot].std(axis=0),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$M_t$')
    plt.title('n_samples over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/n_samples.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Bandit plotting function: all
def bandits_plot_all(returns_expected, bandit_returns, bandit_actions, bandit_predictive, colors, labels, t_plot=None, plot_std=True, plot_save=None):
    """ Plot all results for a set of bandits
        
        Args:
            returns_expected: true expected returns
            bandit_returns: bandits' returns
            bandit_actions: bandits' actions
            bandit_predictive: bandits' action predictive density
        Rets:
    """
        
    # Returns over time
    bandits_plot_returns(returns_expected, bandit_returns, colors, labels, t_plot, plot_std, plot_save)

    # Regret over time
    bandits_plot_regret(returns_expected, bandit_returns, colors, labels, t_plot, plot_std, plot_save)

    # Action predictive density probabilities over time
    bandits_plot_action_density(bandit_predictive, colors, labels, t_plot, plot_std, plot_save)
   
    # Actions over time
    bandits_plot_actions(bandit_actions, colors, labels, t_plot, plot_std, plot_save)

    # Correct actions over time
    bandits_plot_actions_correct(returns_expected, bandit_actions, colors, labels, t_plot, plot_std, plot_save)
    	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
