#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from Bandits import * 
from BayesianBandits import * 
from BayesianBanditsSampling import * 
from BayesianContextualBandits import * 

################################
# Bandit plotting functions
################################

# Bandit plotting function: returns 
def bandits_plot_returns(returns_expected, bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot returns for a set of bandits
        
        Args:
            returns_expected: true expected returns
            bandits: bandit list
        Rets:
    """
       
    # Returns over time
    plt.figure()
    plt.plot(np.arange(t_plot), returns_expected.max()*np.ones(t_plot), 'k', label='Expected')
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot), bandit.returns_R['mean'][0,0:t_plot], colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit.returns_R['mean'][0,0:t_plot]-np.sqrt(bandit.returns_R['var'][0,0:t_plot]), bandit.returns_R['mean'][0,0:t_plot]+np.sqrt(bandit.returns_R['var'][0,0:t_plot]),alpha=0.5, facecolor=colors[n])
    plt.xlabel('t')
    plt.ylabel(r'$y_t$')
    plt.title('Returns over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/returns_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    # Cumulative returns over time
    plt.figure()
    plt.plot(np.arange(t_plot), returns_expected.max()*np.arange(t_plot), 'k', label='Expected')
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot),bandit.returns_R['mean'][0,0:t_plot].cumsum(axis=1), colors[n], label=labels[n])
    plt.xlabel('t')
    plt.ylabel(r'$\sum_{t=0}^Ty_t$')
    plt.title('Cumulative returns over time')
    plt.xlim([0, t_plot-1])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig(plot_save+'/returns_cumulative_std'+str(plot_std)+'.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# Bandit plotting function: returns expected
def bandits_plot_returns_expected(returns_expected, bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot returns for a set of bandits
        
        Args:
            returns_expected: true expected returns
            bandits: bandit list
        Rets:
    """
    
    # Expected returns per arm, over time
    for a in np.arange(0,bandits[0].A):
        plt.figure()
        plt.plot(np.arange(t_plot), returns_expected[a]*np.ones(t_plot), 'k', label='Expected')
        for (n,bandit) in enumerate(bandits):
            plt.plot(np.arange(t_plot), bandit.returns_expected_R['mean'][a,0:t_plot], colors[n], label=labels[n]+' actions')
            if plot_std:
                plt.fill_between(np.arange(t_plot), bandit.returns_expected_R['mean'][a,0:t_plot]-np.sqrt(bandit.returns_expected_R['var'][a,0:t_plot]), bandit.returns_expected_R['mean'][a,0:t_plot]+np.sqrt(bandit.returns_expected_R['var'][a,0:t_plot]),alpha=0.5, facecolor=colors[n])
        plt.ylabel(r'$E\{\mu_{a,t}\}$')
        plt.xlabel('t')
        plt.title('Expected returns over time for arm {}'.format(a))
        plt.xlim([0, t_plot-1])
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        if plot_save is None: 
            plt.show()
        else:
            plt.savefig(plot_save+'/returns_expected_{}_std{}.pdf'.format(a,str(plot_std)), format='pdf', bbox_inches='tight')
            plt.close()

# Bandit plotting function: regrets 
def bandits_plot_regret(returns_expected, bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot regrets for a set of bandits
        
        Args:
            returns_expected: true expected returns
            bandits: bandit list
        Rets:
    """
    
    # Regret over time
    plt.figure()
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot), returns_expected.max()-bandit.returns_R['mean'][0,0:t_plot], colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), (returns_expected.max()-bandit.returns_R['mean'][0,0:t_plot])-np.sqrt(bandit.returns_R['var'][0,0:t_plot]), (returns_expected.max()-bandit.returns_R['mean'][0,0:t_plot])+np.sqrt(bandit.returns_R['var'][0,0:t_plot]),alpha=0.5, facecolor=colors[n])
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
        plt.plot(np.arange(t_plot), (returns_expected.max()-bandit.returns_R['mean'][0,0:t_plot]).cumsum(), colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), (returns_expected.max()-bandit.returns_R['mean'][0,0:t_plot]).cumsum()-np.sqrt(bandit.returns_R['var'][0,0:t_plot]), (returns_expected.max()-bandit.returns_R['mean'][0,0:t_plot]).cumsum()+np.sqrt(bandit.returns_R['var'][0,0:t_plot]),alpha=0.5, facecolor=colors[n])
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


# Bandit plotting function: action predictive density
def bandits_plot_action_density(bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot the computed predictive action density for a set of bandits
        
        Args:
            bandits: bandit list
        Rets:
    """
    
    # Action predictive density probabilities over time
    for a in np.arange(0,bandits[0].A):
        plt.figure()
        for (n,bandit) in enumerate(bandits):
            plt.plot(np.arange(t_plot), bandit.actions_predictive_density_R['mean'][a,0:t_plot], colors[n], label=labels[n])
            if plot_std:
                plt.fill_between(np.arange(t_plot), bandit.actions_predictive_density_R['mean'][a,0:t_plot]-np.sqrt(bandit.actions_predictive_density_R['var'][a,0:t_plot]), bandit.actions_predictive_density_R['mean'][a,0:t_plot]+np.sqrt(bandit.actions_predictive_density_R['var'][a,0:t_plot]),alpha=0.5, facecolor=colors[n])
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
        plt.plot(np.arange(t_plot), (bandit.actions_predictive_density_R['mean'].argmax(axis=0)==(bandit.A-1)).astype(int), colors[n], label=labels[n])
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
def bandits_plot_actions_correct(returns_expected,bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot the probability of selecting correct actions for a set of bandits
        
        Args:
            returns_expected: true expected returns        
            bandits: bandit list
        Rets:
    """
   
	# Correct arm selection probability
    plt.figure()
    for (n,bandit) in enumerate(bandits):
        plt.plot(np.arange(t_plot), bandit.actions_R['mean'][returns_expected.argmax(),0:t_plot], colors[n], label=labels[n])
        if plot_std:
            plt.fill_between(np.arange(t_plot), bandit.actions_R['mean'][returns_expected.argmax(),0:t_plot]-np.sqrt(bandit.actions_R['var'][returns_expected.argmax(),0:t_plot]), bandit.actions_R['mean'][returns_expected.argmax(),0:t_plot]+np.sqrt(bandit.actions_R['var'][returns_expected.argmax(),0:t_plot]),alpha=0.5, facecolor=colors[n])
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

'''import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
ax1.plot(self.actions_predictive_density['mean'][0,:],'b')
ax1.fill_between(np.arange(t_max),self.actions_predictive_density['mean'][0,:]-np.sqrt(self.actions_predictive_density['var'][0,:]), self.actions_predictive_density['mean'][0,:]+np.sqrt(self.actions_predictive_density['var'][0,:]),alpha=0.5, facecolor='b')
ax1.plot(self.actions_predictive_density['mean'][1,:],'g')
ax1.fill_between(np.arange(t_max),self.actions_predictive_density['mean'][1,:]-np.sqrt(self.actions_predictive_density['var'][1,:]), self.actions_predictive_density['mean'][1,:]+np.sqrt(self.actions_predictive_density['var'][1,:]),alpha=0.5, facecolor='g')
ax1.plot(self.actions_predictive_density['mean'][2,:],'r')
ax1.fill_between(np.arange(t_max),self.actions_predictive_density['mean'][2,:]-np.sqrt(self.actions_predictive_density['var'][2,:]), self.actions_predictive_density['mean'][2,:]+np.sqrt(self.actions_predictive_density['var'][2,:]),alpha=0.5, facecolor='r')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('p(a=a*)', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(self.n_samples, 'k')
ax2.set_ylabel('n_samples', color='k')
for tl in ax2.get_yticklabels():
    tl.set_color('k')

plt.show()'''

# Bandit plotting function: all
def bandits_plot_all(returns_expected, bandits, colors, labels, t_plot, plot_std=True, plot_save=None):
    """ Plot all results for a set of bandits
        
        Args:
            bandits: bandit list
        Rets:
    """
        
    # Returns over time
    bandits_plot_returns(returns_expected, bandits, colors, labels, t_plot, plot_std, plot_save)

    # Regret over time
    bandits_plot_regret(returns_expected, bandits, colors, labels, t_plot, plot_std, plot_save)

    # Action predictive density probabilities over time
    bandits_plot_action_density(bandits, colors, labels, t_plot, plot_std, plot_save)
    
    # Correct action predictive density percentages
    bandits_plot_action_density_correct(bandits, colors, labels, t_plot, plot_std, plot_save)
   
    # Actions over time
    bandits_plot_actions(bandits, colors, labels, t_plot, plot_std, plot_save)

    # Correct actions over time
    bandits_plot_actions_correct(returns_expected, bandits, colors, labels, t_plot, plot_std, plot_save)
    
    # Number of samples over time
    bandits_plot_n_samples(bandits, colors, labels, t_plot, plot_std, plot_save)
    	
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    main()
