#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import pickle
import sys, os
import argparse
from itertools import *
import pdb
from matplotlib import colors

# Add path and import Bayesian Bandits
sys.path.append('../src')
from BayesianContextualBanditsSampling import *
from plot_bandits import *

# Main code
def main(M, K, t_max, R):
    print('Bayesian {}-armed contextual bandit with {} samples for {} time-instants and {} realizations'.format(K, M, t_max, R))

    # Directory configuration
    dir_string='../results/{}/K={}/t_max={}/R={}/M={}'.format(os.path.basename(__file__).split('.')[0], K, t_max, R, M)
    os.makedirs(dir_string, exist_ok=True)
    
    # Contextual Bandit configuration
    # Context
    d_context=2
    # Static context
    context=np.ones((d_context, 1))
    
    # Theta
    theta=np.ones((K,d_context))
    theta[0]=-1
    # Mean is linear combination
    returns_expected=np.dot(theta, context)
    # Sigma
    returns_sigma=1
    
    # Reward function
    reward_function={'type':'linear_gaussian', 'dist':stats.norm, 'args':(), 'kwargs':{'loc':returns_expected,'scale':returns_sigma}}
       
    # Reward prior
    Sigmas=np.zeros((K, d_context, d_context))
    for k in np.arange(K):
        Sigmas[k,:,:]=np.eye(d_context)

    reward_prior={'dist': 'NIG', 'alpha': np.ones((K,1)), 'beta': np.ones((K,1)), 'theta': np.ones((K,d_context)), 'Sigma':Sigmas}
    
    # Bandits to evaluate as a list
  
    # Thompson sampling bandit
    sampling={'type':'static', 'n_samples':1}
    ts_bandit=BayesianContextualBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, 1)
    bandits=[ts_bandit]
    bandits_labels=['TS']
           
    # Monte Carlo Bayesian Bandits Sampling
    M_samples=np.array([1000])
    
    # Bayesian sampling bandit
    sampling={'type':'static', 'n_samples':1}
    bandits.append(BayesianContextualBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M_samples[0]))
    bandits_labels.append('TS M={}'.format(M_samples[0]))

    # Monte Carlo sampling, n invPFA_tGaussian
    for M in M_samples:
        sampling={'type':'invPFA_tGaussian', 'n_max':100}
        bandits.append(BayesianContextualBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=(1/Pfa) tGaussian, M={}'.format(M))    
    # Monte Carlo sampling, n loginvPFA_tGaussian
    for M in M_samples:
        sampling={'type':'loginvPFA_tGaussian', 'n_max':100}
        bandits.append(BayesianContextualBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=ln(1/Pfa) tGaussian, M={}'.format(M))
    # Monte Carlo sampling, n log10invPFA_tGaussian
    for M in M_samples:
        sampling={'type':'log10invPFA_tGaussian', 'n_max':100}
        bandits.append(BayesianContextualBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=log10(1/Pfa) tGaussian, M={}'.format(M))

    # Monte Carlo sampling, n invPFA_Markov
    for M in M_samples:
        sampling={'type':'invPFA_Markov', 'n_max':100}
        bandits.append(BayesianContextualBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=(1/Pfa) Markov, M={}'.format(M))
    # Monte Carlo sampling, n loginvPFA_Markov
    for M in M_samples:
        sampling={'type':'loginvPFA_Markov', 'n_max':100}
        bandits.append(BayesianContextualBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=ln(1/Pfa) Markov, M={}'.format(M))
    # Monte Carlo sampling, n log10invPFA_Markov
    for M in M_samples:
        sampling={'type':'log10invPFA_Markov', 'n_max':100}
        bandits.append(BayesianContextualBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=log10(1/Pfa) Markov, M={}'.format(M))

    # Monte Carlo sampling, n invPFA_Chebyshev
    for M in M_samples:
        sampling={'type':'invPFA_Chebyshev', 'n_max':100}
        bandits.append(BayesianContextualBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=(1/Pfa) Chebyshev, M={}'.format(M))        
    # Monte Carlo sampling, n loginvPFA_Chebyshev
    for M in M_samples:
        sampling={'type':'loginvPFA_Chebyshev', 'n_max':100}
        bandits.append(BayesianContextualBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=ln(1/Pfa) Chebyshev, M={}'.format(M))
    # Monte Carlo sampling, n log10invPFA_Chebyshev
    for M in M_samples:
        sampling={'type':'log10invPFA_Chebyshev', 'n_max':100}
        bandits.append(BayesianContextualBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=log10(1/Pfa) Chebyshev, M={}'.format(M))
                
    # Bandits colors
    bandits_colors=['black', 'pink', 'skyblue', 'cyan', 'blue', 'palegreen', 'lime', 'green', 'yellow', 'orange', 'red', 'purple', 'fucsia']
    bandits_colors=[colors.cnames['black'],  colors.cnames['pink'], colors.cnames['skyblue'], colors.cnames['cyan'], colors.cnames['blue'], colors.cnames['palegreen'], colors.cnames['lime'], colors.cnames['green'], colors.cnames['yellow'], colors.cnames['orange'], colors.cnames['red'], colors.cnames['purple'], colors.cnames['fuchsia']]
    
    # Execute each bandit
    for (n,bandit) in enumerate(bandits):
        # With repeated static context
        bandit.execute_realizations(R, np.tile(context, t_max), t_max)
    
    ############################### PLOTTING  ############################### 
    # Plotting overall
    dir_plots=dir_string+'/plots'
    os.makedirs(dir_plots, exist_ok=True)

    # Plotting time: all
    t_plot=t_max
    
    # Plot regret
    plot_std=False
    bandits_plot_regret(returns_expected, bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_regret(returns_expected, bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    
    # Plot returns expected
    plot_std=True
    bandits_plot_returns_expected(returns_expected, bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)

    # Plot action predictive density
    plot_std=True
    bandits_plot_action_density(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    
    # Plot actions
    plot_std=False
    bandits_plot_actions(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_actions(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)

    # Plot correct actions
    plot_std=False
    bandits_plot_actions_correct(returns_expected, bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_actions_correct(returns_expected, bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    
    # Plot sample size
    plot_std=False
    bandits_plot_n_samples(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_n_samples(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    ###############          

    # Save bandits info
    with open(dir_string+'/bandits.pickle', 'wb') as f:
        pickle.dump(bandits, f)
    with open(dir_string+'/bandits_labels.pickle', 'wb') as f:
        pickle.dump(bandits_labels, f)
            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Evaluate Bayesian contextual bandits.')
    parser.add_argument('-M', type=int, default=1000, help='Number of samples for the MC integration')
    parser.add_argument('-K', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')

    # Get arguments
    args = parser.parse_args()
    
    # Call main function
    main(args.M, args.K, args.t_max, args.R)
