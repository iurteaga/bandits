#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import sys, os
import argparse
from itertools import *
import pdb
from matplotlib import colors

# Add path and import Bayesian Bandits
sys.path.append('../src')
from BayesianBanditsSampling import *
from plot_bandits import *

# Main code
def main(K, t_max, R, theta_min, theta_max, theta_diff):
    print('Bayesian {}-armed bayesian bandit for {} time-instants and {} realizations'.format(K, t_max, R))

    # Directory configuration
    dir_string='../results/{}/K={}/t_max={}/R={}'.format(os.path.basename(__file__).split('.')[0], K, t_max, R)
    os.makedirs(dir_string, exist_ok=True)
    
    # Bandit configuration
    # for theta in combinations_with_replacement(np.arange(0.1,1.0,theta_diff),K):
    for theta in combinations(np.arange(theta_min,theta_max,theta_diff),K):
        # For each theta
        theta=np.array([*theta]).reshape(K,1)
        print('theta={}'.format(np.array_str(theta)))
        os.makedirs(dir_string+'/theta={}'.format(theta[:,0]), exist_ok=True)
        
        # Reward function and prior
        reward_function={'dist':stats.bernoulli, 'args':(theta,), 'kwargs':{}}
        returns_expected=theta
        reward_prior={'dist': stats.beta, 'alpha': np.ones((K,1)), 'beta': np.ones((K,1))}
        
        # Bandits to evaluate as a list
      
        # Thompson sampling bandit
        sampling={'type':'static', 'n_samples':1}
        ts_bandit=BayesianBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, 1)
        bandits=[ts_bandit]
        bandits_labels=['TS']
                
        # Monte Carlo Bayesian Bandits Sampling
        M_samples=np.array([100, 500, 1000])
#        for M in M_samples:
#            # Linear sampling 
#            sampling={'type':'linear', 'n_0':1, 'n':1}
#            bandits.append(BayesianBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
#            bandits_labels.append('MC n=linear1,M={}'.format(M))
        for M in M_samples:
            # Log sampling 
            sampling={'type':'invVar', 'n_max':1000}
            bandits.append(BayesianBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
            bandits_labels.append('MC n=invVar,M={}'.format(M))
        for M in M_samples:
            # Sqrt sampling 
            sampling={'type':'invPFA', 'n_max':1000}
            bandits.append(BayesianBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
            bandits_labels.append('MC n=invPFA,M={}'.format(M))            
        
        # Bandits colors
        bandits_colors=['black', 'blue', 'cyan', 'skyblue', 'green', 'lime', 'palegreen', 'red', 'orange', 'yellow', 'purple', 'fucsia', 'pink']
        bandits_colors=[colors.cnames['black'], colors.cnames['blue'], colors.cnames['cyan'], colors.cnames['skyblue'], colors.cnames['green'], colors.cnames['lime'], colors.cnames['palegreen'], colors.cnames['red'], colors.cnames['orange'], colors.cnames['yellow'], colors.cnames['purple'], colors.cnames['fuchsia'], colors.cnames['pink']]
              
        # Execute each bandit
        for (n,bandit) in enumerate(bandits):
            bandit.execute_realizations(R, t_max)
        
        ############################### PLOTTING  ############################### 
        # Plotting overall
        dir_plots=dir_string+'/theta={}'.format(theta[:,0])+'/plots'
        os.makedirs(dir_plots, exist_ok=True)

        # Plotting time: all
        t_plot=t_max
        
        # Plot regret
        plot_std=False
        bandits_plot_regret(returns_expected, bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
        
        # Plot returns expected
        plot_std=True
        bandits_plot_returns_expected(returns_expected, bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)

        # Plot action predictive density
        plot_std=True
        bandits_plot_action_density(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
        
        # Plot correct action density
        plot_std=False
        bandits_plot_action_density_correct(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
        
        # Plot actions
        plot_std=True
        bandits_plot_actions(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)

        # Plot correct actions
        plot_std=False
        bandits_plot_actions_correct(returns_expected, bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
        
        # Plot sample size
        plot_std=False
        bandits_plot_n_samples(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
        ###############          

            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Evaluate Bayesian bandits.')
    parser.add_argument('-K', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-theta_min', type=float, default=0, help='Minimum theta')
    parser.add_argument('-theta_max', type=float, default=1, help='Maximum theta')
    parser.add_argument('-theta_diff', type=float, default=0.5, help='Differences for theta')

    # Get arguments
    args = parser.parse_args()
    
    # Call main function
    main(args.K, args.t_max, args.R, args.theta_min, args.theta_max, args.theta_diff)
