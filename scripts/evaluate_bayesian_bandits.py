#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import sys, os
import argparse
from itertools import *

# Add path and import Bayesian Bandits
sys.path.append('../src')
from BayesianBandits import *
from Bandit_functions import *

# Main code
def main(K, t_max, R, theta_diff, theta_factor):
    print('ayesian {}-armed bayesian bandit for {} time-instants and {} realizations'.format(K, t_max, R))

    # Directory configuration
    dir_string='../results/{}/K={}/t_max={}/R={}'.format(os.path.basename(__file__).split('.')[0], K, t_max, R)
    os.makedirs(dir_string, exist_ok=True)
    
    # Bandit configuration
    # Bernoulli reward function
    if theta_diff == 0:
        # Random theta
        theta=stats.uniform.rvs(size=(K,1))/theta_factor
    else:
        for theta in combinations_with_replacement(np.arange(0.1,1.0,theta_diff),K):
            # For each theta
            theta=np.array([*theta]).reshape(K,1)
            print('theta_{}'.format(np.array_str(theta)))
            os.makedirs(dir_string+'/theta={}'.format(theta[:,0]), exist_ok=True)
            
            # Reward function and prior
            reward_function={'dist':stats.bernoulli, 'args':(theta,), 'kwargs':{}}
            returns_expected=theta
            reward_prior={'dist': stats.beta, 'alpha': np.ones((K,1)), 'beta': np.ones((K,1))}
            
            # Bandits to evaluate as a list
            # Optimal bandit
            optimal_bandit=OptimalBandit(K, reward_function)
            bandits=[optimal_bandit]
            bandits_labels=['Optimal']
            # Monte Carlo Bayesian Bandits
            M_samples=np.array([1, 100])
            for n in np.arange(M_samples.size):
                bandits.append(BayesianBanditMonteCarlo(K, reward_function, reward_prior, M_samples[n]))
                bandits_labels.append('Monte Carlo M={}'.format(M_samples[n]))
                
            # Bandits colors
            bandits_colors=['b', 'g', 'r']
            
            # Execution
            bandits_returns, bandits_actions, bandits_predictive=execute_bandits(K, bandits, R, t_max)
    
            # Plotting
            dir_plots=dir_string+'/theta={}'.format(theta[:,0])+'/plots'
            os.makedirs(dir_plots, exist_ok=True)

            # Plotting time: all
            t_plot=t_max
            
            # Plot regret
            plot_std=False
            bandits_plot_regret(returns_expected, bandits_returns, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)

            # Plot action density
            plot_std=True
            bandits_plot_action_density(bandits_predictive, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)

            # Plot correct actions
            plot_std=False
            bandits_plot_actions_correct(returns_expected, bandits_actions, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
            
            # Data saving
            np.save(dir_string+'/theta={}'.format(theta[:,0])+'/bandits', bandits)
            np.save(dir_string+'/theta={}'.format(theta[:,0])+'/bandits_returns', bandits_returns)
            np.save(dir_string+'/theta={}'.format(theta[:,0])+'/bandits_actions', bandits_actions)
            np.save(dir_string+'/theta={}'.format(theta[:,0])+'/bandits_predictive', bandits_predictive)
            
            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Evaluate Bayesian bandits.')
    parser.add_argument('-K', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-theta_diff', type=float, default=0, help='Differences for theta in each arm, or random theta if 0')
    parser.add_argument('--theta_factor', type=int, default=1, help='Factor for theta\'s')

    # Get arguments
    args = parser.parse_args()
    
    # Call main function
    main(args.K, args.t_max, args.R, args.theta_diff, args.theta_factor)
