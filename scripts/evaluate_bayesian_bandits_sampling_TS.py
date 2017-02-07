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
def main(K, t_max, R, theta):
    print('Bayesian {}-armed bayesian bandit for {} time-instants and {} realizations'.format(K, t_max, R))

    # Directory configuration
    dir_string='../results/{}/K={}/t_max={}/R={}'.format(os.path.basename(__file__).split('.')[0], K, t_max, R)
    os.makedirs(dir_string, exist_ok=True)
    
    # Bandit configuration
    theta=np.array(theta).reshape(K,1)
    print('theta={}'.format(np.array_str(theta)))
    os.makedirs(dir_string+'/theta={}'.format(theta[:,0]), exist_ok=True)
    
    # Reward function and prior
    reward_function={'dist':stats.bernoulli, 'args':(theta,), 'kwargs':{}}
    returns_expected=theta
    reward_prior={'dist': stats.beta, 'alpha': np.ones((K,1)), 'beta': np.ones((K,1))}
    
    # Bandits to evaluate as a list
    bandits=[]
    bandits_labels=[]
    
    # Monte Carlo Bayesian Bandits Sampling
    M_samples=np.array([1, 1000])
    for M in M_samples:
        # Monte Carlo sampling, n=1
        sampling={'type':'static', 'n_samples':1}
        bandits.append(BayesianBanditSamplingMonteCarlo(K, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=1,M={}'.format(M))
            
    # Bandits colors
    bandits_colors=['black', 'red', 'cyan', 'blue', 'lime', 'green', 'orange', 'fuchsia', 'purple']
    bandits_colors=[colors.cnames['black'], colors.cnames['red'], colors.cnames['cyan'], colors.cnames['blue'], colors.cnames['lime'], colors.cnames['green'], colors.cnames['orange'], colors.cnames['fuchsia'], colors.cnames['purple']]
          
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
    plot_std=True
    bandits_plot_regret(returns_expected, bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
        
    # Plot returns expected
    plot_std=True
    bandits_plot_returns_expected(returns_expected, bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)

    # Plot action predictive density
    plot_std=True
    bandits_plot_action_density(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    
    # Plot actions
    plot_std=True
    bandits_plot_actions(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)

    # Plot correct actions
    plot_std=False
    bandits_plot_actions_correct(returns_expected, bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_actions_correct(returns_expected, bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    ###############          

            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Evaluate Bayesian bandits.')
    parser.add_argument('-K', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-theta', nargs='+', type=float, default=0, help='Theta')

    # Get arguments
    args = parser.parse_args()
    
    # Make sure K and theta size match
    assert len(args.theta)==args.K, 'Size of theta={} does not match number of arms K={}'.format(args.theta, args.K)
    # Call main function
    main(args.K, args.t_max, args.R, args.theta)
