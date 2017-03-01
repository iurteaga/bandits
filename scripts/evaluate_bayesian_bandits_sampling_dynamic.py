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
from BayesianBanditsSampling import *
from plot_bandits import *

# Main code
def main(M, A, t_max, R, exec_type, theta):
    print('Bayesian {}-armed bayesian bandit with {} samples for {} time-instants and {} realizations'.format(A, M, t_max, R))

    # Directory configuration
    dir_string='../results/{}/A={}/t_max={}/R={}/M={}'.format(os.path.basename(__file__).split('.')[0], A, t_max, R, M)
    os.makedirs(dir_string, exist_ok=True)
    
    # Bandit configuration
    theta=np.array(theta).reshape(A,1)
    print('theta={}'.format(np.array_str(theta)))
    os.makedirs(dir_string+'/theta={}'.format(theta[:,0]), exist_ok=True)
    
    # Reward function and prior
    reward_function={'dist':stats.bernoulli, 'args':(theta,), 'kwargs':{}}
    returns_expected=theta
    reward_prior={'dist': stats.beta, 'alpha': np.ones((A,1)), 'beta': np.ones((A,1))}
    
    # Bandits to evaluate as a list
  
    # Thompson sampling bandit
    sampling={'type':'static', 'n_samples':1}
    ts_bandit=BayesianBanditSamplingMonteCarlo(A, reward_function, reward_prior, sampling, 1)
    bandits=[ts_bandit]
    bandits_labels=['TS']
           
    # Monte Carlo Bayesian Bandits Sampling
    M_samples=np.array([100, 1000])
    M_samples=np.array([1000])
    
    '''
    alpha=1./10
    # Monte Carlo sampling, n invPnotOpt, n_rate=alpha
    for M in M_samples:
        sampling={'type':'invPnotOpt', 'n_rate':alpha, 'n_max':100}
        bandits.append(BayesianBanditSamplingMonteCarlo(A, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n={:.2f}/PnotOpt,M={}'.format(alpha,M))
    
    # Monte Carlo sampling, n log10invPnotOpt
    for M in M_samples:
        sampling={'type':'log10invPnotOpt', 'n_max':100}
        bandits.append(BayesianBanditSamplingMonteCarlo(A, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=log10(1/PnotOpt),M={}'.format(M))
    '''

    # Monte Carlo sampling, n invPFA_tGaussian
    for M in M_samples:
        sampling={'type':'invPFA_tGaussian', 'n_max':100}
        bandits.append(BayesianBanditSamplingMonteCarlo(A, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=(1/Pfa) tGaussian, M={}'.format(M))    
    # Monte Carlo sampling, n loginvPFA_tGaussian
    for M in M_samples:
        sampling={'type':'loginvPFA_tGaussian', 'n_max':100}
        bandits.append(BayesianBanditSamplingMonteCarlo(A, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=ln(1/Pfa) tGaussian, M={}'.format(M))
    # Monte Carlo sampling, n log10invPFA_tGaussian
    for M in M_samples:
        sampling={'type':'log10invPFA_tGaussian', 'n_max':100}
        bandits.append(BayesianBanditSamplingMonteCarlo(A, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=log10(1/Pfa) tGaussian, M={}'.format(M))

    # Monte Carlo sampling, n invPFA_Markov
    for M in M_samples:
        sampling={'type':'invPFA_Markov', 'n_max':100}
        bandits.append(BayesianBanditSamplingMonteCarlo(A, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=(1/Pfa) Markov, M={}'.format(M))
    # Monte Carlo sampling, n loginvPFA_Markov
    for M in M_samples:
        sampling={'type':'loginvPFA_Markov', 'n_max':100}
        bandits.append(BayesianBanditSamplingMonteCarlo(A, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=ln(1/Pfa) Markov, M={}'.format(M))
    # Monte Carlo sampling, n log10invPFA_Markov
    for M in M_samples:
        sampling={'type':'log10invPFA_Markov', 'n_max':100}
        bandits.append(BayesianBanditSamplingMonteCarlo(A, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=log10(1/Pfa) Markov, M={}'.format(M))

    # Monte Carlo sampling, n invPFA_Chebyshev
    for M in M_samples:
        sampling={'type':'invPFA_Chebyshev', 'n_max':100}
        bandits.append(BayesianBanditSamplingMonteCarlo(A, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=(1/Pfa) Chebyshev, M={}'.format(M))        
    # Monte Carlo sampling, n loginvPFA_Chebyshev
    for M in M_samples:
        sampling={'type':'loginvPFA_Chebyshev', 'n_max':100}
        bandits.append(BayesianBanditSamplingMonteCarlo(A, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=ln(1/Pfa) Chebyshev, M={}'.format(M))
    # Monte Carlo sampling, n log10invPFA_Chebyshev
    for M in M_samples:
        sampling={'type':'log10invPFA_Chebyshev', 'n_max':100}
        bandits.append(BayesianBanditSamplingMonteCarlo(A, reward_function, reward_prior, sampling, M))
        bandits_labels.append('MC n=log10(1/Pfa) Chebyshev, M={}'.format(M))
                
    # Bandits colors
    bandits_colors=['black', 'skyblue', 'cyan', 'blue', 'palegreen', 'lime', 'green', 'yellow', 'orange', 'red', 'purple', 'fucsia', 'pink']
    bandits_colors=[colors.cnames['black'], colors.cnames['skyblue'], colors.cnames['cyan'], colors.cnames['blue'], colors.cnames['palegreen'], colors.cnames['lime'], colors.cnames['green'], colors.cnames['yellow'], colors.cnames['orange'], colors.cnames['red'], colors.cnames['purple'], colors.cnames['fuchsia'], colors.cnames['pink']]
    
    # Execute each bandit
    for (n,bandit) in enumerate(bandits):
        bandit.execute_realizations(R, t_max, exec_type)
    
    ############################### PLOTTING  ############################### 
    # Plotting overall
    dir_plots=dir_string+'/theta={}'.format(theta[:,0])+'/plots'
    os.makedirs(dir_plots, exist_ok=True)

    # Plotting time: all
    t_plot=t_max
    
    # Plot regret
    plot_std=False
    bandits_plot_regret(returns_expected*np.ones(t_plot), bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_regret(returns_expected*np.ones(t_plot), bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    
    # Plot returns expected
    plot_std=True
    bandits_plot_returns_expected(returns_expected*np.ones(t_plot), bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)

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
    bandits_plot_actions_correct(returns_expected*np.ones(t_plot), bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_actions_correct(returns_expected*np.ones(t_plot), bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    
    # Plot sample size
    plot_std=False
    bandits_plot_n_samples(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    plot_std=True
    bandits_plot_n_samples(bandits, bandits_colors, bandits_labels, t_plot, plot_std, plot_save=dir_plots)
    ###############          

    # Save bandits info
    with open(dir_string+'/theta={}'.format(theta[:,0])+'/bandits.pickle', 'wb') as f:
        pickle.dump(bandits, f)
    with open(dir_string+'/theta={}'.format(theta[:,0])+'/bandits_labels.pickle', 'wb') as f:
        pickle.dump(bandits_labels, f)
            
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Evaluate Bayesian bandits.')
    parser.add_argument('-M', type=int, default=1000, help='Number of samples for the MC integration')
    parser.add_argument('-A', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-exec_type', type=str, default='online', help='Type of execution to run: online or all')
    parser.add_argument('-theta', nargs='+', type=float, default=0, help='Theta')

    # Get arguments
    args = parser.parse_args()
    
    # Make sure A and theta size match
    assert len(args.theta)==args.A, 'Size of theta={} does not match number of arms A={}'.format(args.theta, args.A)
    # Call main function
    main(args.M, args.A, args.t_max, args.R, args.exec_type, args.theta)
