#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import pickle
import sys, os
import argparse
from itertools import *
import pdb
# To avoid type 3 fonts
import matplotlib
#matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import colors

# Add path and import Bayesian Bandits
sys.path.append('/home/iurteaga/Columbia/academic/bandits/src')
sys.path.append('/home/iurteaga/Columbia/academic/bandits/scripts')
# Plotting
from plot_Bandits import *
# Optimal Bandit
from OptimalBandit import *
# Sampling
from BayesianBanditSampling import *
from MCBanditSampling import *
from MCMCBanditSampling import *
from VariationalBanditSampling import *
# Quantiles
from BayesianBanditQuantiles import *
from MCBanditQuantiles import *

################# Gibbs sampler iterations #######################
t_max=500
d_context=2
type_context='rand'

## AGGREGATE Based on 1 R
R=1
n_bandits=1 #nonparametric

# gibbs_max_iters
gibbs_max_iters=np.array([1,5,10,25])
# For scenarios
scenarios=['unbalanced']
for scenario in scenarios:
    print('Scenario {}'.format(scenario))
    if scenario=='easy':
        A=2
        # Bandit dir
        bandit_dir='/home/iurteaga/Columbia/academic/bandits/results/evaluate_linearGaussianMixture_BanditSampling_nonparametric_MCMC/A={}/t_max={}/R={}/d_context={}/type_context={}/pi=0.5 0.5 0.5 0.5/theta=0. 0. 1. 1. 2. 2. 3. 3./sigma=1. 1. 1. 1.'.format(A, t_max, R, d_context, type_context)
    elif scenario=='hard':
        A=2
        # Bandit dir
        bandit_dir='/home/iurteaga/Columbia/academic/bandits/results/evaluate_linearGaussianMixture_BanditSampling_nonparametric_MCMC/A={}/t_max={}/R={}/d_context={}/type_context={}/pi=0.5 0.5 0.3 0.7/theta=1. 1. 2. 2. 0. 0. 3. 3./sigma=1. 1. 1. 1.'.format(A, t_max, R, d_context, type_context)
    elif scenario=='heavy':
        A=2
        # Bandit dir
        bandit_dir='/home/iurteaga/Columbia/academic/bandits/results/evaluate_linearGaussianMixture_BanditSampling_nonparametric_MCMC/A={}/t_max={}/R={}/d_context={}/type_context={}/pi=0.75 0.25 0.75 0.25/theta=0. 0. 0. 0. 2. 2. 2. 2./sigma=1. 10._1. 10.'.format(A, t_max, R, d_context, type_context)
    elif scenario=='unbalanced':
        A=3
        # Bandit dir
        bandit_dir='/home/iurteaga/Columbia/academic/bandits/results/evaluate_linearGaussianMixture_BanditSampling_nonparametric_MCMC/A={}/t_max={}/R={}/d_context={}/type_context={}/pi=1._0._0._0.5 0.5 0._0.3 0.6 0.1/theta=1. 1. 2. 2. 3. 3. 1. 1. 2. 2. 3. 3. 0. 0. 3. 3. 4. 4./sigma=1. 1. 1. 1. 1. 1. 1. 1. 1.'.format(A, t_max, R, d_context, type_context)
    else:
        raise ValueError('Unknown scenario {}'.format(scenario))

    # For each gibbs_max_iter and realization
    gibbs_cumregrets={'mean':np.zeros((len(gibbs_max_iters),t_max)), 'var':np.zeros((len(gibbs_max_iters),t_max))}
    gibbs_reward_diff=np.zeros((len(gibbs_max_iters),A,t_max))
    for (gibbs_idx,gibbs_max_iter) in enumerate(gibbs_max_iters):
        total_R=len(os.listdir('{}/pi_expected/{}'.format(bandit_dir,gibbs_max_iter)))
        print('Scenario {} gibbs_max_iter={} with R={}'.format(scenario,gibbs_max_iter, total_R))
        cumregrets=np.zeros((n_bandits,total_R,t_max))
        reward_diff=np.zeros((n_bandits,total_R,A,t_max))
        
        # Load each file
        for (r,bandit_file) in enumerate(os.listdir('{}/pi_expected/{}'.format(bandit_dir,gibbs_max_iter))):
            try:
                # Load bandits
                f=open('{}/pi_expected/{}/{}'.format(bandit_dir,gibbs_max_iter, bandit_file),'rb')
                bandits=pickle.load(f)
                
                #Cumregrets
                cumregrets[0,r,:]=bandits[1].cumregrets
                # Reward diff
                reward_diff[0,r,:,:]=(bandits[1].rewards_expected-bandits[1].true_expected_rewards)
            except:
                print('NOT LOADED: {}'.format(bandit_file))


        # Plot dir
        os.makedirs('./figs/linearGaussianMixture/{}'.format(scenario), exist_ok=True)
        
        # Summarize
        # reward difference
        for a in np.arange(A):
            gibbs_reward_diff[gibbs_idx,a]=reward_diff[0,a,:,:t_max].mean(axis=0)

        # Cumulative regret
        gibbs_cumregrets['mean'][gibbs_idx]=cumregrets[0,:,:t_max].mean(axis=0)
        gibbs_cumregrets['var'][gibbs_idx]=cumregrets[0,:,:t_max].var(axis=0)

    #### PLOTS
    # Cumulative regret
    bandits_colors=[colors.cnames['black'], colors.cnames['green'], colors.cnames['blue'], colors.cnames['red']]
    gibbs_label='$Gibbs_{max}$'
    for (gibbs_idx,gibbs_max_iter) in enumerate(gibbs_max_iters):
        plt.plot(np.arange(t_max), gibbs_cumregrets['mean'][gibbs_idx,:t_max], bandits_colors[gibbs_idx], label=r'{}={}'.format(gibbs_label, gibbs_max_iter))
        plt.fill_between(np.arange(t_max), gibbs_cumregrets['mean'][gibbs_idx,:t_max]-np.sqrt(gibbs_cumregrets['var'][gibbs_idx,:t_max]), gibbs_cumregrets['mean'][gibbs_idx,:t_max]+np.sqrt(gibbs_cumregrets['var'][gibbs_idx,:t_max]),alpha=0.35, facecolor=bandits_colors[gibbs_idx])
        # Cumulative regret gain
        print('cumregret_{}_reduction_gibbs_max_iter{}={}'.format(scenario,gibbs_max_iter, (gibbs_cumregrets['mean'][gibbs_idx,-1]-gibbs_cumregrets['mean'][0,-1])/gibbs_cumregrets['mean'][0,-1]))
    plt.xlabel('t')
    plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
    plt.axis([0, t_max-1,0,plt.ylim()[1]])
    legend = plt.legend(loc='upper left', ncol=1, shadow=False)
    plt.savefig('./figs/linearGaussianMixture/{}/cumregret_gibbs.pdf'.format(scenario), format='pdf', bbox_inches='tight')
    plt.close()
