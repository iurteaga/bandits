#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import scipy.linalg as linalg
import pickle
import sys, os
import argparse
from itertools import *
import pdb
from matplotlib import colors

# Add path and import Bayesian Bandits
sys.path.append('/home/iurteaga/Columbia/academic/bandits/src')
sys.path.append('/home/iurteaga/Columbia/academic/bandits/scripts')
from VariationalBanditSampling import *
from plot_Bandits import *

# From https://gist.github.com/agramfort/850437
def lowess(x, y, f=2. / 3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(np.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest


def aggregate_statistics(avg_a, var_a, R_a, avg_b, var_b, R_b):
    M2 = var_a*(R_a-1) + var_b*(R_b-1) + (avg_b-avg_a)**2 * (R_a*R_b)/(R_a+R_b)
    return (R_a*avg_a+R_b*avg_b)/(R_a+R_b), M2/(R_a+R_b-1), (R_a+R_b)



######################## # EASY, priorK 3
#### AGGREGATE SUFF STATISTICS
Rs=np.array([250, 251, 252, 253, 255, 256, 257, 258])
n_bandits=3
A=2
t_max=500
cumregrets={'mean':np.zeros((n_bandits,t_max)), 'var':np.zeros((n_bandits,t_max))}
rewards_diff=np.zeros((n_bandits,A,t_max))
total_R=np.zeros(n_bandits)
for R in Rs:
    bandits_file='/home/iurteaga/Columbia/academic/bandits/results/evaluate_ContextualLinearGaussianMixtureBandits_VTS_easy_priorK3/A=2/t_max=500/R={}/d_context=2/type_context=rand/pi=0.5_0.5_0.5_0.5/theta=0_0_1_1_2_2_3_3/sigma=1._1._1._1./bandits.pickle'.format(R)

    try:
        # Load bandits
        f=open(bandits_file,'rb')
        bandits=pickle.load(f)
        
        for (n,bandit) in enumerate(bandits):
            # reward difference
            for a in np.arange(A):
                rewards_diff[n,a,:t_max]= (total_R[n]*rewards_diff[n,a,:t_max]+R*(bandit.rewards_expected_R['mean'][a][:t_max]-bandit.true_expected_rewards[a][:t_max]))/(total_R[n]+R)

            # Cumulative regret
            cumregrets['mean'][n,:t_max], cumregrets['var'][n,:t_max], total_R[n] = aggregate_statistics(cumregrets['mean'][n,:t_max], cumregrets['var'][n,:t_max], total_R[n], bandit.cumregrets_R['mean'][0,0:t_max], bandit.cumregrets_R['var'][0,0:t_max], R)
            
    except:
        print('NOT LOADED: {}'.format(bandits_file))
 
#### PLOTS 
bandits_colors=[colors.cnames['black'], colors.cnames['red'], colors.cnames['blue'], colors.cnames['green']]
bandits_labels = ['VTS with K=1', 'VTS with K=2', 'VTS with K=3', 'VTS with K=4']

# Cumulative regret
t_max=500
for n in np.arange(n_bandits):
    plt.plot(np.arange(t_max), cumregrets['mean'][n,:t_max], bandits_colors[n], label=bandits_labels[n])
    plt.fill_between(np.arange(t_max), cumregrets['mean'][n,:t_max]-np.sqrt(cumregrets['var'][n,:t_max]), cumregrets['mean'][n,:t_max]+np.sqrt(cumregrets['var'][n,:t_max]),alpha=0.5, facecolor=bandits_colors[n])
plt.xlabel('t')
plt.ylabel(r'$R_t=\sum_{t=0}^T y_t^*-y_t$')
plt.axis([0, t_max-1,0, plt.ylim()[1]])
legend = plt.legend(loc='upper left', ncol=1, shadow=False)
plt.savefig('./figs/model_a_cumregret.pdf', format='pdf', bbox_inches='tight')
plt.close()

# Reward squared Error
t_max=100
for a in np.arange(A):
    plt.figure()
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_max), np.power(rewards_diff[n,a,:t_max],2), bandits_colors[n], label='{}, MSE={:.4f}'.format(bandits_labels[n], np.power(rewards_diff[n,a,:t_max],2).mean()))
    plt.xlabel('t')
    plt.ylabel(r'$(\mu_{a,t}-\hat{\mu}_{a,t} )^2$')
    plt.axis([0, t_max-1,0,plt.ylim()[1]])
    legend = plt.legend(loc='upper right', ncol=1, shadow=False)
    plt.savefig('./figs/model_a_mse_arm_{}.pdf'.format(a), format='pdf', bbox_inches='tight')
    plt.close()

######################## HARD 2b, priorK 3
#### AGGREGATE SUFF STATISTICS
Rs=np.array([250, 251, 252, 253, 255, 256, 257, 258])
n_bandits=3
A=2
t_max=500
cumregrets={'mean':np.zeros((n_bandits,t_max)), 'var':np.zeros((n_bandits,t_max))}
rewards_diff=np.zeros((n_bandits,A,t_max))
total_R=np.zeros(n_bandits)
for R in Rs:
    bandits_file='/home/iurteaga/Columbia/academic/bandits/results/evaluate_ContextualLinearGaussianMixtureBandits_VTS_hard2b_priorK3/A=2/t_max=500/R={}/d_context=2/type_context=rand/pi=0.5_0.5_0.3_0.7/theta=1_1_2_2_0_0_3_3/sigma=1._1._1._1./bandits.pickle'.format(R)

    try:
        # Load bandits
        f=open(bandits_file,'rb')
        bandits=pickle.load(f)
        
        for (n,bandit) in enumerate(bandits):
            # reward difference
            for a in np.arange(A):
                rewards_diff[n,a,:t_max]= (total_R[n]*rewards_diff[n,a,:t_max]+R*(bandit.rewards_expected_R['mean'][a][:t_max]-bandit.true_expected_rewards[a][:t_max]))/(total_R[n]+R)

            # Cumulative regret
            cumregrets['mean'][n,:t_max], cumregrets['var'][n,:t_max], total_R[n] = aggregate_statistics(cumregrets['mean'][n,:t_max], cumregrets['var'][n,:t_max], total_R[n], bandit.cumregrets_R['mean'][0,0:t_max], bandit.cumregrets_R['var'][0,0:t_max], R)
            
    except:
        print('NOT LOADED: {}'.format(bandits_file))
     
#### PLOTS 
bandits_colors=[colors.cnames['black'], colors.cnames['red'], colors.cnames['blue'], colors.cnames['green']]
bandits_labels = ['VTS with K=1', 'VTS with K=2', 'VTS with K=3', 'VTS with K=4']

# Cumulative regret
t_max=500
for n in np.arange(n_bandits):
    plt.plot(np.arange(t_max), cumregrets['mean'][n,:t_max], bandits_colors[n], label=bandits_labels[n])
    plt.fill_between(np.arange(t_max), cumregrets['mean'][n,:t_max]-np.sqrt(cumregrets['var'][n,:t_max]), cumregrets['mean'][n,:t_max]+np.sqrt(cumregrets['var'][n,:t_max]),alpha=0.5, facecolor=bandits_colors[n])
plt.xlabel('t')
plt.ylabel(r'$R_t=\sum_{t=0}^T y_t^*-y_t$')
plt.axis([0, t_max-1,0,plt.ylim()[1]])
legend = plt.legend(loc='upper left', ncol=1, shadow=False)
plt.savefig('./figs/model_b_cumregret.pdf', format='pdf', bbox_inches='tight')
plt.close()

# Reward squared Error
t_max=100
for a in np.arange(A):
    plt.figure()
    for n in np.arange(n_bandits):
        plt.plot(np.arange(t_max), np.power(rewards_diff[n,a,:t_max],2), bandits_colors[n], label='{}, MSE={:.4f}'.format(bandits_labels[n], np.power(rewards_diff[n,a,:t_max],2).mean()))
    plt.xlabel('t')
    plt.ylabel(r'$(\mu_{a,t}-\hat{\mu}_{a,t} )^2$')
    plt.axis([0, t_max-1,0,plt.ylim()[1]])
    legend = plt.legend(loc='upper right', ncol=1, shadow=False)
    plt.savefig('./figs/model_b_mse_arm_{}.pdf'.format(a), format='pdf', bbox_inches='tight')
    plt.close()

