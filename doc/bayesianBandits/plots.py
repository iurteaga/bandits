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
from BayesianBanditSampling import *
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


####################### PLOTS 
# Predictive density evolution
# A=3
bandits_file='/home/iurteaga/Columbia/academic/bandits/results/evaluate_BernoulliBandits/A=3/t_max=1500/R=5000/M=1000/N_max=20/theta=0.4_0.7_0.8/bandits.pickle'

with open(bandits_file,'rb') as f:
    bandits=pickle.load(f)

# Double sampling
bandit_idx=1
# Plotting duration
t_max=500

# Plot action predictive density per arm
arm_colors=[colors.cnames['blue'], colors.cnames['green'], colors.cnames['red']]
labels=['Arm 0', 'Arm 1', 'Arm 2']
plt.figure()
for a in np.arange(bandits[bandit_idx].A):
    plt.plot(np.arange(t_max), bandits[bandit_idx].arm_predictive_density['mean'][a][:t_max], arm_colors[a], label=labels[a])
    plt.fill_between(np.arange(t_max), bandits[bandit_idx].arm_predictive_density['mean'][a][:t_max]-bandits[bandit_idx].arm_predictive_density['var'][a][:t_max], bandits[bandit_idx].arm_predictive_density['mean'][a][:t_max]+bandits[bandit_idx].arm_predictive_density['var'][a][:t_max],alpha=0.5, facecolor=arm_colors[a])
plt.xlabel('t')
plt.ylabel(r'$\hat{w}_{a,t+1}$')
#plt.xlim([0, t_max-1])
plt.axis([0, t_max-1,0,1])
#legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
legend = plt.legend(loc='right', ncol=1, shadow=False)
plt.savefig('./figs/bernoulli/pred_action_density.pdf', format='pdf', bbox_inches='tight')
plt.close()

# Plot n_samples
# Trick to smooth things out
bandits[bandit_idx].arm_N_samples[bandits[bandit_idx].arm_N_samples==bandits[bandit_idx].arm_predictive_policy['N_max']]=bandits[bandit_idx].arm_N_samples[np.where(bandits[bandit_idx].arm_N_samples==bandits[bandit_idx].arm_predictive_policy['N_max'])[0]-1]
bandits[bandit_idx].arm_N_samples[bandits[bandit_idx].arm_N_samples==bandits[bandit_idx].arm_predictive_policy['N_max']]=bandits[bandit_idx].arm_N_samples[np.where(bandits[bandit_idx].arm_N_samples==bandits[bandit_idx].arm_predictive_policy['N_max'])[0]-1]
# Actual plot
plt.figure()
plt.plot(np.arange(t_max), bandits[bandit_idx].arm_N_samples[:t_max], 'r')
plt.xlabel('t')
plt.ylabel(r'$N_{t+1}$')
#plt.xlim([0, t_max-1])
plt.axis([0,t_max-1,0,bandits[bandit_idx].arm_predictive_policy['N_max']])
plt.savefig('./figs/bernoulli/n_samples.pdf', format='pdf', bbox_inches='tight')
plt.close()

# Cumulative regret
bandits_colors=[colors.cnames['black'], colors.cnames['red']]
bandits_labels = ['Thompson sampling', 'Double sampling']
plt.figure()
for (n,bandit) in enumerate(bandits):
    plt.plot(np.arange(t_max), bandit.regrets_R['mean'][0,0:t_max].cumsum(), bandits_colors[n], label=bandits_labels[n])
plt.xlabel('t')
plt.ylabel(r'$R_t=\sum_{t=0}^T y_t^*-y_t$')
plt.axis([0, t_max-1,0,12])
legend = plt.legend(loc='upper left', ncol=1, shadow=False)
plt.savefig('./figs/bernoulli/cumulative_regret.pdf', format='pdf', bbox_inches='tight')
plt.close()

################# BERNOULLI BANDITS #######################
# Cumulative regret
# A=2, great
bandits_file='/home/iurteaga/Columbia/academic/bandits/results/evaluate_BernoulliBandits/A=2/t_max=1501/R=5000/M=1000/N_max=20/theta=0.4_0.9/bandits.pickle'

with open(bandits_file,'rb') as f:
    bandits=pickle.load(f)
    
bandits_colors=[colors.cnames['black'], colors.cnames['red']]
bandits_labels = ['Thompson sampling', 'Double sampling']
plt.figure()
for (n,bandit) in enumerate(bandits):
    plt.plot(np.arange(t_max), bandit.cumregrets_R['mean'][0,0:t_max], bandits_colors[n], label=bandits_labels[n])
    plt.fill_between(np.arange(t_max), bandit.cumregrets_R['mean'][0,0:t_max]-np.sqrt(bandit.cumregrets_R['var'][0,0:t_max]), bandit.cumregrets_R['mean'][0,0:t_max]+np.sqrt(bandit.cumregrets_R['var'][0,0:t_max]),alpha=0.5, facecolor=bandits_colors[n])
plt.xlabel('t')
plt.ylabel(r'$R_t=\sum_{t=0}^T y_t^*-y_t$')
plt.axis([0, t_max-1,0,15])
legend = plt.legend(loc='upper left', ncol=1, shadow=False)
plt.savefig('./figs/bernoulli/cumulative_regret_great.pdf', format='pdf', bbox_inches='tight')
plt.close()

# A=2, not great
bandits_file='/home/iurteaga/Columbia/academic/bandits/results/evaluate_BernoulliBandits/A=2/t_max=1501/R=5000/M=1000/N_max=20/theta=0.65_0.9/bandits.pickle'

with open(bandits_file,'rb') as f:
    bandits=pickle.load(f)
    
bandits_colors=[colors.cnames['black'], colors.cnames['red']]
bandits_labels = ['Thompson sampling', 'Double sampling']
plt.figure()
for (n,bandit) in enumerate(bandits):
    plt.plot(np.arange(t_max), bandit.cumregrets_R['mean'][0,0:t_max], bandits_colors[n], label=bandits_labels[n])
    plt.fill_between(np.arange(t_max), bandit.cumregrets_R['mean'][0,0:t_max]-np.sqrt(bandit.cumregrets_R['var'][0,0:t_max]), bandit.cumregrets_R['mean'][0,0:t_max]+np.sqrt(bandit.cumregrets_R['var'][0,0:t_max]),alpha=0.5, facecolor=bandits_colors[n])
plt.xlabel('t')
plt.ylabel(r'$R_t=\sum_{t=0}^T y_t^*-y_t$')
plt.axis([0, t_max-1,0,15])
legend = plt.legend(loc='upper left', ncol=1, shadow=False)
plt.savefig('./figs/bernoulli/cumulative_regret_notgreat.pdf', format='pdf', bbox_inches='tight')
plt.close()

# Ploting Min KL and rel diff
min_KL=np.array([])
regret_reldiff=np.array([])

t_max=1500
t_plot=1000

# Execution parameters
R=5000
M=1000
theta_min=0.05
theta_max=1
theta_diff=0.05

# Bandit size
A=2
for theta in combinations(np.arange(theta_min,theta_max,theta_diff),A):
    # For each theta
    theta=np.array(theta).reshape(A,1)
    theta_str=str.replace(str.replace(str.strip(np.array_str(theta.flatten()),'[]'), '  ', '_'), ' ', '')
    # Directory to load
    dir_string='/home/iurteaga/Columbia/academic/bandits/results/evaluate_BernoulliBandits/A={}/t_max={}/R={}/M={}/N_max=20/theta={}'.format(A, t_max, R, M, theta_str)
    
    # Pickle load
    try:
        f=open(dir_string+'/bandits.pickle', 'rb')
        bandits = pickle.load(f)
        
        # KL divergence
        per_arm_KL=theta[:-1]*np.log(theta[:-1]/theta[-1])+(1-theta[:-1])*np.log((1-theta[:-1])/(1-theta[-1]))
        min_KL=np.append(min_KL, per_arm_KL.min())
        
        print('theta={}, min_KL={}'.format(theta_str, min_KL[-1]))
                       
        # TS Cumulative regret
        TS_cumregret=bandits[0].regrets_R['mean'][0,0:t_max].cumsum()
        # Cumulative regret            
        bandit_cumregret=bandits[1].regrets_R['mean'][0,0:t_max].cumsum()
        # Relative regret at t_plot
        regret_reldiff=np.append(regret_reldiff, (bandit_cumregret[t_plot]-TS_cumregret[t_plot])/TS_cumregret[t_plot])
    except:
        print('MISSING theta={}'.format(theta_str))
        
# Bandit size
A=3
for theta in combinations(np.arange(theta_min,theta_max,theta_diff),A):
    # For each theta
    theta=np.array(theta).reshape(A,1)
    theta_str=str.replace(str.replace(str.strip(np.array_str(theta.flatten()),'[]'), '  ', '_'), ' ', '')
    
    # Directory to load
    dir_string='/home/iurteaga/Columbia/academic/bandits/results/evaluate_BernoulliBandits/A={}/t_max={}/R={}/M={}/N_max=20/theta={}'.format(A, t_max, R, M, theta_str)
    
    # Pickle load
    try:
        f=open(dir_string+'/bandits.pickle', 'rb')
        bandits = pickle.load(f)
        
        # KL divergence
        per_arm_KL=theta[:-1]*np.log(theta[:-1]/theta[-1])+(1-theta[:-1])*np.log((1-theta[:-1])/(1-theta[-1]))
        min_KL=np.append(min_KL, per_arm_KL.min())
        
        print('theta={}, min_KL={}'.format(theta_str, min_KL[-1]))
                       
        # TS Cumulative regret
        TS_cumregret=bandits[0].regrets_R['mean'][0,0:t_max].cumsum()
        # Cumulative regret            
        bandit_cumregret=bandits[1].regrets_R['mean'][0,0:t_max].cumsum()
        # Relative regret at t_plot
        regret_reldiff=np.append(regret_reldiff, (bandit_cumregret[t_plot]-TS_cumregret[t_plot])/TS_cumregret[t_plot])
    except:
        print('MISSING theta={}'.format(theta_str))        
        
#### Sort and plot
assert min_KL.size==regret_reldiff.size
# Min KL vs regret reldiff
min_KL_idx=np.argsort(min_KL, kind='mergesort') # Mergesort because is stable!
for try_f in np.array([0.65, 0.7, 0.75, 0.8, 0.85]):
    plt.figure()
    plt.plot(np.arange(min_KL.size), np.zeros(min_KL.size), color=colors.cnames['black'])
    plt.scatter(min_KL[min_KL_idx], regret_reldiff[min_KL_idx], color=colors.cnames['salmon'])
    plt.plot(min_KL[min_KL_idx], lowess(min_KL[min_KL_idx], regret_reldiff[min_KL_idx], f=try_f, iter=25) , color=colors.cnames['red'])
    plt.xlabel('Bandit learning difficulty\n(Minimum KL)')
    plt.ylabel('Relative regret difference\n($\Delta_{1000}$)')
    plt.xlim([min(min_KL), max(min_KL)])
    plt.savefig('./figs/bernoulli/min_KL_relDiff_t{}_{}.pdf'.format(t_plot,try_f), format='pdf', bbox_inches='tight')
    plt.close()

################# CONTEXTUAL LINEAR GAUSSIAN BANDITS #######################
t_max=1000
# Cumulative regret
bandits_file='/home/iurteaga/Columbia/academic/bandits/results/evaluate_ContextualLinearGaussianBandits/A=2/t_max=1500/R=5000/M=1000/N_max=20/d_context=2/type_context=rand/theta=0.4_0.4_0.8_0.8/sigma=0.2_0.2/bandits.pickle'

with open(bandits_file,'rb') as f:
    bandits=pickle.load(f)
    
bandits_colors=[colors.cnames['black'], colors.cnames['red']]
bandits_labels = ['Thompson sampling', 'Double sampling']
plt.figure()
for (n,bandit) in enumerate(bandits):
    plt.plot(np.arange(t_max), bandit.cumregrets_R['mean'][0,0:t_max], bandits_colors[n], label=bandits_labels[n])
    plt.fill_between(np.arange(t_max), bandit.cumregrets_R['mean'][0,0:t_max]-np.sqrt(bandit.cumregrets_R['var'][0,0:t_max]), bandit.cumregrets_R['mean'][0,0:t_max]+np.sqrt(bandit.cumregrets_R['var'][0,0:t_max]),alpha=0.5, facecolor=bandits_colors[n])
plt.xlabel('t')
plt.ylabel(r'$R_t=\sum_{t=0}^T y_t^*-y_t$')
plt.axis([0, t_max-1,0,15])
legend = plt.legend(loc='upper left', ncol=1, shadow=False)
plt.savefig('./figs/linearGaussian/cumulative_regret_great.pdf', format='pdf', bbox_inches='tight')
plt.close()

# d_context=3, not great
bandits_file='/home/iurteaga/Columbia/academic/bandits/results/evaluate_ContextualLinearGaussianBandits/A=2/t_max=1500/R=5000/M=1000/N_max=20/d_context=2/type_context=rand/theta=0.5_0.5_0.8_0.8/sigma=1._1./bandits.pickle'

with open(bandits_file,'rb') as f:
    bandits=pickle.load(f)
    
bandits_colors=[colors.cnames['black'], colors.cnames['red']]
bandits_labels = ['Thompson sampling', 'Double sampling']
plt.figure()
for (n,bandit) in enumerate(bandits):
    plt.plot(np.arange(t_max), bandit.cumregrets_R['mean'][0,0:t_max], bandits_colors[n], label=bandits_labels[n])
    plt.fill_between(np.arange(t_max), bandit.cumregrets_R['mean'][0,0:t_max]-np.sqrt(bandit.cumregrets_R['var'][0,0:t_max]), bandit.cumregrets_R['mean'][0,0:t_max]+np.sqrt(bandit.cumregrets_R['var'][0,0:t_max]),alpha=0.5, facecolor=bandits_colors[n])
plt.xlabel('t')
plt.ylabel(r'$R_t=\sum_{t=0}^T y_t^*-y_t$')
plt.axis([0, t_max-1,0,65])
legend = plt.legend(loc='upper left', ncol=1, shadow=False)
plt.savefig('./figs/linearGaussian/cumulative_regret_notgreat.pdf', format='pdf', bbox_inches='tight')
plt.close()

# Ploting Min KL and rel diff
min_KL=np.array([])
regret_reldiff=np.array([])

t_max=1500
t_plot=1000

# Execution parameters
R=5000
M=1000
d_context=2

# Bandit size
A=2
# For each provided theta
for theta_factor in np.arange(0.1,1.1,0.1):
    if A==2:
        theta=theta_factor*np.array([-1.0*np.ones(d_context), np.ones(d_context)])
    elif A==3:
        theta=theta_factor*np.array([-1.0*np.ones(d_context), np.zeros(d_context), np.ones(d_context)])
    else:
        raise ValueError('Script not ready for number of arms={}'.format(A))

    # theta in string format
    theta_str=str.replace(str.replace(str.strip(np.array_str(theta.flatten()),' []'),'  ','_'),' ','_')
    # For each provided sigma
    for sigma_factor in np.arange(0.1,1.1,0.1):
        sigma=sigma_factor*np.ones(A)
        # sigma in string format
        sigma_str=str.replace(str.strip(np.array_str(sigma.flatten()),' []'), '  ', '_')
                
        # Directory to load
        dir_string='/home/iurteaga/Columbia/academic/bandits/results/evaluate_ContextualLinearGaussianBandits/A={}/t_max={}/R={}/M={}/N_max=20/d_context={}/type_context=rand/theta={}/sigma={}'.format(A, t_max, R, M, d_context, theta_str, sigma_str)
        # Pickle load
        try:
            f=open(dir_string+'/bandits.pickle', 'rb')
            bandits = pickle.load(f)
            
            # KL divergence between two arms!
            KL=0.5*(np.power((bandits[1].true_expected_rewards[1]-bandits[1].true_expected_rewards[0]),2)/bandits[1].reward_function['sigma'][1])
            print('KL average={} var={}'.format(KL.mean(), KL.var()))
            per_arm_KL=KL.mean()
            min_KL=np.append(min_KL, per_arm_KL)
            
            print('theta={}, sigma={}, min_KL={}'.format(theta_str, sigma_str, min_KL[-1]))
                           
            # TS Cumulative regret
            TS_cumregret=bandits[0].cumregrets_R['mean'][0,0:t_max]
            # Cumulative regret            
            bandit_cumregret=bandits[1].cumregrets_R['mean'][0,0:t_max]
            # Relative regret at t_plot
            regret_reldiff=np.append(regret_reldiff, (bandit_cumregret[t_plot]-TS_cumregret[t_plot])/TS_cumregret[t_plot])
        except:
            print('MISSING theta={} sigma={}'.format(theta_str,sigma_str))
        
        
#### Sort and plot
assert min_KL.size==regret_reldiff.size
# Min KL vs regret reldiff
min_KL_idx=np.argsort(min_KL, kind='mergesort') # Mergesort because is stable!
for try_f in np.array([0.65, 0.7, 0.75, 0.8, 0.85]):
    plt.figure()
    plt.plot(np.arange(min_KL.size), np.zeros(min_KL.size), color=colors.cnames['black'])
    plt.scatter(min_KL[min_KL_idx], regret_reldiff[min_KL_idx], color=colors.cnames['salmon'])
    plt.plot(min_KL[min_KL_idx], lowess(min_KL[min_KL_idx], regret_reldiff[min_KL_idx], f=try_f, iter=25) , color=colors.cnames['red'])
    plt.xlabel('Bandit learning difficulty\n(Minimum KL)')
    plt.ylabel('Relative regret difference\n($\Delta_{1000}$)')
    plt.xlim([min(min_KL), max(min_KL)])
    plt.savefig('./figs/linearGaussian/min_KL_relDiff_t{}_{}.pdf'.format(t_plot,try_f), format='pdf', bbox_inches='tight')
    plt.close()
