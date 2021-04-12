#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import pickle
import sys, os, glob
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

from matplotlib import colors

# Function to agregate statistics obtained
def aggregate_statistics(avg_a, var_a, R_a, avg_b, var_b, R_b):
    M2 = var_a*(R_a-1) + var_b*(R_b-1) + (avg_b-avg_a)**2 * (R_a*R_b)/(R_a+R_b)
    return (R_a*avg_a+R_b*avg_b)/(R_a+R_b), M2/(R_a+R_b-1), (R_a+R_b)

# Function to plot cumulative regret with respect to optimal true expected reward
def plot_cum_regret_all_to_optimal(R_opt_true_expected_rewards, R_rewards, bandit_labels, save_dir, bandits_colors):
    # Just to be consistent in legends
    if np.any(bandit_labels=='LinFullPost'):
        bandit_labels[bandit_labels=='LinFullPost']='LinearGaussian TS'
    if np.any(bandit_labels=='Nonparametric-TS'):
        bandit_labels[bandit_labels=='Nonparametric-TS']='Nonparametric TS'
    
    # Plot dir
    os.makedirs(save_dir, exist_ok=True)
    if R_opt_true_expected_rewards.ndim==2:
        cumregret_to_optexpected=np.nancumsum(R_opt_true_expected_rewards[:,None,:]-R_rewards,axis=2)
    else:
        cumregret_to_optexpected=np.nancumsum(R_opt_true_expected_rewards-R_rewards,axis=2) 
    
    t=np.arange(R_opt_true_expected_rewards.shape[-1])
    
    # Cum regret over time (no std)
    plt.figure()
    for n,bandit_label in enumerate(bandit_labels):
        plt.plot(t, cumregret_to_optexpected.mean(axis=0)[n], bandits_colors[n], label=bandit_label)
    plt.xlabel('t')
    plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
    plt.title('Cumulative (mean) regret over time')
    plt.xlim([0, t[-1]])
    plt.ylim([0,plt.ylim()[1]])
    #legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    plt.savefig('{}/cum_optexpected_regret.pdf'.format(save_dir), format='pdf', bbox_inches='tight')
    plt.close()
    
    # Cum regret over time (with std)
    plt.figure()
    for n,bandit_label in enumerate(bandit_labels):
        plt.plot(t, cumregret_to_optexpected.mean(axis=0)[n], bandits_colors[n], label=bandit_label)
        plt.fill_between(t,
                        cumregret_to_optexpected.mean(axis=0)[n]-cumregret_to_optexpected.std(axis=0)[n],
                        cumregret_to_optexpected.mean(axis=0)[n]+cumregret_to_optexpected.std(axis=0)[n],
                        alpha=0.5, facecolor=bandits_colors[n]
                        )
    plt.xlabel('t')
    plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
    plt.title('Cumulative (mean) regret over time')
    plt.xlim([0, t[-1]])
    plt.ylim([0,plt.ylim()[1]])
    #legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    plt.savefig('{}/cum_optexpected_regret_std.pdf'.format(save_dir), format='pdf', bbox_inches='tight')
    plt.close()
    
    # Cum regret over time (no std): only first five
    plt.figure()
    for n,bandit_label in enumerate(bandit_labels[:5]):
        plt.plot(t, cumregret_to_optexpected.mean(axis=0)[n], bandits_colors[n], label=bandit_label)
    plt.xlabel('t')
    plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
    plt.title('Cumulative (mean) regret over time')
    plt.xlim([0, t[-1]])
    plt.ylim([0,plt.ylim()[1]])
    #legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    plt.savefig('{}/cum_optexpected_regret_top_five.pdf'.format(save_dir), format='pdf', bbox_inches='tight')
    plt.close()
    
    # Cum regret over time (with std): only fist five
    plt.figure()
    for n,bandit_label in enumerate(bandit_labels[:5]):
        plt.plot(t, cumregret_to_optexpected.mean(axis=0)[n], bandits_colors[n], label=bandit_label)
        plt.fill_between(t,
                        cumregret_to_optexpected.mean(axis=0)[n]-cumregret_to_optexpected.std(axis=0)[n],
                        cumregret_to_optexpected.mean(axis=0)[n]+cumregret_to_optexpected.std(axis=0)[n],
                        alpha=0.5, facecolor=bandits_colors[n]
                        )
    plt.xlabel('t')
    plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
    plt.title('Cumulative (mean) regret over time')
    plt.xlim([0, t[-1]])
    plt.ylim([0,plt.ylim()[1]])
    #legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    plt.savefig('{}/cum_optexpected_regret_top_five_std.pdf'.format(save_dir), format='pdf', bbox_inches='tight')
    plt.close()

def cum_regret_all_to_optimal(R_opt_true_expected_rewards, R_rewards, bandit_labels, title):
    # Just to be consistent in legends
    if np.any(bandit_labels=='LinFullPost'):
        bandit_labels[bandit_labels=='LinFullPost']='LinearGaussian TS'
    if np.any(bandit_labels=='Nonparametric-TS'):
        bandit_labels[bandit_labels=='Nonparametric-TS']='Nonparametric TS'
    
    # Plot dir
    if R_opt_true_expected_rewards.ndim==2:
        cumregret_to_optexpected=np.nancumsum(R_opt_true_expected_rewards[:,None,:]-R_rewards,axis=2)
    else:
        cumregret_to_optexpected=np.nancumsum(R_opt_true_expected_rewards-R_rewards,axis=2) 
    
    # Table
    print('---------------------------------------------------')
    print('------\t {} \t------'.format(title))
    print('Algorithm \t \t & Cumregret  & Relative cumregret reduction \\\\ \hline')
    for n,bandit_label in enumerate(bandit_labels):
        print('{:20} \t & {:.3f} & %{:.3f} \\\\ \hline'.format(
                        bandit_label,
                        cumregret_to_optexpected.mean(axis=0)[n,-1],
                        (cumregret_to_optexpected.mean(axis=0)[n,-1]-cumregret_to_optexpected.mean(axis=0)[0,-1])/cumregret_to_optexpected.mean(axis=0)[0,-1]*100
                        )
                )

    print('---------------------------------------------------')
    

# Function to plot cumulative rewards
def plot_cum_rewards(R_rewards, bandit_labels, save_dir, bandits_colors):
    # Just to be consistent in legends
    if np.any(bandit_labels=='LinFullPost'):
        bandit_labels[bandit_labels=='LinFullPost']='LinearGaussian TS'
    if np.any(bandit_labels=='Nonparametric-TS'):
        bandit_labels[bandit_labels=='Nonparametric-TS']='Nonparametric TS'
    
    # Plot dir
    os.makedirs(save_dir, exist_ok=True)
    cumrewards=np.nancumsum(R_rewards,axis=2)
    t=np.arange(R_rewards.shape[2])
    
    # Cum rewards over time (no std)
    plt.figure()
    for n,bandit_label in enumerate(bandit_labels):
        plt.plot(t, cumrewards.mean(axis=0)[n], bandits_colors[n], label=bandit_label)
    plt.xlabel('t')
    plt.ylabel(r'$\sum_{t=0}^T \bar{y}_{t}$')
    plt.title('Cumulative (mean) rewards over time')
    plt.xlim([0, t[-1]])
    plt.ylim([0,plt.ylim()[1]])
    #legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    plt.savefig('{}/cum_rewards.pdf'.format(save_dir), format='pdf', bbox_inches='tight')
    plt.close()
    
    # Cum rewards over time (with std)
    plt.figure()
    for n,bandit_label in enumerate(bandit_labels):
        plt.plot(t, cumrewards.mean(axis=0)[n], bandits_colors[n], label=bandit_label)
        plt.fill_between(t,
                        cumrewards.mean(axis=0)[n]-cumrewards.std(axis=0)[n],
                        cumrewards.mean(axis=0)[n]+cumrewards.std(axis=0)[n],
                        alpha=0.5, facecolor=bandits_colors[n]
                        )
    plt.xlabel('t')
    plt.ylabel(r'$\sum_{t=0}^T \bar{y}_{t}$')
    plt.title('Cumulative (mean) rewards over time')
    plt.xlim([0, t[-1]])
    plt.ylim([0,plt.ylim()[1]])
    #legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    plt.savefig('{}/cum_rewards_std.pdf'.format(save_dir), format='pdf', bbox_inches='tight')
    plt.close()
    
    # Cum rewards over time (no std): only first five
    plt.figure()
    for n,bandit_label in enumerate(bandit_labels[:5]):
        plt.plot(t, cumrewards.mean(axis=0)[n], bandits_colors[n], label=bandit_label)
    plt.xlabel('t')
    plt.ylabel(r'$\sum_{t=0}^T \bar{y}_{t}$')
    plt.title('Cumulative (mean) rewards over time')
    plt.xlim([0, t[-1]])
    plt.ylim([0,plt.ylim()[1]])
    #legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    plt.savefig('{}/cum_rewards_top_five.pdf'.format(save_dir), format='pdf', bbox_inches='tight')
    plt.close()
    
    # Cum rewards over time (with std): only first five
    plt.figure()
    for n,bandit_label in enumerate(bandit_labels[:5]):
        plt.plot(t, cumrewards.mean(axis=0)[n], bandits_colors[n], label=bandit_label)
        plt.fill_between(t,
                        cumrewards.mean(axis=0)[n]-cumrewards.std(axis=0)[n],
                        cumrewards.mean(axis=0)[n]+cumrewards.std(axis=0)[n],
                        alpha=0.5, facecolor=bandits_colors[n]
                        )
    plt.xlabel('t')
    plt.ylabel(r'$\sum_{t=0}^T \bar{y}_{t}$')
    plt.title('Cumulative (mean) rewards over time')
    plt.xlim([0, t[-1]])
    plt.ylim([0,plt.ylim()[1]])
    #legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    legend = plt.legend(loc='upper left', ncol=1, shadow=True)
    plt.savefig('{}/cum_rewards_top_five_std.pdf'.format(save_dir), format='pdf', bbox_inches='tight')
    plt.close()
    
def rewards_table(R_opt_rewards, R_rewards, bandit_labels, title, bandits_colors):
    '''
        Displays summary statistics of the performance of each algorithm.
    '''
    # Just to be consistent in legends
    if np.any(bandit_labels=='LinFullPost'):
        bandit_labels[bandit_labels=='LinFullPost']='LinearGaussian TS'
    if np.any(bandit_labels=='Nonparametric-TS'):
        bandit_labels[bandit_labels=='Nonparametric-TS']='Nonparametric TS'
    
    # Optimal rewards
    if R_opt_rewards.ndim==2:
        opt_rewards=np.nansum(R_opt_rewards,axis=1)
    else:
        opt_rewards=np.nansum(R_opt_rewards,axis=2)
    # Figure out performance order of interest
    final_rewards=np.nansum(R_rewards,axis=2)
    #performance_order=np.argsort(final_rewards.mean(axis=0))[::-1]
    
    # Summary
    print('---------------------------------------------------')
    print('------\t {} \t------'.format(title))
    print('Algorithm \t \t & Total reward \\\\ \hline')
    print('Optimal \t \t & {:.3f} $\pm$ {:.3f} \\\\ \hline'.format(opt_rewards.mean(), opt_rewards.std()))
    for n,bandit_label in enumerate(bandit_labels):
        print('{:20} \t & {:.3f} $\pm$ {:.3f} \\\\ \hline'.format(
                        bandit_label,
                        final_rewards.mean(axis=0)[n],
                        final_rewards.std(axis=0)[n]
                        )
                )

    print('---------------------------------------------------')
    
def exec_time_analysis(R_times, bandit_labels, save_dir, bandit_colors=None):
    '''
        Displays execution time statistics of each algorithm.
    '''
    # Just to be consistent in legends
    if np.any(bandit_labels=='LinFullPost'):
        bandit_labels[bandit_labels=='LinFullPost']='LinearGaussian TS'
    if np.any(bandit_labels=='Nonparametric-TS'):
        bandit_labels[bandit_labels=='Nonparametric-TS']='Nonparametric TS'
    
    # Plot dir
    os.makedirs(save_dir, exist_ok=True)
    # Figure out performance order of interest
    total_times_mean=R_times.mean(axis=0)
    total_times_std=R_times.std(axis=0)
    performance_order=np.argsort(total_times_mean)
    
    # Summary
    print('---------------------------------------------------')
    print('------\t {} \t------'.format(save_dir.split('/')[-2]))
    print('Algorithm \t \t & Total time \\\\ \hline')
    for n in performance_order:
        print('{:20} \t & {:.3f} $\pm$ {:.3f} \\\\ \hline'.format(
                        bandit_labels[n],
                        total_times_mean[n],
                        total_times_std[n],
                        )
                )
    print('---------------------------------------------------')
    print('Total execution mean time={}'.format(total_times_mean.sum()))
    print('---------------------------------------------------')
    if bandit_colors is None:
        bandit_colors=['b']*total_times_mean.size
        bandit_colors[0]='r' # Nonparametrics in red
    plt.figure()
    plt.bar(np.arange(total_times_mean.size), total_times_mean, yerr=np.sqrt(total_times_std), color=bandit_colors, ecolor=bandit_colors)
    plt.xticks(np.arange(total_times_mean.size), bandit_labels, rotation=50, ha='right')
    plt.ylabel(r'$s$')
    plt.savefig('{}/exec_time_barplot.pdf'.format(save_dir), format='pdf', bbox_inches='tight')
    plt.close()


################# Linear Gaussian, from showdown  #######################
bandit_showdown_dir='/home/iurteaga/Columbia/academic/bandits/results/bandit_showdown'

bandit_colors=[colors.cnames['red'], colors.cnames['black'], colors.cnames['green'], colors.cnames['lime'], colors.cnames['slategrey'], colors.cnames['darkorange'], colors.cnames['darkgoldenrod'], colors.cnames['saddlebrown'], colors.cnames['olivedrab'], colors.cnames['yellow'],  colors.cnames['purple'], colors.cnames['orange'], colors.cnames['palegreen'], colors.cnames['fuchsia'], colors.cnames['pink'], colors.cnames['saddlebrown'], colors.cnames['chocolate'], colors.cnames['burlywood'], colors.cnames['black']]
baseline_colors=[colors.cnames['red'], colors.cnames['green'], colors.cnames['lime'], colors.cnames['slategrey'], colors.cnames['darkorange'], colors.cnames['darkgoldenrod'], colors.cnames['saddlebrown'], colors.cnames['olivedrab'], colors.cnames['yellow'],  colors.cnames['purple'], colors.cnames['orange'], colors.cnames['palegreen'], colors.cnames['fuchsia'], colors.cnames['pink'], colors.cnames['saddlebrown'], colors.cnames['chocolate'], colors.cnames['burlywood'], colors.cnames['black']]
oracle_colors=[colors.cnames['red'], colors.cnames['blue']]

experiment='linear_t1500_gibbsmaxiter10_trainfreq1'
# Load results
with open('{}/{}/R_results.npz'.format(bandit_showdown_dir, experiment), 'rb') as f:
    all_results=np.load(f, allow_pickle=True)
    # Oracle: Linear TS
    linear_ts_idx=np.where(all_results['labels']=='LinFullPost')[0][0]
    nonparametric_ts_idx=np.where(all_results['labels']=='Nonparametric-TS')[0][0]
    non_oracle_idx=np.setdiff1d(np.arange(all_results['labels'].size), linear_ts_idx)
    
    # Display results
    plot_cum_regret_all_to_optimal(all_results['opt_true_expected_rewards'], all_results['rewards'][:,non_oracle_idx,:], all_results['labels'][non_oracle_idx], './figs/linear_showdown_baselines/', baseline_colors)
    cum_regret_all_to_optimal(all_results['opt_true_expected_rewards'], all_results['rewards'][:,non_oracle_idx,:], all_results['labels'][non_oracle_idx], 'Linear Gaussian Baselines')
    plot_cum_regret_all_to_optimal(all_results['opt_true_expected_rewards'], all_results['rewards'][:,[nonparametric_ts_idx,linear_ts_idx],:], all_results['labels'][[nonparametric_ts_idx,linear_ts_idx]], './figs/linear_showdown_oracle/', oracle_colors)
    rewards_table(all_results['opt_rewards'], all_results['rewards'][:,non_oracle_idx,:], all_results['labels'][non_oracle_idx], 'Linear Gaussian Baselines', baseline_colors)
    rewards_table(all_results['opt_rewards'], all_results['rewards'][:,[nonparametric_ts_idx,linear_ts_idx],:], all_results['labels'][[nonparametric_ts_idx,linear_ts_idx]], 'Linear Gaussian Oracle', oracle_colors)
    exec_time_analysis(all_results['times'], all_results['labels'], './figs/linear_showdown/')

experiment='sparse_linear_t1500_gibbsmaxiter10_trainfreq1'
# Load results
with open('{}/{}/R_results.npz'.format(bandit_showdown_dir, experiment), 'rb') as f:
    all_results=np.load(f, allow_pickle=True)
    # Oracle: Linear TS
    linear_ts_idx=np.where(all_results['labels']=='LinFullPost')[0][0]
    nonparametric_ts_idx=np.where(all_results['labels']=='Nonparametric-TS')[0][0]
    non_oracle_idx=np.setdiff1d(np.arange(all_results['labels'].size), linear_ts_idx)
    
    # Display results
    plot_cum_regret_all_to_optimal(all_results['opt_true_expected_rewards'], all_results['rewards'][:,non_oracle_idx,:], all_results['labels'][non_oracle_idx], './figs/sparse_linear_showdown_baselines/', baseline_colors)
    cum_regret_all_to_optimal(all_results['opt_true_expected_rewards'], all_results['rewards'][:,non_oracle_idx,:], all_results['labels'][non_oracle_idx], 'Sparse Linear Gaussian Baselines')
    plot_cum_regret_all_to_optimal(all_results['opt_true_expected_rewards'], all_results['rewards'][:,[nonparametric_ts_idx,linear_ts_idx],:], all_results['labels'][[nonparametric_ts_idx,linear_ts_idx]], './figs/sparse_linear_showdown_oracle/', oracle_colors)
    rewards_table(all_results['opt_rewards'], all_results['rewards'][:,non_oracle_idx,:], all_results['labels'][non_oracle_idx], 'Sparse Linear Gaussian Baselines', baseline_colors)
    rewards_table(all_results['opt_rewards'], all_results['rewards'][:,[nonparametric_ts_idx,linear_ts_idx],:], all_results['labels'][[nonparametric_ts_idx,linear_ts_idx]], 'Sparse Linear Gaussian Oracle', oracle_colors)
    exec_time_analysis(all_results['times'], all_results['labels'], './figs/linear_showdown/')
    exec_time_analysis(all_results['times'], all_results['labels'], './figs/sparse_linear_showdown/')

################# Bernoulli (for appendix)  #######################
main_dir='/home/iurteaga/Columbia/academic/bandits/results/evaluate_Bernoulli_Bandit_nonparametric'
t_max=1500
M=1
N_max=1
R=1
# Different parameterizations
As=[2]
theta_min=0.5
theta_max=1
theta_diff=0.2
n_bandits=2 # Gaussian and nonparametric
# Plot dir
os.makedirs('./figs/bernoulli', exist_ok=True)
print('***** Bernoulli *****')
# Process
for A in As:
    for theta in combinations(np.arange(theta_min,theta_max,theta_diff),A):
        # For each theta (in string format)
        theta_str='_'.join([format(_, '.1f') for _ in theta])

        # New parameterization
        r_loaded=False
        # Cumregrets
        cumregrets_R={'mean':np.zeros((n_bandits,t_max)), 'var':np.zeros((n_bandits,t_max)), 'R':np.zeros(n_bandits)}
        # Reward diffs
        rewards_diff=np.zeros((n_bandits,A,t_max))
        # Load each file
        bandit_dir='{}/A={}/t_max={}/R={}/M={}/N_max={}/theta={}'.format(main_dir,A,t_max,R,M,N_max,theta_str)
        if os.path.isdir(bandit_dir):
            all_bandit_files = glob.glob('{}/bandits_r*.pickle'.format(bandit_dir))
            total_R=len(all_bandit_files)
            cumregrets=np.zeros((n_bandits,total_R,t_max))
            reward_diff=np.zeros((n_bandits,total_R,A,t_max))
            R_loaded=0
            for (r,bandit_file) in enumerate(all_bandit_files):
                try:
                    # Load bandits
                    f=open('{}'.format(bandit_file),'rb')
                    bandits=pickle.load(f)
                    for (n,bandit) in enumerate(bandits):
                        #Cumregrets
                        cumregrets[n,r,:]=bandits[n].cumregrets
                        # Reward diff
                        reward_diff[n,r,:,:]=(bandits[n].rewards_expected-bandits[n].true_expected_rewards)
                    # Count if successful
                    R_loaded+=1
                except:
                    print('NOT LOADED: {}'.format(bandit_file))

            if R_loaded>0:
                r_loaded=True
                # reward difference averages
                rewards_diff=reward_diff.mean(axis=1)
                # Cumulative regret averages
                cumregrets_R['mean']=cumregrets.mean(axis=1)
                cumregrets_R['var']=cumregrets.var(axis=1)
                cumregrets_R['R']=R_loaded
                
                if R_loaded!=total_R:
                    print('Only {} out {} files loaded'.format(R_loaded, total_R))
        else:
            print('NOT a directory: {}'.format(bandit_dir))
                
        if r_loaded:
            print('Cumregret for {} realizations available'.format(cumregrets_R['R']))
            #### PLOTS
            bandits_colors=[colors.cnames['blue'], colors.cnames['red']]
            bandits_labels = ['Bernoulli TS', 'Nonparametric TS']

            # Cumulative regret
            for n in np.arange(n_bandits):
                plt.plot(np.arange(t_max), cumregrets_R['mean'][n,:t_max], bandits_colors[n], label=bandits_labels[n])
                plt.fill_between(np.arange(t_max), cumregrets_R['mean'][n,:t_max]-np.sqrt(cumregrets_R['var'][n,:t_max]), cumregrets_R['mean'][n,:t_max]+np.sqrt(cumregrets_R['var'][n,:t_max]),alpha=0.35, facecolor=bandits_colors[n])
            plt.xlabel('t')
            plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
            plt.axis([0, t_max-1,0,plt.ylim()[1]])
            legend = plt.legend(loc='upper left', ncol=1, shadow=False)
            plt.savefig('./figs/bernoulli/cumregret_A{}_{}.pdf'.format(A, theta_str.replace('.','')), format='pdf', bbox_inches='tight')
            plt.close()

################# Static Gaussian (for appendix)  #######################
main_dir='/home/iurteaga/Columbia/academic/bandits/results/evaluate_linearGaussian_Bandit_nonparametric'
t_max=1500
M=1
N_max=1
d_context=1
type_context='static'
R=1
# Different parameterizations
As=[2]
theta_factors=[0.1, 0.5, 1]
sigma_factors=[1.]
n_bandits=2 # Gaussian and nonparametric
# Plot dir
os.makedirs('./figs/staticGaussian', exist_ok=True)
print('***** Static Gaussian *****')

# Process
for A in As:
    for theta_factor in theta_factors:
        tmp_range=np.arange(np.floor(A/2))+1
        if A%2:
            a_range=np.concatenate([-1*tmp_range[::-1][:], np.array([0]), tmp_range])
        else:
            a_range=np.concatenate([-1*tmp_range[::-1][:], tmp_range])

        theta=theta_factor*a_range[:,None]*np.ones(d_context)

        # theta in string format
        #theta_str='_'.join('{}'.format(_) for _ in theta.flatten())
        theta_str='_'.join(str.strip(np.array_str(theta.flatten()),'[]').split())

        # For each provided sigma
        for sigma_factor in sigma_factors:
            sigma=sigma_factor*np.ones(A)
            # sigma in string format
            #sigma_str='_'.join('{}'.format(_) for _ in sigma.flatten())
            sigma_str='_'.join(str.strip(np.array_str(sigma.flatten()),'[]').split())

            # New parameterization
            r_loaded=False
            # Cumregrets
            cumregrets_R={'mean':np.zeros((n_bandits,t_max)), 'var':np.zeros((n_bandits,t_max)), 'R':np.zeros(n_bandits)}
            # Reward diffs
            rewards_diff=np.zeros((n_bandits,A,t_max))
            # Load each file
            bandit_dir='{}/A={}/t_max={}/R={}/M={}/N_max={}/d_context={}/type_context={}/theta={}/sigma={}'.format(main_dir,A,t_max,R,M,N_max,d_context,type_context,theta_str,sigma_str)
            if os.path.isdir(bandit_dir):
                all_bandit_files = glob.glob('{}/bandits_r*.pickle'.format(bandit_dir))
                total_R=len(all_bandit_files)
                cumregrets=np.zeros((n_bandits,total_R,t_max))
                reward_diff=np.zeros((n_bandits,total_R,A,t_max))
                R_loaded=0
                for (r,bandit_file) in enumerate(all_bandit_files):
                    try:
                        # Load bandits
                        f=open('{}'.format(bandit_file),'rb')
                        bandits=pickle.load(f)
                        for (n,bandit) in enumerate(bandits):
                            #Cumregrets
                            cumregrets[n,r,:]=bandits[n].cumregrets
                            # Reward diff
                            reward_diff[n,r,:,:]=(bandits[n].rewards_expected-bandits[n].true_expected_rewards)
                        # Count if successful
                        R_loaded+=1
                    except:
                        print('NOT LOADED: {}'.format(bandit_file))

                if R_loaded>0:
                    r_loaded=True
                    # reward difference averages
                    rewards_diff=reward_diff.mean(axis=1)
                    # Cumulative regret averages
                    cumregrets_R['mean']=cumregrets.mean(axis=1)
                    cumregrets_R['var']=cumregrets.var(axis=1)
                    cumregrets_R['R']=R_loaded
                    
                    if R_loaded!=total_R:
                        print('Only {} out {} files loaded'.format(R_loaded, total_R))
            else:
                print('NOT a directory: {}'.format(bandit_dir))
                    
            if r_loaded:
                print('Cumregret for {} realizations available'.format(cumregrets_R['R']))
                #### PLOTS
                bandits_colors=[colors.cnames['blue'], colors.cnames['red']]
                bandits_labels = ['Gaussian TS', 'Nonparametric TS']

                # Cumulative regret
                for n in np.arange(n_bandits):
                    plt.plot(np.arange(t_max), cumregrets_R['mean'][n,:t_max], bandits_colors[n], label=bandits_labels[n])
                    plt.fill_between(np.arange(t_max), cumregrets_R['mean'][n,:t_max]-np.sqrt(cumregrets_R['var'][n,:t_max]), cumregrets_R['mean'][n,:t_max]+np.sqrt(cumregrets_R['var'][n,:t_max]),alpha=0.35, facecolor=bandits_colors[n])
                plt.xlabel('t')
                plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
                plt.axis([0, t_max-1,0,plt.ylim()[1]])
                legend = plt.legend(loc='upper left', ncol=1, shadow=False)
                plt.savefig('./figs/staticGaussian/cumregret_A{}_{}_{}.pdf'.format(A, theta_str.replace('.',''),sigma_str.replace('.','')), format='pdf', bbox_inches='tight')
                plt.close()

                # Reward squared Error
                '''
                t_plot=100
                for a in np.arange(A):
                    plt.figure()
                    for n in np.arange(n_bandits):
                        plt.plot(np.arange(t_plot), np.power(rewards_diff[n,a,:t_plot],2), bandits_colors[n], label='{}, MSE={:.4f}'.format(bandits_labels[n], np.power(rewards_diff[n,a,:t_plot],2).mean()))
                    plt.xlabel('t')
                    plt.ylabel(r'$(\mu_{a,t}-\hat{\mu}_{a,t} )^2$')
                    plt.axis([0, t_plot-1,0,plt.ylim()[1]])
                    legend = plt.legend(loc='upper right', ncol=1, shadow=False)
                    plt.savefig('./figs/staticGaussian/mse_a_{}_A{}_{}_{}.pdf'.format(a,A, theta_str.replace('.',''),sigma_str.replace('.','')), format='pdf', bbox_inches='tight')
                    plt.close()
                '''

################# Linear Gaussian #######################
main_dir='/home/iurteaga/Columbia/academic/bandits/results/evaluate_linearGaussian_Bandit_nonparametric'
t_max=500
M=1
N_max=1
d_context=2
type_context='rand'
# Different parameterizations
As=[2,3,4,5]
theta_factors=[0.1, 0.5, 1.]
sigma_factors=[1.]
## AGGREGATE SUFF STATISTICS
Rs=np.array([1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])
n_bandits=2 #Linear Gaussian and nonparametric
# Plot dir
os.makedirs('./figs/linearGaussian', exist_ok=True)

# Process
for A in As:
    for theta_factor in theta_factors:
        tmp_range=np.arange(np.floor(A/2))+1
        if A%2:
            a_range=np.concatenate([-1*tmp_range[::-1][:], np.array([0]), tmp_range])
        else:
            a_range=np.concatenate([-1*tmp_range[::-1][:], tmp_range])

        theta=theta_factor*a_range[:,None]*np.ones(d_context)

        # theta in string format
        #theta_str='_'.join('{}'.format(_) for _ in theta.flatten())
        theta_str='_'.join(str.strip(np.array_str(theta.flatten()),'[]').split())

        # For each provided sigma
        for sigma_factor in sigma_factors:
            sigma=sigma_factor*np.ones(A)
            # sigma in string format
            #sigma_str='_'.join('{}'.format(_) for _ in sigma.flatten())
            sigma_str='_'.join(str.strip(np.array_str(sigma.flatten()),'[]').split())

            # New parameterization
            r_loaded=False
            # Cumregrets
            cumregrets_R={'mean':np.zeros((n_bandits,t_max)), 'var':np.zeros((n_bandits,t_max)), 'R':np.zeros(n_bandits)}
            # Reward diffs
            rewards_diff=np.zeros((n_bandits,A,t_max))
            for R in Rs:
                if R==1:
                    # Load each file
                    bandit_dir='{}/A={}/t_max={}/R={}/M={}/N_max={}/d_context={}/type_context={}/theta={}/sigma={}'.format(main_dir,A,t_max,R,M,N_max,d_context,type_context,theta_str,sigma_str)
                    if os.path.isdir(bandit_dir):
                        total_R=len(os.listdir('{}'.format(bandit_dir)))
                        cumregrets=np.zeros((n_bandits,total_R,t_max))
                        reward_diff=np.zeros((n_bandits,total_R,A,t_max))
                        R_loaded=0
                        for (r,bandit_file) in enumerate(os.listdir('{}'.format(bandit_dir))):
                            try:
                                # Load bandits
                                f=open('{}/{}'.format(bandit_dir,bandit_file),'rb')
                                bandits=pickle.load(f)
                                for (n,bandit) in enumerate(bandits):
                                    #Cumregrets
                                    cumregrets[n,r,:]=bandits[n].cumregrets
                                    # Reward diff
                                    reward_diff[n,r,:,:]=(bandits[n].rewards_expected-bandits[n].true_expected_rewards)
                                # Count if successful
                                R_loaded+=1
                            except:
                                print('NOT LOADED: {}'.format(bandit_file))

                        if R_loaded>0:
                            r_loaded=True
                            # reward difference averages
                            rewards_diff=reward_diff.mean(axis=1)
                            # Cumulative regret averages
                            cumregrets_R['mean']=cumregrets.mean(axis=1)
                            cumregrets_R['var']=cumregrets.var(axis=1)
                            cumregrets_R['R']=R_loaded
                            
                            if R_loaded!=total_R:
                                print('Only {} out {} files loaded'.format(R_loaded, total_R))
                    else:
                        print('NOT a directory: {}'.format(bandit_dir))
                else:
                    # Bandit
                    bandits_file='{}/A={}/t_max={}/R={}/M={}/N_max={}/d_context={}/type_context={}/theta={}/sigma={}/bandits.pickle'.format(main_dir,A,t_max,R,M,N_max,d_context,type_context,theta_str,sigma_str)
                    
                    try:
                        # Load bandits
                        f=open(bandits_file,'rb')
                        bandits=pickle.load(f)
                        
                        for (n,bandit) in enumerate(bandits):
                            # reward difference
                            for a in np.arange(A):
                                rewards_diff[n,a,:t_max]= (cumregrets_R['R'][n]*rewards_diff[n,a,:t_max]+R*(bandit.rewards_expected_R['mean'][a][:t_max]-bandit.true_expected_rewards[a][:t_max]))/(cumregrets_R['R'][n]+R)
                            
                            # Cumulative regret
                            cumregrets_R['mean'][n,:t_max], cumregrets_R['var'][n,:t_max], cumregrets_R['R'][n] = aggregate_statistics(cumregrets_R['mean'][n,:t_max], cumregrets_R['var'][n,:t_max], cumregrets_R['R'][n], bandit.cumregrets_R['mean'][0,0:t_max], bandit.cumregrets_R['var'][0,0:t_max], R)

                        r_loaded=True
                    except:
                        print('NOT LOADED: {}'.format(bandits_file))
                    
            if r_loaded:
                print('Cumregret for {} realizations available'.format(cumregrets_R['R']))
                #### PLOTS
                bandits_colors=[colors.cnames['blue'], colors.cnames['red']]
                bandits_labels = ['LinearGaussian TS', 'Nonparametric TS']

                # Cumulative regret
                for n in np.arange(n_bandits):
                    plt.plot(np.arange(t_max), cumregrets_R['mean'][n,:t_max], bandits_colors[n], label=bandits_labels[n])
                    plt.fill_between(np.arange(t_max), cumregrets_R['mean'][n,:t_max]-np.sqrt(cumregrets_R['var'][n,:t_max]), cumregrets_R['mean'][n,:t_max]+np.sqrt(cumregrets_R['var'][n,:t_max]),alpha=0.35, facecolor=bandits_colors[n])
                plt.xlabel('t')
                plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
                plt.axis([0, t_max-1,0,plt.ylim()[1]])
                legend = plt.legend(loc='upper left', ncol=1, shadow=False)
                plt.savefig('./figs/linearGaussian/cumregret_A{}_{}_{}.pdf'.format(A, theta_str.replace('.',''),sigma_str.replace('.','')), format='pdf', bbox_inches='tight')
                plt.close()

                # Reward squared Error
                '''
                t_plot=100
                for a in np.arange(A):
                    plt.figure()
                    for n in np.arange(n_bandits):
                        plt.plot(np.arange(t_plot), np.power(rewards_diff[n,a,:t_plot],2), bandits_colors[n], label='{}, MSE={:.4f}'.format(bandits_labels[n], np.power(rewards_diff[n,a,:t_plot],2).mean()))
                    plt.xlabel('t')
                    plt.ylabel(r'$(\mu_{a,t}-\hat{\mu}_{a,t} )^2$')
                    plt.axis([0, t_plot-1,0,plt.ylim()[1]])
                    legend = plt.legend(loc='upper right', ncol=1, shadow=False)
                    plt.savefig('./figs/linearGaussian/mse_a_{}_A{}_{}_{}.pdf'.format(a,A, theta_str.replace('.',''),sigma_str.replace('.','')), format='pdf', bbox_inches='tight')
                    plt.close()
                '''

################# Linear Gaussian Mixtures, from showdown  #######################
experiments=[
    'linear_gaussian_mixture_easy_t1000_gibbsmaxiter10_trainfreq1',
    'linear_gaussian_mixture_hard_t1000_gibbsmaxiter10_trainfreq1',
    'linear_gaussian_mixture_unbalanced_t1000_gibbsmaxiter10_trainfreq1',
    'linear_gaussian_mixture_heavy_tail_t1000_gibbsmaxiter10_trainfreq1',
    ]

# Load results
for experiment in experiments:
    experiment_title=experiment.split('_t1000_')[0]
    with open('{}/{}/R_results.npz'.format(bandit_showdown_dir, experiment), 'rb') as f:
        all_results=np.load(f, allow_pickle=True)        
        # Display results
        plot_cum_regret_all_to_optimal(all_results['opt_true_expected_rewards'], all_results['rewards'], all_results['labels'], './figs/{}_baselines/'.format(experiment_title), bandit_colors)
        cum_regret_all_to_optimal(all_results['opt_true_expected_rewards'], all_results['rewards'], all_results['labels'], '{}'.format(experiment_title))
        rewards_table(all_results['opt_rewards'], all_results['rewards'], all_results['labels'], '{} Baselines'.format(experiment_title), bandit_colors)
        exec_time_analysis(all_results['times'], all_results['labels'], './figs/{}/'.format(experiment_title))

# Evaluating nonparametric TS with different gibbsmax iterations
experiments=[
    'linear_gaussian_mixture_easy_t1000_gibbsmaxiter1_trainfreq1',
    'linear_gaussian_mixture_hard_t1000_gibbsmaxiter1_trainfreq1',
    'linear_gaussian_mixture_unbalanced_t1000_gibbsmaxiter1_trainfreq1',
    'linear_gaussian_mixture_heavy_tail_t1000_gibbsmaxiter1_trainfreq1',
    ]

R=500
t=1000
gibbsmaxiters=[1,5,10]
# Load results
for experiment in experiments:
    experiment_title=experiment.split('_t1000_')[0]
    # Plot dir
    plot_dir='./figs/{}_gibbsmaxiter/'.format(experiment_title)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Variables
    R_opt_rewards=np.zeros((R,len(gibbsmaxiters),t))
    R_opt_true_expected_rewards=np.zeros((R,len(gibbsmaxiters),t))
    R_rewards=np.zeros((R,len(gibbsmaxiters),t))
    R_regrets=np.zeros((R,len(gibbsmaxiters),t))
    R_elapsed_times=np.zeros((R,len(gibbsmaxiters)))
    R_elapsed_times_others=np.zeros((R,len(gibbsmaxiters),9)) # Hacked for now
    R_labels=[]
    for n_iter,gibbsmaxiter in enumerate(gibbsmaxiters):
        with open('{}/{}/R_results.npz'.format(bandit_showdown_dir, experiment.replace('gibbsmaxiter1','gibbsmaxiter{}'.format(gibbsmaxiter))), 'rb') as f:
            all_results=np.load(f, allow_pickle=True)
            # Nonparametric TS
            nonparametric_ts_idx=np.where(all_results['labels']=='Nonparametric-TS')[0][0]
            other_idx=np.setdiff1d(np.arange(all_results['labels'].size),nonparametric_ts_idx)
            other_labels=all_results['labels'][other_idx]
            # Load from this run
            R_labels.append('Nonparametric TS Gibbs max={}'.format(gibbsmaxiter))
            R_opt_rewards[:,n_iter,:]=all_results['opt_rewards'][:R,:]
            R_opt_true_expected_rewards[:,n_iter,:]=all_results['opt_true_expected_rewards'][:R,:]
            R_rewards[:,n_iter,:]=all_results['rewards'][:R,nonparametric_ts_idx,:]
            R_elapsed_times[:,n_iter]=all_results['times'][:R,nonparametric_ts_idx]
            R_elapsed_times_others[:,n_iter]=all_results['times'][:R,other_idx]

    # Display results
    my_colors=[colors.cnames['salmon'], colors.cnames['tomato'], colors.cnames['red']]
    my_colors=[colors.cnames['yellow'], colors.cnames['orange'], colors.cnames['red']]
    plot_cum_regret_all_to_optimal(R_opt_true_expected_rewards, R_rewards, R_labels, './figs/{}_gibbsmaxiter/'.format(experiment_title), my_colors)
    rewards_table(R_opt_rewards, R_rewards, R_labels, experiment_title, my_colors)
    exec_time_analysis(R_elapsed_times, R_labels, './figs/{}_gibbsmaxiter/'.format(experiment_title), my_colors)
    
    # Execution times together
    exec_time_analysis(np.concatenate((R_elapsed_times,R_elapsed_times_others.mean(axis=1)), axis=1), R_labels+other_labels.tolist(), './figs/{}/'.format(experiment_title), my_colors+[colors.cnames['blue']]*other_labels.size)

################# Gibbs sampler iterations #######################
'''
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
'''
################# Linear Gaussian Mixture #######################
t_max=500
d_context=2
type_context='rand'

## AGGREGATE Based on 1 R
R=1
n_bandits=2 #Fixed K and nonparametric

# For executions
executions=['MCMC']
for execution in executions:
    print('Execution {}'.format(execution))
    # For scenarios
    scenarios=['easy', 'hard', 'heavy', 'unbalanced']
    for scenario in scenarios:
        print('Execution {} Scenario {}'.format(execution, scenario))
        if scenario=='easy':
            A=2
            # Bandit dir
            bandit_dir='/home/iurteaga/Columbia/academic/bandits/results/evaluate_linearGaussianMixture_BanditSampling_{}/A={}/t_max={}/R={}/d_context={}/type_context={}/pi=0.5 0.5 0.5 0.5/theta=0. 0. 1. 1. 2. 2. 3. 3./sigma=1. 1. 1. 1.'.format(execution, A, t_max, R, d_context, type_context)
            # Prior_Ks
            prior_Ks=[1,2,3]
        elif scenario=='hard':
            A=2
            # Bandit dir
            bandit_dir='/home/iurteaga/Columbia/academic/bandits/results/evaluate_linearGaussianMixture_BanditSampling_{}/A={}/t_max={}/R={}/d_context={}/type_context={}/pi=0.5 0.5 0.3 0.7/theta=1. 1. 2. 2. 0. 0. 3. 3./sigma=1. 1. 1. 1.'.format(execution, A, t_max, R, d_context, type_context)
            # Prior_Ks
            prior_Ks=[1,2,3]
        elif scenario=='heavy':
            A=2
            # Bandit dir
            bandit_dir='/home/iurteaga/Columbia/academic/bandits/results/evaluate_linearGaussianMixture_BanditSampling_{}/A={}/t_max={}/R={}/d_context={}/type_context={}/pi=0.75 0.25 0.75 0.25/theta=0. 0. 0. 0. 2. 2. 2. 2./sigma=1. 10._1. 10.'.format(execution, A, t_max, R, d_context, type_context)
            # Prior_Ks
            prior_Ks=[1,2]
        elif scenario=='unbalanced':
            A=3
            # Bandit dir
            bandit_dir='/home/iurteaga/Columbia/academic/bandits/results/evaluate_linearGaussianMixture_BanditSampling_{}/A={}/t_max={}/R={}/d_context={}/type_context={}/pi=1._0._0._0.5 0.5 0._0.3 0.6 0.1/theta=1. 1. 2. 2. 3. 3. 1. 1. 2. 2. 3. 3. 0. 0. 3. 3. 4. 4./sigma=1. 1. 1. 1. 1. 1. 1. 1. 1.'.format(execution, A, t_max, R, d_context, type_context)
            # Prior_Ks
            prior_Ks=[1,2,3]
        else:
            raise ValueError('Unknown scenario {}'.format(scenario))

        nonparametric_R=0
        priorK_cumregrets={'mean':np.zeros((len(prior_Ks),t_max)), 'var':np.zeros((len(prior_Ks),t_max))}
        nonparametric_cumregrets={'mean':np.zeros(t_max), 'var':np.zeros(t_max)}
        priorK_reward_diff=np.zeros((len(prior_Ks),A,t_max))
        nonparametric_reward_diff=np.zeros((A,t_max))
        for (K_idx,prior_K) in enumerate(prior_Ks):
            print('Execution {} Scenario {} prior_K={}'.format(execution,scenario,prior_K))
            total_R=len(os.listdir('{}/prior_K={}/pi_expected'.format(bandit_dir,prior_K)))
            cumregrets=np.zeros((n_bandits,total_R,t_max))
            reward_diff=np.zeros((n_bandits,total_R,A,t_max))
            
            # Load each file
            for (r,bandit_file) in enumerate(os.listdir('{}/prior_K={}/pi_expected'.format(bandit_dir,prior_K))):
                try:
                    # Load bandits
                    f=open('{}/prior_K={}/pi_expected/{}'.format(bandit_dir,prior_K, bandit_file),'rb')
                    bandits=pickle.load(f)
                    
                    if execution=='MCMC':
                        #Cumregrets
                        cumregrets[0,r,:]=bandits[1].cumregrets
                        cumregrets[1,r,:]=bandits[2].cumregrets
                        # Reward diff
                        reward_diff[0,r,:,:]=(bandits[1].rewards_expected-bandits[1].true_expected_rewards)
                        reward_diff[1,r,:,:]=(bandits[2].rewards_expected-bandits[1].true_expected_rewards)
                    else:
                        #Cumregrets
                        cumregrets[0,r,:]=bandits[0].cumregrets
                        cumregrets[1,r,:]=bandits[1].cumregrets
                        # Reward diff
                        reward_diff[0,r,:,:]=(bandits[0].rewards_expected-bandits[1].true_expected_rewards)
                        reward_diff[1,r,:,:]=(bandits[1].rewards_expected-bandits[1].true_expected_rewards)
                    
                except:
                    print('NOT LOADED: {}'.format(bandit_file))


            # Plot dir
            os.makedirs('./figs/linearGaussianMixture/{}'.format(scenario), exist_ok=True)
            '''
            #### PLOTS
            bandits_colors=[colors.cnames['blue'], colors.cnames['red']]
            bandits_labels = ['K={} Mixture Model TS'.format(prior_K), 'Nonparametric TS']

            # Cumulative regret
            for n in np.arange(n_bandits):
                plt.plot(np.arange(t_max), cumregrets[n,:,:t_max].mean(axis=0), bandits_colors[n], label=bandits_labels[n])
                plt.fill_between(np.arange(t_max), cumregrets[n,:,:t_max].mean(axis=0)-np.sqrt(cumregrets[n,:,:t_max].var(axis=0)), cumregrets[n,:,:t_max].mean(axis=0)+np.sqrt(cumregrets[n,:,:t_max].var(axis=0)),alpha=0.35, facecolor=bandits_colors[n])
            plt.xlabel('t')
            plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
            plt.axis([0, t_max-1,0,plt.ylim()[1]])
            legend = plt.legend(loc='upper left', ncol=1, shadow=False)
            plt.savefig('./figs/linearGaussianMixture/{}/cumregret_priorK{}_{}_R{}.pdf'.format(scenario,prior_K,execution,total_R), format='pdf', bbox_inches='tight')
            plt.close()
            
            # Cumulative regret gain
            print('cumregret_{}_priorK{}_{}_R{}_reduction={}'.format(scenario,prior_K,execution,total_R,(cumregrets[0,:,-1].mean(axis=0)-cumregrets[1,:,-1].mean(axis=0))/cumregrets[0,:,-1].mean(axis=0)))

            # Reward squared Error
            t_plot=100
            for a in np.arange(A):
                plt.figure()
                for n in np.arange(n_bandits):
                    plt.plot(np.arange(t_plot), np.power(reward_diff[n,a,:,:t_plot].mean(axis=0),2), bandits_colors[n], label='{}, MSE={:.4f}'.format(bandits_labels[n], np.power(reward_diff[n,a,:,:t_plot].mean(axis=0),2).mean()))
                plt.xlabel('t')
                plt.ylabel(r'$(\mu_{a,t}-\hat{\mu}_{a,t} )^2$')
                plt.axis([0, t_plot-1,0,plt.ylim()[1]])
                legend = plt.legend(loc='upper right', ncol=1, shadow=False)
                plt.savefig('./figs/linearGaussianMixture/{}/mse_a_{}_priorK{}_{}_R{}.pdf'.format(scenario,a,prior_K,execution,total_R), format='pdf', bbox_inches='tight')
                plt.close()
            '''
            
            # Concatenate            
            # reward difference
            for a in np.arange(A):
                nonparametric_reward_diff[a,:t_max]=(nonparametric_R*nonparametric_reward_diff[a,:t_max]+total_R*reward_diff[1,a,:,:t_max].mean(axis=0))/(nonparametric_R+total_R)
                priorK_reward_diff[K_idx,a]=reward_diff[0,a,:,:t_max].mean(axis=0)

            # Cumulative regret
            nonparametric_cumregrets['mean'][:t_max], nonparametric_cumregrets['var'][:t_max], nonparametric_R = aggregate_statistics(nonparametric_cumregrets['mean'][:t_max], nonparametric_cumregrets['var'][:t_max], nonparametric_R, cumregrets[1,:,:t_max].mean(axis=0), cumregrets[1,:,:t_max].var(axis=0), total_R)
            priorK_cumregrets['mean'][K_idx]=cumregrets[0,:,:t_max].mean(axis=0)
            priorK_cumregrets['var'][K_idx]=cumregrets[0,:,:t_max].var(axis=0)

        #### PLOTS
        if scenario=='easy':
            # Cumulative regret
            bandits_colors=[colors.cnames['black'], colors.cnames['blue'], colors.cnames['green'], colors.cnames['red']]
            bandits_labels = ['K=1 Mixture Model TS', 'K=2 Oracle TS', 'K=3 Mixture Model TS', 'Nonparametric TS']
            n=1
            plt.plot(np.arange(t_max), priorK_cumregrets['mean'][n,:t_max], bandits_colors[n], label=bandits_labels[n])
            plt.fill_between(np.arange(t_max), priorK_cumregrets['mean'][n,:t_max]-np.sqrt(priorK_cumregrets['var'][n,:t_max]), priorK_cumregrets['mean'][n,:t_max]+np.sqrt(priorK_cumregrets['var'][n,:t_max]),alpha=0.35, facecolor=bandits_colors[n])
            plt.plot(np.arange(t_max), nonparametric_cumregrets['mean'][:t_max], bandits_colors[-1], label=bandits_labels[-1])
            plt.fill_between(np.arange(t_max), nonparametric_cumregrets['mean'][:t_max]-np.sqrt(nonparametric_cumregrets['var'][:t_max]), nonparametric_cumregrets['mean'][:t_max]+np.sqrt(nonparametric_cumregrets['var'][:t_max]),alpha=0.35, facecolor=bandits_colors[-1])
            plt.xlabel('t')
            plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
            plt.axis([0, t_max-1,0,plt.ylim()[1]])
            legend = plt.legend(loc='upper left', ncol=1, shadow=False)
            plt.savefig('./figs/linearGaussianMixture/{}/cumregret_R{}.pdf'.format(scenario,nonparametric_R), format='pdf', bbox_inches='tight')
            plt.close()
            
            # Cumulative regret gain
            print('cumregret_{}_R_{}_K2OracleTS_reduction={}'.format(scenario,nonparametric_R,(priorK_cumregrets['mean'][n,-1]-nonparametric_cumregrets['mean'][-1])/nonparametric_cumregrets['mean'][-1]))
            
        elif scenario=='hard':
            # Cumulative regret
            bandits_colors=[colors.cnames['black'], colors.cnames['blue'], colors.cnames['green'], colors.cnames['red']]
            bandits_labels = ['K=1 Mixture Model TS', 'K=2 Oracle TS', 'K=3 Mixture Model TS', 'Nonparametric TS']
            n=1
            plt.plot(np.arange(t_max), priorK_cumregrets['mean'][n,:t_max], bandits_colors[n], label=bandits_labels[n])
            plt.fill_between(np.arange(t_max), priorK_cumregrets['mean'][n,:t_max]-np.sqrt(priorK_cumregrets['var'][n,:t_max]), priorK_cumregrets['mean'][n,:t_max]+np.sqrt(priorK_cumregrets['var'][n,:t_max]),alpha=0.35, facecolor=bandits_colors[n])
            plt.plot(np.arange(t_max), nonparametric_cumregrets['mean'][:t_max], bandits_colors[-1], label=bandits_labels[-1])
            plt.fill_between(np.arange(t_max), nonparametric_cumregrets['mean'][:t_max]-np.sqrt(nonparametric_cumregrets['var'][:t_max]), nonparametric_cumregrets['mean'][:t_max]+np.sqrt(nonparametric_cumregrets['var'][:t_max]),alpha=0.35, facecolor=bandits_colors[-1])
            plt.xlabel('t')
            plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
            plt.axis([0, t_max-1,0,plt.ylim()[1]])
            legend = plt.legend(loc='upper left', ncol=1, shadow=False)
            plt.savefig('./figs/linearGaussianMixture/{}/cumregret_R{}.pdf'.format(scenario,nonparametric_R), format='pdf', bbox_inches='tight')
            plt.close()
            
            # Cumulative regret gain
            print('cumregret_{}_R_{}_K2OracleTS_reduction={}'.format(scenario,nonparametric_R,(priorK_cumregrets['mean'][n,-1]-nonparametric_cumregrets['mean'][-1])/nonparametric_cumregrets['mean'][-1]))
        elif scenario=='heavy':
            bandits_colors=[colors.cnames['blue'], colors.cnames['blue'], colors.cnames['red']]
            bandits_labels = ['K=1 Mispecified TS', 'K=2 Oracle TS', 'Nonparametric TS']
            # Cumulative regret for Oracle
            n=1
            plt.plot(np.arange(t_max), priorK_cumregrets['mean'][n,:t_max], bandits_colors[n], label=bandits_labels[n])
            plt.fill_between(np.arange(t_max), priorK_cumregrets['mean'][n,:t_max]-np.sqrt(priorK_cumregrets['var'][n,:t_max]), priorK_cumregrets['mean'][n,:t_max]+np.sqrt(priorK_cumregrets['var'][n,:t_max]),alpha=0.35, facecolor=bandits_colors[n])
            plt.plot(np.arange(t_max), nonparametric_cumregrets['mean'][:t_max], bandits_colors[-1], label=bandits_labels[-1])
            plt.fill_between(np.arange(t_max), nonparametric_cumregrets['mean'][:t_max]-np.sqrt(nonparametric_cumregrets['var'][:t_max]), nonparametric_cumregrets['mean'][:t_max]+np.sqrt(nonparametric_cumregrets['var'][:t_max]),alpha=0.35, facecolor=bandits_colors[-1])
            plt.xlabel('t')
            plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
            plt.axis([0, t_max-1,0,plt.ylim()[1]])
            legend = plt.legend(loc='upper left', ncol=1, shadow=False)
            plt.savefig('./figs/linearGaussianMixture/{}/cumregret_R{}.pdf'.format(scenario,nonparametric_R), format='pdf', bbox_inches='tight')
            plt.close()
            
            # Cumulative regret gain
            print('cumregret_{}_R_{}_K2OracleTS_reduction={}'.format(scenario,nonparametric_R,(priorK_cumregrets['mean'][n,-1]-nonparametric_cumregrets['mean'][-1])/nonparametric_cumregrets['mean'][-1]))
            
            # Cumulative regret for Mispecified
            n=0
            plt.plot(np.arange(t_max), priorK_cumregrets['mean'][n,:t_max], bandits_colors[n], label=bandits_labels[n])
            plt.fill_between(np.arange(t_max), priorK_cumregrets['mean'][n,:t_max]-np.sqrt(priorK_cumregrets['var'][n,:t_max]), priorK_cumregrets['mean'][n,:t_max]+np.sqrt(priorK_cumregrets['var'][n,:t_max]),alpha=0.35, facecolor=bandits_colors[n])
            plt.plot(np.arange(t_max), nonparametric_cumregrets['mean'][:t_max], bandits_colors[-1], label=bandits_labels[-1])
            plt.fill_between(np.arange(t_max), nonparametric_cumregrets['mean'][:t_max]-np.sqrt(nonparametric_cumregrets['var'][:t_max]), nonparametric_cumregrets['mean'][:t_max]+np.sqrt(nonparametric_cumregrets['var'][:t_max]),alpha=0.35, facecolor=bandits_colors[-1])
            plt.xlabel('t')
            plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
            plt.axis([0, t_max-1,0,plt.ylim()[1]])
            legend = plt.legend(loc='upper left', ncol=1, shadow=False)
            plt.savefig('./figs/linearGaussianMixture/{}/cumregret_R{}_mispecified.pdf'.format(scenario,nonparametric_R), format='pdf', bbox_inches='tight')
            plt.close()

            # Cumulative regret gain
            print('cumregret_{}_R_{}_K1MispecifiedTS_reduction={}'.format(scenario,nonparametric_R,(priorK_cumregrets['mean'][n,-1]-nonparametric_cumregrets['mean'][-1])/nonparametric_cumregrets['mean'][-1]))
        elif scenario=='unbalanced':
            # Cumulative regret
            bandits_colors=[colors.cnames['black'], colors.cnames['green'], colors.cnames['blue'], colors.cnames['red']]
            bandits_labels = ['K=1 Mixture Model TS', 'K=2 Mixture Model TS', 'K=3 Oracle TS', 'Nonparametric TS']
            n=2
            plt.plot(np.arange(t_max), priorK_cumregrets['mean'][n,:t_max], bandits_colors[n], label=bandits_labels[n])
            plt.fill_between(np.arange(t_max), priorK_cumregrets['mean'][n,:t_max]-np.sqrt(priorK_cumregrets['var'][n,:t_max]), priorK_cumregrets['mean'][n,:t_max]+np.sqrt(priorK_cumregrets['var'][n,:t_max]),alpha=0.35, facecolor=bandits_colors[n])
            plt.plot(np.arange(t_max), nonparametric_cumregrets['mean'][:t_max], bandits_colors[-1], label=bandits_labels[-1])
            plt.fill_between(np.arange(t_max), nonparametric_cumregrets['mean'][:t_max]-np.sqrt(nonparametric_cumregrets['var'][:t_max]), nonparametric_cumregrets['mean'][:t_max]+np.sqrt(nonparametric_cumregrets['var'][:t_max]),alpha=0.35, facecolor=bandits_colors[-1])
            plt.xlabel('t')
            plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
            plt.axis([0, t_max-1,0,plt.ylim()[1]])
            legend = plt.legend(loc='upper left', ncol=1, shadow=False)
            plt.savefig('./figs/linearGaussianMixture/{}/cumregret_R{}.pdf'.format(scenario,nonparametric_R), format='pdf', bbox_inches='tight')
            plt.close()
            
            # Cumulative regret gain
            print('cumregret_{}_R_{}_K3OracleTS_reduction={}'.format(scenario,nonparametric_R,(priorK_cumregrets['mean'][n,-1]-nonparametric_cumregrets['mean'][-1])/nonparametric_cumregrets['mean'][-1]))

        '''
        #### ALL PLOTS
        bandits_colors=[colors.cnames['black'], colors.cnames['blue'], colors.cnames['green'], colors.cnames['red']]
        bandits_labels = ['K=1 Mixture Model TS', 'K=2 Mixture Model TS', 'K=3 Mixture Model TS', 'Nonparametric TS']

        # Cumulative regret
        for n in np.arange(len(prior_Ks)):
            plt.plot(np.arange(t_max), priorK_cumregrets['mean'][n,:t_max], bandits_colors[n], label=bandits_labels[n])
            plt.fill_between(np.arange(t_max), priorK_cumregrets['mean'][n,:t_max]-np.sqrt(priorK_cumregrets['var'][n,:t_max]), priorK_cumregrets['mean'][n,:t_max]+np.sqrt(priorK_cumregrets['var'][n,:t_max]),alpha=0.35, facecolor=bandits_colors[n])
        plt.plot(np.arange(t_max), nonparametric_cumregrets['mean'][:t_max], bandits_colors[-1], label=bandits_labels[-1])
        plt.fill_between(np.arange(t_max), nonparametric_cumregrets['mean'][:t_max]-np.sqrt(nonparametric_cumregrets['var'][:t_max]), nonparametric_cumregrets['mean'][:t_max]+np.sqrt(nonparametric_cumregrets['var'][:t_max]),alpha=0.35, facecolor=bandits_colors[-1])
        plt.xlabel('t')
        plt.ylabel(r'$R_t=\sum_{t=0}^T \mu^*_t-\bar{y}_{t}$')
        plt.axis([0, t_max-1,0,plt.ylim()[1]])
        legend = plt.legend(loc='upper left', ncol=1, shadow=False)
        plt.savefig('./figs/linearGaussianMixture/{}/cumregret_allprior_{}_R{}.pdf'.format(scenario,execution,nonparametric_R), format='pdf', bbox_inches='tight')
        plt.close()


        # Reward squared Error
        t_plot=100
        for a in np.arange(A):
            plt.figure()
            for n in np.arange(len(prior_Ks)):
                plt.plot(np.arange(t_plot), np.power(priorK_reward_diff[n,a,:t_plot],2), bandits_colors[n], label='{}, MSE={:.4f}'.format(bandits_labels[n], np.power(priorK_reward_diff[n,a,:t_plot],2).mean()))
            plt.plot(np.arange(t_plot), np.power(nonparametric_reward_diff[a,:t_plot],2), bandits_colors[-1], label='{}, MSE={:.4f}'.format(bandits_labels[-1], np.power(nonparametric_reward_diff[a,:t_plot],2).mean()))
            plt.xlabel('t')
            plt.ylabel(r'$(\mu_{a,t}-\hat{\mu}_{a,t} )^2$')
            plt.axis([0, t_plot-1,0,plt.ylim()[1]])
            legend = plt.legend(loc='upper right', ncol=1, shadow=False)
            plt.savefig('./figs/linearGaussianMixture/{}/mse_a_{}_allprior_{}_R{}.pdf'.format(scenario,a,execution,nonparametric_R), format='pdf', bbox_inches='tight')
            plt.close()
        '''
    
################# Real data results, from showdown  #######################
# Oncoassign
experiment='oncoassign_t1000_gibbsmaxiter10_trainfreq1'
experiment_title='precision_oncology'
experiment_tmax=896 #We actually only have these many cases, so no need to plot further
# Load results
with open('{}/{}/R_results.npz'.format(bandit_showdown_dir, experiment), 'rb') as f:
    all_results=np.load(f, allow_pickle=True)    
    # Display results
    plot_cum_regret_all_to_optimal(all_results['opt_true_expected_rewards'][:,:experiment_tmax], all_results['rewards'][:,:,:experiment_tmax], all_results['labels'], './figs/{}/'.format(experiment_title), bandit_colors)
    plot_cum_rewards(all_results['rewards'][:,:,:experiment_tmax], all_results['labels'], './figs/{}/'.format(experiment_title), bandit_colors)
    rewards_table(all_results['opt_rewards'][:,:experiment_tmax], all_results['rewards'][:,:,:experiment_tmax], all_results['labels'], '{}'.format(experiment_title), bandit_colors)
    exec_time_analysis(all_results['times'], all_results['labels'], './figs/{}/'.format(experiment_title))

# Real data
experiment='yahoo_20090501_bandit_2_t25000_gibbsmaxiter10_trainfreq1'
experiment_title='yahoo_may_01'
with open('{}/{}/R_results.npz'.format(bandit_showdown_dir, experiment), 'rb') as f:
    all_results=np.load(f, allow_pickle=True)
    # Oracle: Linear TS
    linear_ts_idx=np.where(all_results['labels']=='LinFullPost')[0][0]
    nonparametric_ts_idx=np.where(all_results['labels']=='Nonparametric-TS')[0][0]
    non_oracle_idx=np.setdiff1d(np.arange(all_results['labels'].size), linear_ts_idx)
    
    # Display results
    plot_cum_regret_all_to_optimal(all_results['opt_true_expected_rewards'], all_results['rewards'], all_results['labels'], './figs/{}/'.format(experiment_title), bandit_colors)
    plot_cum_rewards(all_results['rewards'], all_results['labels'], './figs/{}/'.format(experiment_title), bandit_colors)
    rewards_table(all_results['opt_rewards'], all_results['rewards'], all_results['labels'], '{}'.format(experiment_title), bandit_colors)
    exec_time_analysis(all_results['times'], all_results['labels'], './figs/{}/'.format(experiment_title))

