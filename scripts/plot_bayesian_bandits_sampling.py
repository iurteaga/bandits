#!/usr/bin/python

# Imports
import numpy as np
import pickle
import scipy.stats as stats
import sys, os, re
import argparse
from itertools import *
import pdb

# Add path and import Bayesian Bandits
sys.path.append('../src')
from BayesianBanditsSampling import *
from plot_bandits import *

# Main code
def main(A, t_max, R, exec_type, M, theta_min, theta_max, theta_diff):
    print('Ploting for bayesian {}-armed bayesian bandit with for {} time-instants and {} realizations'.format(A, t_max, R))

    # Save dir
    save_dir='./plots/A={}/t_max={}/R={}/M={}'.format(A,t_max,R,M)
    os.makedirs(save_dir, exist_ok=True)

    # Variables
    kl_div=np.array([])
    lai_robbins=np.array([])
    
    # Difference
    log10invPFA_tGaussian_diff=np.array([])
    loginvPFA_Markov_diff=np.array([])
    loginvPFA_Chebyshev_diff=np.array([])
    # Relative difference
    log10invPFA_tGaussian_reldiff=np.array([])
    loginvPFA_Markov_reldiff=np.array([])
    loginvPFA_Chebyshev_reldiff=np.array([])
    
    # When to compute difference
    #Diff at the end
    diff_id=-1
    # Diff at 500
    diff_id=500
    diff_id=1499
        
    # for theta in combinations_with_replacement(np.arange(0.1,1.0,theta_diff),A):
    for theta in combinations(np.arange(theta_min,theta_max,theta_diff),A):
        # For each theta (in string format)
        theta=np.array(theta).reshape(A,1)
        theta_diff=np.diff(theta.T).T
        kl_div_per_arm=theta[:-1]*np.log(theta[:-1]/theta[-1])+(1-theta[:-1])*np.log((1-theta[:-1])/(1-theta[-1]))
        kl_div=np.append(kl_div, kl_div_per_arm.sum())
        lai_robbins=np.append(lai_robbins, (theta_diff/kl_div_per_arm).sum())

        
        # Directory configuration
        run_script='evaluate_bayesian_bandits_sampling_dynamic'
        dir_string='../results/{}/A={}/t_max={}/R={}/M={}/theta={}'.format(run_script, A, t_max, R, M, np.array_str(theta[:,0]))

        print('theta={}, Dl={}, LR={}'.format(np.array_str(theta[:,0]), kl_div[-1], lai_robbins[-1]))
        
        # Pickle load
        with open(dir_string+'/bandits.pickle', 'rb') as f:
            bandits = pickle.load(f)
        with open(dir_string+'/bandits_labels.pickle', 'rb') as f:
            bandits_labels = pickle.load(f)
            
        # Interesting bandits
        TS_id=0
        bandit_log10invPFA_tGaussian_id=3
        bandit_loginvPFA_Markov_id=5
        bandit_loginvPFA_Chebyshev_id=8

        # Cumulative regret
        TS_cumregret=(theta.max()-bandits[TS_id].returns_R['mean'][0,0:t_max]).cumsum()
        TS_cumregret_sigma=np.sqrt(bandits[TS_id].returns_R['var'][0,0:t_max])
        if exec_type == 'general':
            TS_cumregret_all=(theta.max()-bandits[TS_id].returns_R['all']).cumsum(axis=2)
            TS_actions_correct=(bandits[TS_id].actions_R['all'][:,theta.argmax(),:].sum(axis=0))/R

        bandit_log10invPFA_tGaussian_cumregret=(theta.max()-bandits[bandit_log10invPFA_tGaussian_id].returns_R['mean'][0,0:t_max]).cumsum()
        bandit_log10invPFA_tGaussian_cumregret_sigma=np.sqrt(bandits[bandit_log10invPFA_tGaussian_id].returns_R['var'][0,0:t_max])        
        bandit_log10invPFA_tGaussian_diff=bandit_log10invPFA_tGaussian_cumregret-TS_cumregret
        bandit_log10invPFA_tGaussian_reldiff=bandit_log10invPFA_tGaussian_diff/TS_cumregret
        if exec_type == 'general':
            bandit_log10invPFA_tGaussian_cumregret_all=(theta.max()-bandits[bandit_log10invPFA_tGaussian_id].returns_R['all']).cumsum(axis=2)
            bandit_log10invPFA_tGaussian_actions_correct=(bandits[bandit_log10invPFA_tGaussian_id].actions_R['all'][:,theta.argmax(),:].sum(axis=0))/R
            
        bandit_loginvPFA_Markov_cumregret=(theta.max()-bandits[bandit_loginvPFA_Markov_id].returns_R['mean'][0,0:t_max]).cumsum()
        bandit_loginvPFA_Markov_cumregret_sigma=np.sqrt(bandits[bandit_loginvPFA_Markov_id].returns_R['var'][0,0:t_max])
        bandit_loginvPFA_Markov_diff=bandit_loginvPFA_Markov_cumregret-TS_cumregret
        bandit_loginvPFA_Markov_reldiff=bandit_loginvPFA_Markov_diff/TS_cumregret
        if exec_type == 'general':
            bandit_loginvPFA_Markov_cumregret_all=(theta.max()-bandits[bandit_loginvPFA_Markov_id].returns_R['all']).cumsum(axis=2)
            bandit_loginvPFA_Markov_actions_correct=(bandits[bandit_loginvPFA_Markov_id].actions_R['all'][:,theta.argmax(),:].sum(axis=0))/R

        bandit_loginvPFA_Chebyshev_cumregret=(theta.max()-bandits[bandit_loginvPFA_Chebyshev_id].returns_R['mean'][0,0:t_max]).cumsum()
        bandit_loginvPFA_Chebyshev_cumregret_sigma=np.sqrt(bandits[bandit_loginvPFA_Chebyshev_id].returns_R['var'][0,0:t_max])
        bandit_loginvPFA_Chebyshev_diff=bandit_loginvPFA_Chebyshev_cumregret-TS_cumregret
        bandit_loginvPFA_Chebyshev_reldiff=bandit_loginvPFA_Chebyshev_diff/TS_cumregret
        if exec_type == 'general':
            bandit_loginvPFA_Chebyshev_cumregret_all=(theta.max()-bandits[bandit_loginvPFA_Chebyshev_id].returns_R['all']).cumsum(axis=2)
            bandit_loginvPFA_Chebyshev_actions_correct=(bandits[bandit_loginvPFA_Chebyshev_id].actions_R['all'][:,theta.argmax(),:].sum(axis=0))/R
        
        # Plot
        os.makedirs(save_dir+'/theta={}'.format(theta[:,0]), exist_ok=True)
        # Cumregret
        plt.figure()
        plt.plot(np.arange(t_max), TS_cumregret, 'k', label=bandits_labels[TS_id])
        plt.plot(np.arange(t_max), bandit_log10invPFA_tGaussian_cumregret, 'b', label=bandits_labels[bandit_log10invPFA_tGaussian_id])
        plt.plot(np.arange(t_max), bandit_loginvPFA_Markov_cumregret, 'g', label=bandits_labels[bandit_loginvPFA_Markov_id])
        plt.plot(np.arange(t_max), bandit_loginvPFA_Chebyshev_cumregret, 'r', label=bandits_labels[bandit_loginvPFA_Chebyshev_id])
        plt.xlabel('t')
        plt.ylabel(r'$L_t=\sum_{t=0}^T y_t^*-y_t$')
        plt.title('Cumulative regret over time')
        plt.xlim([0, t_max-1])
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        plt.savefig(save_dir+'/theta={}/regret_cumulative.pdf'.format(theta[:,0]), format='pdf', bbox_inches='tight')
        plt.close()
        
        # Cumregret +- sigma
        plt.figure()
        plt.plot(np.arange(t_max), TS_cumregret, 'k', label=bandits_labels[TS_id])
        plt.fill_between(np.arange(t_max), TS_cumregret-TS_cumregret_sigma, TS_cumregret+TS_cumregret_sigma,alpha=0.4, facecolor='k')
        plt.plot(np.arange(t_max), bandit_log10invPFA_tGaussian_cumregret, 'b', label=bandits_labels[bandit_log10invPFA_tGaussian_id])
        plt.fill_between(np.arange(t_max), bandit_log10invPFA_tGaussian_cumregret-bandit_log10invPFA_tGaussian_cumregret_sigma, bandit_log10invPFA_tGaussian_cumregret+bandit_log10invPFA_tGaussian_cumregret_sigma,alpha=0.4, facecolor='b')
        plt.plot(np.arange(t_max), bandit_loginvPFA_Markov_cumregret, 'g', label=bandits_labels[bandit_loginvPFA_Markov_id])
        plt.fill_between(np.arange(t_max), bandit_loginvPFA_Markov_cumregret-bandit_loginvPFA_Markov_cumregret_sigma, bandit_loginvPFA_Markov_cumregret+bandit_loginvPFA_Markov_cumregret_sigma,alpha=0.4, facecolor='g')
        plt.plot(np.arange(t_max), bandit_loginvPFA_Chebyshev_cumregret, 'r', label=bandits_labels[bandit_loginvPFA_Chebyshev_id])
        plt.fill_between(np.arange(t_max), bandit_loginvPFA_Chebyshev_cumregret-bandit_loginvPFA_Chebyshev_cumregret_sigma, bandit_loginvPFA_Chebyshev_cumregret+bandit_loginvPFA_Chebyshev_cumregret_sigma,alpha=0.4, facecolor='r')
        plt.xlabel('t')
        plt.ylabel(r'$L_t=\sum_{t=0}^T y_t^*-y_t$')
        plt.title('Cumulative regret over time')
        plt.xlim([0, t_max-1])
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        plt.savefig(save_dir+'/theta={}/regret_cumulative_sigma.pdf'.format(theta[:,0]), format='pdf', bbox_inches='tight')
        plt.close()
        
        if exec_type == 'general':
            t_hist=10
            # Cumregret histograms
            plt.figure()
            plt.hist(TS_cumregret_all[:,0,t_hist], normed=1, color='k', label=bandits_labels[TS_id])
            plt.hist(bandit_log10invPFA_tGaussian_cumregret_all[:,0,t_hist], normed=1, color='b', label=bandits_labels[bandit_log10invPFA_tGaussian_id])
            plt.hist(bandit_loginvPFA_Markov_cumregret_all[:,0,t_hist], normed=1, color='g', label=bandits_labels[bandit_loginvPFA_Markov_id])
            plt.hist(bandit_loginvPFA_Chebyshev_cumregret_all[:,0,t_hist], normed=1, color='r', label=bandits_labels[bandit_loginvPFA_Chebyshev_id])
            plt.xlabel(r'$L_t=\sum_{t=0}^T y_t^*-y_t$')
            plt.ylabel('pdf')
            plt.title('Cumulative regret histogram at t={}'.format(t_hist))
            legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
            plt.savefig(save_dir+'/theta={}/regret_cumulative_hist_{}.pdf'.format(theta[:,0], t_hist), format='pdf', bbox_inches='tight')
            plt.close()
            
            # Correct actions
            plt.figure()
            plt.plot(np.arange(t_max), TS_actions_correct, 'k', label=bandits_labels[TS_id])
            plt.plot(np.arange(t_max), bandit_log10invPFA_tGaussian_actions_correct, 'b', label=bandits_labels[bandit_log10invPFA_tGaussian_id])
            plt.plot(np.arange(t_max), bandit_loginvPFA_Markov_actions_correct, 'g', label=bandits_labels[bandit_loginvPFA_Markov_id])
            plt.plot(np.arange(t_max), bandit_loginvPFA_Chebyshev_actions_correct, 'r', label=bandits_labels[bandit_loginvPFA_Chebyshev_id])
            plt.xlabel('t')
            plt.ylabel(r'$P(a=a^*)$')
            plt.title('Correct action played over time')
            plt.axis([0, t_max-1,0,1])
            legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
            plt.savefig(save_dir+'/theta={}/correct_actions.pdf'.format(theta[:,0]), format='pdf', bbox_inches='tight')
            plt.close()
        
        # Difference
        log10invPFA_tGaussian_diff=np.append(log10invPFA_tGaussian_diff, bandit_log10invPFA_tGaussian_diff[diff_id])
        log10invPFA_tGaussian_reldiff=np.append(log10invPFA_tGaussian_reldiff, bandit_log10invPFA_tGaussian_reldiff[diff_id])
        loginvPFA_Markov_diff=np.append(loginvPFA_Markov_diff, bandit_loginvPFA_Markov_diff[diff_id])
        loginvPFA_Markov_reldiff=np.append(loginvPFA_Markov_reldiff, bandit_loginvPFA_Markov_reldiff[diff_id])
        loginvPFA_Chebyshev_diff=np.append(loginvPFA_Chebyshev_diff, bandit_loginvPFA_Chebyshev_diff[diff_id])
        loginvPFA_Chebyshev_reldiff=np.append(loginvPFA_Chebyshev_reldiff, bandit_loginvPFA_Chebyshev_reldiff[diff_id])
        
    # Plot differences
    kl_div_sortIdx=np.argsort(kl_div)
    lai_robbins_sortIdx=np.argsort(lai_robbins)

    # KL div vs regret diff
    plt.figure()
    plt.plot(kl_div[kl_div_sortIdx], log10invPFA_tGaussian_diff[kl_div_sortIdx], 'b', label='tGaussian n=log10(1/Pfa), M=1000')
    plt.plot(kl_div[kl_div_sortIdx], loginvPFA_Markov_diff[kl_div_sortIdx], 'g', label='Markov n=ln(1/Pfa), M=1000')
    plt.plot(kl_div[kl_div_sortIdx], loginvPFA_Chebyshev_diff[kl_div_sortIdx], 'r', label='Chebyshev n=ln(1/Pfa), M=1000')
    plt.xlabel('KL divergence')
    plt.ylabel('Regret diff with TS')
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)    
    plt.savefig(save_dir+'/klDiv_regretDiff_at_{}.pdf'.format(diff_id), format='pdf', bbox_inches='tight')
    plt.close()
    
    # KL div vs regret reldiff
    plt.figure()
    plt.plot(kl_div[kl_div_sortIdx], log10invPFA_tGaussian_reldiff[kl_div_sortIdx], 'b', label='tGaussian n=log10(1/Pfa), M=1000')
    plt.plot(kl_div[kl_div_sortIdx], loginvPFA_Markov_reldiff[kl_div_sortIdx], 'g', label='Markov n=ln(1/Pfa), M=1000')
    plt.plot(kl_div[kl_div_sortIdx], loginvPFA_Chebyshev_reldiff[kl_div_sortIdx], 'r', label='Chebyshev n=ln(1/Pfa), M=1000')
    plt.xlabel('KL divergence')
    plt.ylabel('Regret relative diff with TS')
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)    
    plt.savefig(save_dir+'/klDiv_regretRelDiff_at_{}.pdf'.format(diff_id), format='pdf', bbox_inches='tight')
    plt.close()

    '''
    # Lai Robbins vs regret diff
    plt.figure()
    plt.plot(lai_robbins[lai_robbins_sortIdx], log10invPFA_tGaussian_diff[lai_robbins_sortIdx], 'b', label='tGaussian n=log10(1/Pfa), M=1000')
    plt.plot(lai_robbins[lai_robbins_sortIdx], loginvPFA_Markov_diff[lai_robbins_sortIdx], 'g', label='Markov n=ln(1/Pfa), M=1000')
    plt.plot(lai_robbins[lai_robbins_sortIdx], loginvPFA_Chebyshev_diff[lai_robbins_sortIdx], 'r', label='Chebyshev n=ln(1/Pfa), M=1000')
    plt.xlabel('lai_robbins')
    plt.ylabel('Regret diff with TS')
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)    
    plt.savefig(save_dir+'/lai_robbins_regretDiff_at_{}.pdf'.format(diff_id), format='pdf', bbox_inches='tight')
    plt.close()
    
    # Lai Robbins vs regret reldiff
    plt.figure()
    plt.plot(lai_robbins[lai_robbins_sortIdx], log10invPFA_tGaussian_reldiff[lai_robbins_sortIdx], 'b', label='tGaussian n=log10(1/Pfa), M=1000')
    plt.plot(lai_robbins[lai_robbins_sortIdx], loginvPFA_Markov_reldiff[lai_robbins_sortIdx], 'g', label='Markov n=ln(1/Pfa), M=1000')
    plt.plot(lai_robbins[lai_robbins_sortIdx], loginvPFA_Chebyshev_reldiff[lai_robbins_sortIdx], 'r', label='Chebyshev n=ln(1/Pfa), M=1000')
    plt.xlabel('lai_robbins')
    plt.ylabel('Regret relative diff with TS')
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)    
    plt.savefig(save_dir+'/lai_robbins_regretRelDiff_at_{}.pdf'.format(diff_id), format='pdf', bbox_inches='tight')
    plt.close()
    '''
        
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Evaluation plots for executed Bayesian bandits.')
    parser.add_argument('-A', type=int, default=2, help='Number of arms of the bandit')
    parser.add_argument('-t_max', type=int, default=10, help='Time-instants to run the bandit')
    parser.add_argument('-R', type=int, default=1, help='Number of realizations to run')
    parser.add_argument('-exec_type', type=str, default='online', help='Type of execution to run: online or all')
    parser.add_argument('-M', type=int, default=0, help='Number of MC samples used')
    parser.add_argument('-theta_min', type=float, default=0, help='Minimum theta')
    parser.add_argument('-theta_max', type=float, default=1, help='Maximum theta')
    parser.add_argument('-theta_diff', type=float, default=0.5, help='Differences for theta')

    # Get arguments
    args = parser.parse_args()
    
    # Call main function
    main(args.A, args.t_max, args.R, args.exec_type, args.M, args.theta_min, args.theta_max, args.theta_diff)
