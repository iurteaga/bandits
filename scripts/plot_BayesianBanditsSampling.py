#!/usr/bin/python

# Imports
import numpy as np
import pickle
import scipy.stats as stats
import scipy.linalg as linalg
import sys, os, re
import argparse
from itertools import *
import pdb
import matplotlib.pyplot as plt
from matplotlib import colors

# Add path and import Bayesian Bandits
sys.path.append('../src')
from BayesianBanditsSampling import *

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

# Main code
def main(script, A, t_max, R, exec_type, M, theta_min, theta_max, theta_diff):
    print('Ploting {}-armed bayesian bandit sampling script {} for {} time-instants and {} realizations'.format(script, A, t_max, R))

    # Save dir
    save_dir='./plots/{}/A={}/t_max={}/R={}/M={}'.format(script,A,t_max,R,M)
    os.makedirs(save_dir, exist_ok=True)

    # Performance variables
    # Plain diff and rel-diff
    min_theta_diff=np.array([])
    min_theta_reldiff=np.array([])
    # Kullback-Leibler divergence
    all_KL=np.array([])
    min_KL=np.array([])
    avg_KL=np.array([])
    
    # Lai-Robbins metric
    all_LR=np.array([])
    min_LR=np.array([])
    avg_LR=np.array([])
    
    # Bandits to evaluate
    TS_id=0
    bandits_evaluate=np.array([3,5,8]) # 3=log10invPFA_tGaussian_id, 5=loginvPFA_Markov_id, 8=loginvPFA_Chebyshev_id
    bandits_colors=[colors.cnames['red'], colors.cnames['blue'], colors.cnames['green']]
    # Relative regret difference with TS
    regret_reldiff={}
    for bIdx in bandits_evaluate:
        regret_reldiff[bIdx]=np.array([])
    
    # When to compute difference
    #Diff at the end
    t_diff=-1
    # Diff at 500
    t_diff=999
    
    # What to plot
    plot_cumregret=False
    t_plot=1000
        
    # for theta in combinations_with_replacement(np.arange(0.1,1.0,theta_diff),A):
    for theta in combinations(np.arange(theta_min,theta_max,theta_diff),A):
        # For each theta
        theta=np.array(theta).reshape(A,1)
        theta_diff=np.diff(theta.T).T

        # Directory to load
        dir_string='../results/{}/A={}/t_max={}/R={}/M={}/theta={}'.format(script, A, t_max, R, M, np.array_str(theta[:,0]))
        
        # Missing theta for A=3
        #if (not ((theta-np.array([[ 0.05], [ 0.35], [ 0.65]]))<np.power(10., -15.)).all()) and (not ((theta-np.array([[ 0.05], [ 0.4], [ 0.9]]))<np.power(10., -15.)).all()) and (not ((theta-np.array([[ 0.05], [ 0.45], [ 0.85]]))<np.power(10., -15.)).all()):
        
        # Pickle load
        with open(dir_string+'/bandits.pickle', 'rb') as f, open(dir_string+'/bandits_labels.pickle', 'rb') as g:
            bandits = pickle.load(f)
            bandits_labels = pickle.load(g)
            
            # Differences
            min_theta_diff=np.append(min_theta_diff, theta_diff.min())
            min_theta_reldiff=np.append(min_theta_reldiff, theta_diff.min()/theta[-1])
            
            # KL divergence
            per_arm_KL=theta[:-1]*np.log(theta[:-1]/theta[-1])+(1-theta[:-1])*np.log((1-theta[:-1])/(1-theta[-1]))
            all_KL=np.append(all_KL, per_arm_KL.sum())
            min_KL=np.append(min_KL, per_arm_KL.min())
            avg_KL=np.append(avg_KL, per_arm_KL.mean())
            
            # Lai-Robbins
            all_LR=np.append(all_LR, (theta_diff/per_arm_KL).sum())
            min_LR=np.append(min_LR, (theta_diff/per_arm_KL).min())
            avg_LR=np.append(avg_LR, (theta_diff/per_arm_KL).mean())

            print('theta={}, min_theta_diff={}, min_theta_reldiff={}, all_KL={}, min_KL={}, all_LR={}, min_LR={}'.format(np.array_str(theta[:,0]), min_theta_diff[-1], min_theta_reldiff[-1], all_KL[-1], min_KL[-1], all_LR[-1], min_KL[-1]))
                           
            # TS Cumulative regret
            TS_cumregret=(theta.max()-bandits[TS_id].returns_R['mean'][0,:]).cumsum()
            TS_cumregret_sigma=np.sqrt(bandits[TS_id].returns_R['var'][0,:])

            # If plotting cumulative regret
            if plot_cumregret:
                os.makedirs(save_dir+'/theta={}'.format(theta[:,0]), exist_ok=True)
                # TS Cumregret
                plt.figure()
                plt.plot(np.arange(t_plot), TS_cumregret, 'k', label=bandits_labels[TS_id])
            # For each bandit                
            for idx, bIdx in enumerate(bandits_evaluate):
                # Cumulative regret            
                this_cumregret=(theta.max()-bandits[bIdx].returns_R['mean'][0,:]).cumsum()
                this_cumregret_sigma=np.sqrt(bandits[bIdx].returns_R['var'][0,:])            
                # Relative regret at t_diff
                regret_reldiff[bIdx]=np.append(regret_reldiff[bIdx], (this_cumregret[t_diff]-TS_cumregret[t_diff])/TS_cumregret[t_diff])
                
                if plot_cumregret:
                    # Plot
                    plt.plot(np.arange(t_plot), this_cumregret, bandits_colors[idx], label=bandits_labels[bIdx])
            
            if plot_cumregret:
                # Finalize plot
                plt.xlabel('t')
                plt.ylabel(r'$L_t=\sum_{t=0}^T y_t^*-y_t$')
                plt.title('Cumulative regret over time')
                plt.xlim([0, t_plot-1])
                legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
                plt.savefig(save_dir+'/theta={}/regret_cumulative.pdf'.format(theta[:,0]), format='pdf', bbox_inches='tight')
                plt.close()
            
    #### Sort and plot
    # Min theta diff vs regret reldiff
    min_theta_diff_idx=np.argsort(min_theta_diff, kind='mergesort') # Mergesort because is stable!
    plt.figure()
    # For each bandit                
    for idx, bIdx in enumerate(bandits_evaluate):
        plt.scatter(min_theta_diff[min_theta_diff_idx], regret_reldiff[bIdx][min_theta_diff_idx], color=bandits_colors[idx], label=bandits_labels[bIdx])
        '''
        markerline, stemlines, baseline=plt.stem(min_theta_diff[min_theta_diff_idx], regret_reldiff[bIdx][min_theta_diff_idx], label=bandits_labels[bIdx])
        plt.setp(stemlines, linewidth=0)
        plt.setp(markerline, 'markerfacecolor', bandits_colors[idx])
        plt.setp(baseline, 'color', 'k')
        '''
        
    plt.xlabel('Min theta diff')
    plt.ylabel('Cumulative Regret relative difference')
    plt.xlim([min(min_theta_diff), max(min_theta_diff)])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    plt.savefig(save_dir+'/min_theta_diff_relDiff_at_{}.pdf'.format(t_diff), format='pdf', bbox_inches='tight')
    plt.close()

    # Min theta rel diff vs regret reldiff
    min_theta_reldiff_idx=np.argsort(min_theta_reldiff, kind='mergesort') # Mergesort because is stable!
    plt.figure()
    # For each bandit                
    for idx, bIdx in enumerate(bandits_evaluate):
        plt.scatter(min_theta_reldiff[min_theta_reldiff_idx], regret_reldiff[bIdx][min_theta_reldiff_idx], color=bandits_colors[idx], label=bandits_labels[bIdx])
        '''
        markerline, stemlines, baseline=plt.stem(min_theta_reldiff[min_theta_reldiff_idx], regret_reldiff[bIdx][min_theta_reldiff_idx], label=bandits_labels[bIdx])
        plt.setp(stemlines, linewidth=0)
        plt.setp(markerline, 'markerfacecolor', bandits_colors[idx])
        plt.setp(baseline, 'color', 'k')
        '''
        
    plt.xlabel('Min theta rel diff')
    plt.ylabel('Cumulative Regret relative difference')
    plt.xlim([min(min_theta_reldiff), max(min_theta_reldiff)])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)    
    plt.savefig(save_dir+'/min_theta_reldiff_relDiff_at_{}.pdf'.format(t_diff), format='pdf', bbox_inches='tight')
    plt.close()
    
    # Total KL vs regret reldiff
    all_KL_idx=np.argsort(all_KL, kind='mergesort') # Mergesort because is stable!
    plt.figure()
    # For each bandit                
    for idx, bIdx in enumerate(bandits_evaluate):
        plt.scatter(all_KL[all_KL_idx], regret_reldiff[bIdx][all_KL_idx], color=bandits_colors[idx], label=bandits_labels[bIdx])
        '''
        markerline, stemlines, baseline=plt.stem(all_KL[all_KL_idx], regret_reldiff[bIdx][all_KL_idx], label=bandits_labels[bIdx])
        plt.setp(stemlines, linewidth=0)
        plt.setp(markerline, 'markerfacecolor', bandits_colors[idx])
        plt.setp(baseline, 'color', 'k')
        '''
        
    plt.xlabel('Total KL')
    plt.ylabel('Cumulative Regret relative difference')
    plt.xlim([min(all_KL), max(all_KL)])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)    
    plt.savefig(save_dir+'/all_KL_relDiff_at_{}.pdf'.format(t_diff), format='pdf', bbox_inches='tight')
    plt.close()

    # Min KL vs regret reldiff
    min_KL_idx=np.argsort(min_KL, kind='mergesort') # Mergesort because is stable!
    plt.figure()
    # For each bandit                
    for idx, bIdx in enumerate(bandits_evaluate):
        plt.scatter(min_KL[min_KL_idx], regret_reldiff[bIdx][min_KL_idx], color=bandits_colors[idx], label=bandits_labels[bIdx])
        '''
        markerline, stemlines, baseline=plt.stem(min_KL[min_KL_idx], regret_reldiff[bIdx][min_KL_idx], label=bandits_labels[bIdx])
        plt.setp(stemlines, linewidth=0)
        plt.setp(markerline, 'markerfacecolor', bandits_colors[idx])
        plt.setp(baseline, 'color', 'k')
        '''
        
    plt.xlabel('Min KL')
    plt.ylabel('Cumulative Regret relative difference')
    plt.xlim([min(min_KL), max(min_KL)])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)    
    plt.savefig(save_dir+'/min_KL_relDiff_at_{}.pdf'.format(t_diff), format='pdf', bbox_inches='tight')
    plt.close()

    # Average KL vs regret reldiff
    avg_KL_idx=np.argsort(avg_KL, kind='mergesort') # Mergesort because is stable!
    plt.figure()
    # For each bandit                
    for idx, bIdx in enumerate(bandits_evaluate):
        plt.scatter(avg_KL[avg_KL_idx], regret_reldiff[bIdx][avg_KL_idx], color=bandits_colors[idx], label=bandits_labels[bIdx])
        '''
        markerline, stemlines, baseline=plt.stem(avg_KL[avg_KL_idx], regret_reldiff[bIdx][avg_KL_idx], label=bandits_labels[bIdx])
        plt.setp(stemlines, linewidth=0)
        plt.setp(markerline, 'markerfacecolor', bandits_colors[idx])
        plt.setp(baseline, 'color', 'k')
        '''
        
    plt.xlabel('Average KL')
    plt.ylabel('Cumulative Regret relative difference')
    plt.xlim([min(avg_KL), max(avg_KL)])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)    
    plt.savefig(save_dir+'/avg_KL_relDiff_at_{}.pdf'.format(t_diff), format='pdf', bbox_inches='tight')
    plt.close()
    
    # Total LR vs regret reldiff
    all_LR_idx=np.argsort(all_LR, kind='mergesort') # Mergesort because is stable!
    plt.figure()
    # For each bandit                
    for idx, bIdx in enumerate(bandits_evaluate):
        plt.scatter(all_LR[all_LR_idx], regret_reldiff[bIdx][all_LR_idx], color=bandits_colors[idx], label=bandits_labels[bIdx])
        '''
        markerline, stemlines, baseline=plt.stem(all_LR[all_LR_idx], regret_reldiff[bIdx][all_LR_idx], label=bandits_labels[bIdx])
        plt.setp(stemlines, linewidth=0)
        plt.setp(markerline, 'markerfacecolor', bandits_colors[idx])
        plt.setp(baseline, 'color', 'k')
        '''
    plt.xlabel('Total LR')
    plt.ylabel('Cumulative Regret relative difference')
    plt.xlim([min(all_LR), max(all_LR)])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)    
    plt.savefig(save_dir+'/all_LR_relDiff_at_{}.pdf'.format(t_diff), format='pdf', bbox_inches='tight')
    plt.close()

    # Min LR vs regret reldiff
    min_LR_idx=np.argsort(min_LR, kind='mergesort') # Mergesort because is stable!
    plt.figure()
    # For each bandit                
    for idx, bIdx in enumerate(bandits_evaluate):
        plt.scatter(min_LR[min_LR_idx], regret_reldiff[bIdx][min_LR_idx], color=bandits_colors[idx], label=bandits_labels[bIdx])
        '''
        markerline, stemlines, baseline=plt.stem(min_LR[min_LR_idx], regret_reldiff[bIdx][min_LR_idx], label=bandits_labels[bIdx])
        plt.setp(stemlines, linewidth=0)
        plt.setp(markerline, 'markerfacecolor', bandits_colors[idx])
        plt.setp(baseline, 'color', 'k')
        '''
        
    plt.xlabel('Min LR')
    plt.ylabel('Cumulative Regret relative difference')
    plt.xlim([min(min_LR), max(min_LR)])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)    
    plt.savefig(save_dir+'/min_LR_relDiff_at_{}.pdf'.format(t_diff), format='pdf', bbox_inches='tight')
    plt.close()

    # Average LR vs regret reldiff
    avg_LR_idx=np.argsort(avg_LR, kind='mergesort') # Mergesort because is stable!
    plt.figure()
    # For each bandit                
    for idx, bIdx in enumerate(bandits_evaluate):
        plt.scatter(avg_LR[avg_LR_idx], regret_reldiff[bIdx][avg_LR_idx], color=bandits_colors[idx], label=bandits_labels[bIdx])
        '''
        markerline, stemlines, baseline=plt.stem(avg_LR[avg_LR_idx], regret_reldiff[bIdx][avg_LR_idx], label=bandits_labels[bIdx])
        plt.setp(stemlines, linewidth=0)
        plt.setp(markerline, 'markerfacecolor', bandits_colors[idx])
        plt.setp(baseline, 'color', 'k')
        '''
        
    plt.xlabel('Average LR')
    plt.ylabel('Cumulative Regret relative difference')
    plt.xlim([min(avg_LR), max(avg_LR)])
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)    
    plt.savefig(save_dir+'/avg_LR_relDiff_at_{}.pdf'.format(t_diff), format='pdf', bbox_inches='tight')
    plt.close()
        
    pdb.set_trace()
        
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Evaluation plots for executed Bayesian bandits.')
    parser.add_argument('-script', type=str, default='evaluate_BayesianBanditsSampling_dynamic', help='Execution script used to create data to plot')
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
    main(args.script, args.A, args.t_max, args.R, args.exec_type, args.M, args.theta_min, args.theta_max, args.theta_diff)
