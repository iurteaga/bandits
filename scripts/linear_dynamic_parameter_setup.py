#!/usr/bin/python

# Imports
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle

# Dimensionalities
A=2
d_context=2
t_max=1500

'''
# 0/1 separated
C_sigma=0.000000001
dynamic_parameters={'type':'zero_one_separated', 'dynamics_A':np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'dynamics_C':np.ones(A)[:,None,None]*C_sigma*np.eye(d_context)[None,:,:]}

theta_0=np.zeros((A, d_context))
theta_0[1]=np.ones(d_context)
dynamic_parameters['theta']=np.zeros((A,d_context,t_max))
dynamic_parameters['theta'][:,:,0]=theta_0
for t in np.arange(t_max-1):
    dynamic_parameters['theta'][:,:,t+1]=np.einsum('adb,ab->ad',dynamic_parameters['dynamics_A'],dynamic_parameters['theta'][:,:,t]) + np.einsum('adb,ab->ad',np.linalg.cholesky(dynamic_parameters['dynamics_C']),stats.norm.rvs(size=(A,d_context)))

for a in np.arange(A):
    plt.plot(dynamic_parameters['theta'][a,0,:], 'b', dynamic_parameters['theta'][a,1,:],'r'), plt.show()

# Save
with open('./dynamic_parameters_{}_A{}_dcontext{}_tmax{}.pickle'.format(dynamic_parameters['type'],A,d_context,t_max), 'wb') as f:
    pickle.dump(dynamic_parameters, f)
    
# -1/1 separated
C_sigma=0.000000001
dynamic_parameters={'type':'minusone_one_separated', 'dynamics_A':np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'dynamics_C':np.ones(A)[:,None,None]*C_sigma*np.eye(d_context)[None,:,:]}

theta_0=np.zeros((A, d_context))
theta_0[0]=-np.ones(d_context)
theta_0[1]=np.ones(d_context)
dynamic_parameters['theta']=np.zeros((A,d_context,t_max))
dynamic_parameters['theta'][:,:,0]=theta_0
for t in np.arange(t_max-1):
    dynamic_parameters['theta'][:,:,t+1]=np.einsum('adb,ab->ad',dynamic_parameters['dynamics_A'],dynamic_parameters['theta'][:,:,t]) + np.einsum('adb,ab->ad',np.linalg.cholesky(dynamic_parameters['dynamics_C']),stats.norm.rvs(size=(A,d_context)))

for a in np.arange(A):
    plt.plot(dynamic_parameters['theta'][a,0,:], 'b', dynamic_parameters['theta'][a,1,:],'r'), plt.show()

# Save
with open('./dynamic_parameters_{}_A{}_dcontext{}_tmax{}.pickle'.format(dynamic_parameters['type'],A,d_context,t_max), 'wb') as f:
    pickle.dump(dynamic_parameters, f)
'''

# Hand-crafted symmetric mixing
C_sigma=0.01
dynamic_parameters={'type':'symmetric_mixing', 'dynamics_A':np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'dynamics_C':np.ones(A)[:,None,None]*C_sigma*np.eye(d_context)[None,:,:]}
# Arm 0 is anticorrelated, arm1 is correlated
dynamic_parameters['dynamics_A']=np.array([
    [[0.9,-0.1],[-0.1,0.9]],
    [[0.9,0.1],[0.1,0.9]]
    ])

dynamic_parameters['theta']=np.zeros((A,d_context,t_max))
dynamic_parameters['theta'][:,:,0]=np.zeros((A, d_context))

for t in np.arange(t_max-1):
    dynamic_parameters['theta'][:,:,t+1]=np.einsum('adb,ab->ad',dynamic_parameters['dynamics_A'],dynamic_parameters['theta'][:,:,t]) + np.einsum('adb,ab->ad',np.linalg.cholesky(dynamic_parameters['dynamics_C']),stats.norm.rvs(size=(A,d_context)))

for a in np.arange(A):
    plt.plot(dynamic_parameters['theta'][a,0,:], 'b', dynamic_parameters['theta'][a,1,:],'r'), plt.show()

# Save
with open('./dynamic_parameters_{}_A{}_dcontext{}_tmax{}.pickle'.format(dynamic_parameters['type'],A,d_context,t_max), 'wb') as f:
    pickle.dump(dynamic_parameters, f)

# Hand-crafted noise mixing
C_sigma=0.01
dynamic_parameters={'type':'noise_mixing', 'dynamics_A':np.ones(A)[:,None,None]*np.eye(d_context)[None,:,:], 'dynamics_C':np.ones(A)[:,None,None]*C_sigma*np.eye(d_context)[None,:,:]}
# Arm 0 is anticorrelated, arm1 is correlated
dynamic_parameters['dynamics_A']=np.array([
    [[0.5,0.0],[0.0,0.5]],
    [[0.9,0.1],[0.1,0.9]]
    ])

dynamic_parameters['theta']=np.zeros((A,d_context,t_max))
dynamic_parameters['theta'][:,:,0]=np.zeros((A, d_context))

for t in np.arange(t_max-1):
    dynamic_parameters['theta'][:,:,t+1]=np.einsum('adb,ab->ad',dynamic_parameters['dynamics_A'],dynamic_parameters['theta'][:,:,t]) + np.einsum('adb,ab->ad',np.linalg.cholesky(dynamic_parameters['dynamics_C']),stats.norm.rvs(size=(A,d_context)))

for a in np.arange(A):
    plt.plot(dynamic_parameters['theta'][a,0,:], 'b', dynamic_parameters['theta'][a,1,:],'r'), plt.show()

# Save
with open('./dynamic_parameters_{}_A{}_dcontext{}_tmax{}.pickle'.format(dynamic_parameters['type'],A,d_context,t_max), 'wb') as f:
    pickle.dump(dynamic_parameters, f)


