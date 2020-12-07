# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:49:28 2020

@author: cheng
"""
import numpy as np
import matplotlib.pyplot as plt
    
    
def plot_pred(xy, y_prime, N=10, groundtruth=True):
    """
    This is the plot function to plot the first scene
    """
    
    fig,ax = plt.subplots()
    pred_seq = y_prime.shape[2]
    obs_seq = xy.shape[1] - pred_seq
    
    if groundtruth:
        for i in range(N):
            # plot observation
            ax.plot(xy[i, :obs_seq, 2], xy[i, :obs_seq, 3], color='k')
            # plot ground truth
            ax.plot(xy[i, obs_seq-1:, 2], xy[i, obs_seq-1:, 3], color='r')
            for j, pred in enumerate(y_prime[i]):
                # concate the first step for visulization purpose
                pred = np.concatenate((xy[i, obs_seq-1:obs_seq, 2:4], pred), axis=0)            
                ax.plot(pred[:, 0], pred[:, 1], color='b')                
    else:
        x = xy
        obs_seq = x.shape[1]        
        for i in range(N):
            # plot observation
            ax.plot(x[i, :, 2], x[i, :, 3], color='k')
            for j, pred in enumerate(y_prime[i]):
                # concate the first step for visulization
                pred = np.concatenate((x[i, obs_seq-1:obs_seq, 2:4], pred), axis=0)            
                ax.plot(pred[:, 0], pred[:, 1], color='b')                
    ax.set_aspect("equal")
    plt.show()
    plt.gcf().clear()
    plt.close()        