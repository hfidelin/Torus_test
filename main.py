#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:20:29 2024

@author: fidelin
"""
import numpy as np
import matplotlib.pyplot as plt
from kf import KF


# Declaring some global variable

M = 4
LAMBDA = 1000.
SIGMA_EPS = 10.
SIGMA_ETA = 50.
SIGMA_P = 10.
DT = 1.
NUM_STEP = 3600



if __name__ == "__main__":
    
    
    pos_satellite = np.genfromtxt('input/satellite_positions.csv',
                                  delimiter=',',
                                  skip_header=1)



    Sat0 = np.array([pos_satellite[:, 0], pos_satellite[:, 4]]).T
    Sat1 = np.array([pos_satellite[:, 1], pos_satellite[:, 5]]).T
    Sat2 = np.array([pos_satellite[:, 2], pos_satellite[:, 6]]).T
    Sat3 = np.array([pos_satellite[:, 3], pos_satellite[:, 7]]).T

    SAT = np.array([Sat0, Sat1, Sat2, Sat3])

    true_pos = np.genfromtxt('ground_truth/receiver_positions.csv',
                                  delimiter=',',
                                  skip_header=1)
    
    measurement = np.genfromtxt('input/measurement.csv',
                                delimiter=',',
                                skip_header=1)
    
    y = np.array([measurement[:, 0],
                  measurement[:, 1],
                  measurement[:, 2],
                  measurement[:, 3],
                  measurement[:, 4],
                  measurement[:, 5],
                  measurement[:, 6],
                  measurement[:, 7]]).T
    
    
    # Initial position
    p0 = np.array([0, 0]) 
    
    # Constant velocity
    v0 = np.array([2, 2])
    
    # Initial ambiguity vector
    n0 = np.array([0, 0, 0, 0])
    
    kf = KF(M=M, initial_p=p0, initial_v=v0, initial_n=n0)
    
    
    n1 = np.zeros(NUM_STEP)
    n2 = np.zeros(NUM_STEP)
    n3 = np.zeros(NUM_STEP)
    n4 = np.zeros(NUM_STEP)
    
    err_pos = np.zeros(NUM_STEP)
    err_vel = np.zeros(NUM_STEP)
    rmse = np.zeros(NUM_STEP)
    
    # TIME ITERATION LOOP
    for k in range(NUM_STEP):
        
        # Position error
        err_pos[k] = np.linalg.norm(kf.p - true_pos[k])
        
        # Velocity error
        err_vel[k] = np.linalg.norm(kf.v - v0)

        # Retrieving ambiguity number
        n1[k] = kf.n[0]
        n2[k] = kf.n[1]
        n3[k] = kf.n[2]
        n4[k] = kf.n[3]
        
        # Prediction step
        kf.predict(dt=DT, sigma_p=SIGMA_P)
        
        # Updating step
        kf.update(k=k, 
                  SAT=SAT, 
                  y=y[k], 
                  sigma_eps=SIGMA_EPS, 
                  sigma_eta=SIGMA_ETA)
    
        # RMSE 
        rmse[k] = np.sqrt(((y[k] - kf.innov) **2).mean())
    
    
    # %% Position error in loglog scale
    
    plt.loglog(err_pos, 'b')
    plt.title(r"Position error")
    plt.xlabel("Iteration")
    plt.ylabel(r"Error$")
    plt.grid()

    # %% Velocity error in loglog scale
    
    plt.loglog(err_vel, 'b')
    plt.title(r"Velocity error")
    plt.xlabel("Iteration")
    plt.ylabel(r"Error$")
    plt.grid()
    
    # %% Ambiguity number
    
    plt.plot(n1, label=r"$n_1$")
    plt.plot(n2, label=r"$n_2$")
    plt.plot(n3, label=r"$n_3$")
    plt.plot(n4, label=r"$n_4$")
    
    plt.title(r"Ambiguity numbers")
    plt.xlabel("Iteration")
    plt.ylabel(r"$n$")
    
    plt.grid()
    plt.legend()
    
    # %% RMSE of innovation
    
    plt.loglog(rmse)
    plt.title(r"RMSE between innovation and observation model")
    plt.xlabel("Iteration")
    plt.ylabel(r"Error rmse")
    plt.grid()
    
    
    
    
    
    
    
    