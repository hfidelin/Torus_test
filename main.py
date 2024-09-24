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

LAMBDA = 1000.
SIGMA_EPS = 10.
SIGMA_ETA = 50.
SIGMA_P = 10.


def Y(k, measurment):
    return np.array([measurement[:, 0][k],
                     measurement[:, 1][k],
                     measurement[:, 2][k],
                     measurement[:, 3][k],
                     measurement[:, 4][k],
                     measurement[:, 5][k],
                     measurement[:, 6][k],
                     measurement[:, 7][k]])

if __name__ == "__main__":
    
    pos_satellite = np.genfromtxt('input/satellite_positions.csv',
                                  delimiter=',',
                                  skip_header=1)



    Sat0 = np.array([pos_satellite[:, 0], pos_satellite[:, 4]]).T
    Sat1 = np.array([pos_satellite[:, 1], pos_satellite[:, 5]]).T
    Sat2 = np.array([pos_satellite[:, 2], pos_satellite[:, 6]]).T
    Sat3 = np.array([pos_satellite[:, 3], pos_satellite[:, 7]]).T

    SAT = np.array([Sat0, Sat1, Sat2, Sat3])

    pos_receiver = np.genfromtxt('ground_truth/receiver_positions.csv',
                                  delimiter=',',
                                  skip_header=1)
    
    measurement = np.genfromtxt('input/measurement.csv',
                                delimiter=',',
                                skip_header=1)
    
    
    
    
    p0 = np.array([0, 0]) 
    v0 = np.array([2, 2])
    n0 = np.array([0, 0, 0, 0])
    
    kf = KF(initial_p=p0, initial_v=v0, initial_n=n0)
    
    dt = 1.
    NUM_STEP = 3600

    px = np.zeros(NUM_STEP)
    py = np.zeros(NUM_STEP)
    
    
    vx = np.zeros(NUM_STEP)
    vy = np.zeros(NUM_STEP)
    
    for k in range(NUM_STEP):
        
        px[k] = kf.p[0]
        py[k] = kf.p[1]
        
        kf.predict(dt=1., sigma_p=SIGMA_P)
        kf.update(k=k, 
                  SAT=SAT, 
                  Y=Y(k, measurement), 
                  sigma_eps=SIGMA_EPS, 
                  sigma_eta=SIGMA_ETA)
    
    #plt.imshow(np.linalg.inv(kf.S))        
    
    
    
    # %%
    plt.plot(px, py)
    plt.plot(pos_receiver[:, 0], pos_receiver[:, 1])
    plt.grid()
       
    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    