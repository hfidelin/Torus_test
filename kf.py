#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 20:07:12 2024

Kalman Filter class 

@author: fidelin
"""
import numpy as np
import matplotlib.pyplot as plt

# Declaring some global variable

M = 4


# %% Loading data

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


# %%


class KF():
    
    def __init__(self, 
                 initial_p : np.array, 
                 initial_v : np.array, 
                 initial_n : np.array) -> None:
        
        global M
        
        self.p = initial_p
        self.v = initial_v
        self.n = initial_n
        
        # Initialize P matrix at identity for now
        self.P = np.eye(4+M, 4+M)
        
    def predict(self, dt : float, sigma_p : float) -> None:
        
        # Initiate F matrix
        F = np.eye(4+M)
        F[0, 2] = dt
        F[1, 3] = dt
        
        # Initiate Q matrix
        Q = np.zeros((4+M, 4+M))
        Q[0, 0] = sigma_p ** 2
        Q[1, 1] = sigma_p ** 2
        
        # Create state vector
        x = np.concatenate((self.p, self.v, self.n))
        
        new_x = F @ x
        new_P = F @ self.P @ F.T + Q
        
        self.p = new_x[0: 2]
        self.v = new_x[2: 4]
        self.n = new_x[4::]
        self.P = new_P
    
    
    
    def _coef_H(self, i : int, j : int, k : int, SAT : np.array) -> float :
        

        
        if i == 0 or i == 4:
            
            return -(SAT[j, k, 0] - self.p[0]) / \
                np.linalg.norm(SAT[j, k, :] - self.p)
        
        elif i == 1 or i == 5: 
            
            return -(SAT[j, k, 1] - self.p[1]) / \
                np.linalg.norm(SAT[j, k, :] - self.p)
        
        else :
            return 0.
        
    
    def _dhx(self, Sat : np.array) -> float:
        return -px * (Sat[0] - self.p[0]) / np.linalg.norm(Sat - self.p)
    
    def _dhy(self, Sat : np.array) -> float:
        return -py * (Sat[1] - self.p[1]) / np.linalg.norm(Sat - self.p)
    
    
    def _init_H(self, k : int, SAT : np.array) -> np.array:
        
        global M
        
        H = np.zeros((2 * M, 4 + M))
        
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                
                jj = j % M
                
                H[i, j] = self._coef_H(i, jj, k, SAT)  
        return H
    
    def _init_R(self, k : int, sigma_eps : float, sigma_eta : float) -> np.array:
        
        global M
        
        v1 = sigma_eps ** 2 * np.ones((M,))
        v2 = sigma_eta ** 2 * np.ones((M,))
        
        v = np.hstack((v1, v2))
        K = np.diag(v)
        return K     
                
                
    
    def update(self, k : int, 
               SAT : np.array, 
               Y : np.array,
               sigma_eps : float, 
               sigma_eta : float) -> None:
        
        H = self._init_H(k=k, SAT=SAT)
        R = self._init_R(k=k, sigma_eps=sigma_eps, sigma_eta=sigma_eta)
        
        # Creating state vector
        x = np.concatenate((self.p, self.v, self.n))
        
        # Compute matrix
        S = H @ self.P @ H.T + R
        
        # Compute Kalman matrix
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Computing residual vector
        v = Y - H @ x
        
        
        
        new_P = self.P - K @ H @ self.P
        new_x = x + K @ v
        
        #update state vector and covariance matrix
        self.p = new_x[0: 2]
        self.v = new_x[2: 4]
        self.n = new_x[4::]
        self.P = new_P
    
   
if __name__ == "__main__":
    
    p0 = np.array([0, 0]) 
    v0 = np.array([2, 2])
    n0 = np.array([0, 0, 0, 0])
    kf = KF(initial_p=p0, initial_v=v0, initial_n=n0)
    
    # for _ in range(10):
        
    #     det_before = np.linalg.det(kf.P)
    #     kf.predict(dt=1.)
    #     det_after = np.linalg.det(kf.P)
    #     print(det_after > det_before)
    kf.update(k=0)
    
    
    
    # %%
    # plt.ion()
    # plt.figure()
    dt = 1.
    NUM_STEP = 3600
    
    mus = []
    covs = []
    px = np.zeros(NUM_STEP)
    py = np.zeros(NUM_STEP)
    
    for t in range(NUM_STEP):
        
        px[t] = kf.p[0]
        py[t] = kf.p[1]
        # mus.append(kf.p)
        # covs.append(kf.P)
    
        kf.predict(dt=1.)
    
    plt.plot(px, py)
    


    plt.plot(pos_receiver[:, 0], pos_receiver[:, 1])
    plt.grid()
    










    
        