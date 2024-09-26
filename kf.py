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
        """
        Prediction step of KF

        Parameters
        ----------
        dt : float
            Time step.
        sigma_p : float
            Evolution noise.

        Returns
        -------
        None
            Proceed to state k to stake k+1.

        """
        # Initiate state transition matrix
        F = np.eye(4+M)
        F[0, 2] = dt
        F[1, 3] = dt
        
        # F[4, 4] = 1000
        # F[5, 5] = 1000
        # F[6, 6] = 1000
        # F[7, 7] = 1000
        
        # Initiate process noice covariance matrix
        Q = np.zeros((4+M, 4+M))
        Q[0, 0] = sigma_p ** 2
        Q[1, 1] = sigma_p ** 2
        
        # Create state vector
        x = np.concatenate((self.p, self.v, self.n))
        
        # Compute new state vector and covariance matrix
        new_x = F @ x
        new_P = F @ self.P @ F.T + Q
        
        self.p = new_x[0: 2]
        self.v = new_x[2: 4]
        self.n = new_x[4::]
        self.P = new_P
    
    
    
    def _coef_H(self, i : int, j : int, k : int, SAT : np.array) -> float :
        """
        Compute the coefficient of the Jacobian matrix H at indices (i,j) at
        time step k.

        Parameters
        ----------
        i : int
            row index.
        j : int
            column index.
        k : int
            time number of step.
        SAT : np.array
            Array containing all satellite position.

        Returns
        -------
        float
            coefficient of h at (i,j).

        """
        if i == 0:
            
            return -(SAT[j, k, 0] - self.p[0]) / \
                np.linalg.norm(SAT[j, k, :] - self.p)
        
        elif i == 1: 
            
            return -(SAT[j, k, 1] - self.p[1]) / \
                np.linalg.norm(SAT[j, k, :] - self.p)
        
        else :
            return 0.
        
    
    def _init_H(self, k : int, SAT : np.array) -> np.array:
        """
        Compute Jacobian matrix for measurement step of KF

        Parameters
        ----------
        k : int
            number of time step.
        SAT : np.array
            Array containing all satellite position.

        Returns
        -------
        H : np.array
            Array of shape (4+M, 2M).

        """
        global M
        
        # Initiate matrix
        H = np.zeros((2 * M, 4 + M))
        
        # Double for loop to cross every matrix coefficient
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                
                # Using the correct satellite
                jj = j % M
                
                # Computing Hij coefficient
                H[i, j] = self._coef_H(i, jj, k, SAT)  
        
        return H
    
    def _init_R(self, sigma_eps : float, sigma_eta : float) -> np.array:
        """
        Compute covariance matrix associated to measurement noise

        Parameters
        ----------
        sigma_eps : float
            measurement noise.
        sigma_eta : float
            measurement noise.

        Returns
        -------
        K : np.array
            Array of shape (2M, 2M).

        """
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
        """
        Updated value of state vector using measutement step of KF

        Parameters
        ----------
        k : int
            number of time step.
        SAT : np.array
            Array containing all satellite position.
        Y : np.array
            Measurement model at step k, must be of shape (2M,).
        sigma_eps : float
            measurement noise.
        sigma_eta : float
            measurement noise.

        Returns
        -------
        None
            Perform measurement step.

        """
        
        H = self._init_H(k=k, SAT=SAT)
        R = self._init_R(sigma_eps=sigma_eps, sigma_eta=sigma_eta)
        
        self.H = H
        self.R = R
        
        # Creating state vector
        x = np.concatenate((self.p, self.v, self.n))
        
        # Computing residual vector
        self.innov = Y - H @ x
        
        # Compute S matrix
        S = H @ self.P @ H.T + R
        
        self.S = S
        # Compute Kalman matrix
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.K = K
        
        new_P = self.P - K @ H @ self.P
        new_x = x + K @ self.innov
        
        #update state vector and covariance matrix
        self.p = new_x[0: 2]
        self.v = new_x[2: 4]
        self.n = new_x[4::]
        self.P = new_P
    
    










    
        