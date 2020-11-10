# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:20:14 2019

@author: Andrew
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.sparse import diags
from scipy import optimize
import math 
import time

D_hat = pd.read_csv("prdata.txt",header=None,delim_whitespace=True) ### Blurred, gappy data
D_hat = pd.DataFrame.to_numpy(D_hat)
M = pd.DataFrame.to_numpy(pd.read_csv("mask.txt",header=None,delim_whitespace=True))
n = 220
m = 520
B = np.zeros((D_hat.shape[0],D_hat.shape[1]))
s = 0.45
power = 10
B = diags([s,(1-(2*s)),s],[-1,0,1],shape=(n,n)).toarray()
A = np.linalg.matrix_power(B,power)  ### Blurring operator


### TSVD Reconstruction ###
p=85
X_hat_tsvd = np.zeros((n,m))
for j in range(0,m):
    start_time = time.time()
    column = M[:,j]
    rows = np.nonzero(column)
    a = A[rows][:]
    d_hat = D_hat[:,j]
    d_hat = d_hat[rows][:].reshape((a.shape[0],1))
    U,S,V = np.linalg.svd(a,full_matrices=True)
    S = S.reshape(a.shape[0],1)
    x_hat = np.zeros((A.shape[0],1))
    for i in range(0,p):
        sigma_i = S[i][0]
        u_i = U[:,i].reshape(U.shape[0],1)
        v_i = V[i,:].reshape(V.shape[0],1)
        x_hat = x_hat+(((np.transpose(u_i)@d_hat)/sigma_i)*v_i)
    x_hat_tsvd = x_hat
    X_hat_tsvd[:,j] = x_hat_tsvd[:,0] 
reconstructed_image_tsvd = Image.fromarray(X_hat_tsvd)
#reconstructed_image_tsvd.show()


### Tikhonov Reconstruction ###
X_hat_tk = np.zeros((n,m))
L_type = 2
L_0 = np.identity(220)
L_1 = diags([-1,1],[0,1],shape=(219,220)).toarray()
L_2 = diags([1,-2,1],[0,1,2],shape=(218,220)).toarray()

x_norms = np.zeros((100,1))
residuals = np.zeros((100,1))

if L_type == 0:
    L = L_0
elif L_type == 1:
    L = L_1
elif L_type == 2:
    L = L_2
for c in range(0,400):    
    l = 0.01+(0.1*c)
    for j in range(0,m):
        d_hat = D_hat[:,j]
        B = (np.transpose(A)@A)+((l**2)*(np.transpose(L)@L))
        x_hat_tk = (np.linalg.inv(B)@np.transpose(A)@d_hat)
        X_hat_tk[:,j] = x_hat_tk[:]        
    reconstructed_image_tk = Image.fromarray(X_hat_tk)
    x_norms[c][0] = np.linalg.norm(X_hat_tk,"fro")
    residuals[c][0] = np.linalg.norm(((A@X_hat_tk)-D_hat),"fro")
    if c%15 == 0:
        print(l)
        L_curve = plt.figure()
        plt.plot(residuals,x_norms, "b*")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Frobenius Norm of Residual")
        plt.ylabel("Frobenius Norm of Reconstructed Data")
#    plt.savefig("Tikhonov L Curve.jpg")    
