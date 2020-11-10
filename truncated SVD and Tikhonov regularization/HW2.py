# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:18:38 2019

@author: Andrew
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Reconstruction Function ###
def regularize(A,D_hat,method,p):
    global X_hat
    global S
    if method == "tsvd":
        U,S,V = np.linalg.svd(A,full_matrices=True)
        S = S.reshape(A.shape[0],1)
        X_hat = np.zeros((D_hat.shape[0],D_hat.shape[1]))
        for j in range(0,D_hat.shape[1]):
            x_hat = np.zeros((D_hat.shape[0],1))
            d_hat = D_hat[:,j].reshape(D_hat.shape[0],1)
            for i in range(0,p):
                sigma_i = S[i][0]
                u_i = U[:,i].reshape(U.shape[0],1)
                v_i = V[i,:].reshape(V.shape[0],1)
                x_hat = x_hat+(((np.transpose(u_i)@d_hat)/sigma_i)*v_i)
            X_hat[:,j] = x_hat[:,0]
        reconstructed_image = Image.fromarray(X_hat)
#        reconstructed_image.show()
    elif method == "tikhonov":
        U,S,V = np.linalg.svd(A,full_matrices=True)
        S = S.reshape(A.shape[0],1)
        X_hat = np.zeros((D_hat.shape[0],D_hat.shape[1]))
        for j in range(0,D_hat.shape[1]):
            x_hat = np.zeros((D_hat.shape[0],1))
            d_hat = D_hat[:,j].reshape(D_hat.shape[0],1)
            for i in range(0,D_hat.shape[0]):
                sigma_i = S[i][0]
                f_i = (sigma_i**2)/((sigma_i**2)+p**2)
                u_i = U[:,i].reshape(U.shape[0],1)
                v_i = V[i,:].reshape(V.shape[0],1)
                x_hat = x_hat+(f_i*(((np.transpose(u_i)@d_hat)/sigma_i)*v_i))
            X_hat[:,j] = x_hat[:,0]
        reconstructed_image = Image.fromarray(X_hat)
#        reconstructed_image.show()

### Define Reconstruction Function Inputs ###
image_data = pd.read_csv("hw2data.txt",sep="\s+",header=None)
D_hat = pd.DataFrame.to_numpy(image_data)
input_image = Image.fromarray(D_hat)
B = np.zeros((D_hat.shape[0],D_hat.shape[1]))
L = 0.45
power = 10
for i in range(0,B.shape[1]):
    if i<B.shape[1]-1:
        B[i][i] = (1-(2*L))
        B[i][i+1] = L
        B[i+1][i] = L
    else:
        B[i][i] = (1-(2*L))
A = np.linalg.matrix_power(B,power)
method = "tikhonov"
lambda_tk = 5.7999999999999995e-06
#p = 5.7999999999999995e-06
#regularize(A,D_hat,method,p)
### Generate PLots ###
if method == "tsvd":
    x_norms = np.zeros((D_hat.shape[0],1))
    residuals = np.zeros((D_hat.shape[0],1))
    filter_numbers = np.zeros((128,1))
    index = np.zeros((128,1))
    for p in range(0,128):
        regularize(A,D_hat,method,p)
        x_norms[p][0] = np.linalg.norm(X_hat,"fro")
        residuals[p][0] = np.linalg.norm(((A@X_hat)-D_hat),"fro")
        sigma = S[p][0]
        index[p][0] = p
        f_i = (sigma**2/(sigma**2+lambda_tk**2))
        filter_numbers[p][0] = f_i
    L_curve = plt.figure()
    plt.plot(residuals,x_norms, "b*")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frobenius Norm of Residual")
    plt.ylabel("Frobenius Norm of Reconstructed Data")
    plt.savefig("TSVD L Curve.jpg")
    f2 = plt.figure()
    plt.plot(index,S,"b*",label="Singular Values")
    plt.plot(index,filter_numbers,"r*",label="Filter Numbers")
    plt.legend()
    plt.savefig("Plot of Singular Values and Filter Numbers.jpg")

elif method == "tikhonov":
    x_norms = np.zeros((100,1))
    residuals = np.zeros((100,1))
    for c in range(0,100):
        p = 0.0000008+(c*0.000001)
        regularize(A,D_hat,method,p)
        x_norms[c][0] = np.linalg.norm(X_hat,"fro")
        residuals[c][0] = np.linalg.norm(((A@X_hat)-D_hat),"fro")
        if c%10 == 0:
            print(p)
            L_curve = plt.figure()
            plt.plot(residuals,x_norms, "b*")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Frobenius Norm of Residual")
            plt.ylabel("Frobenius Norm of Reconstructed Data")
        plt.savefig("Tikhonov L Curve.jpg")
        


    


