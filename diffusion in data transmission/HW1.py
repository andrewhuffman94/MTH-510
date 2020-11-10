# -*- coding: utf-8 -*-
"""
Created on Mon Oct 7 11:52:05 2019

@author: Andrew
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

## Load image and convert to matrix of grayscale form ##
clown = Image.open("clown1.jpg")
clown = clown.convert("L")
image = clown
clown_matrix = np.asarray(clown.getdata(), dtype=np.float64).reshape((clown.size[1],clown.size[0]))
X = clown_matrix

# Generate noise matrix 
eps = np.zeros((200,320))
mu = 0
sigma = 0.1
for i in range(0,image.size[1]):
    for j in range(0,image.size[0]):
        eps[i][j] = np.random.normal(mu,sigma)

# Generate blurring operator and reconstruct blurred image 
B = np.zeros((image.size[1],image.size[1]))
L = 0.1
con_num = np.zeros((20,1))
k_vector = np.zeros((20,1))
absolute_error = np.zeros((20,1))
relative_error = np.zeros((20,1))
for i in range (0,200):
    if i<199:
        B[i][i] = (1-(2*L))
        B[i][i+1] = L
        B[i+1][i] = L
    else:
        B[i][i] = (1-(2*L))
        
for k in range(1,21):
    k_vector[k-1][0] = k
    A = np.linalg.matrix_power(B,k)
    A_inv = np.linalg.inv(A)
    con_num[k-1][0] = np.linalg.cond(A)
    D_noisy = np.matmul(A,X)+eps
    X_noisy = np.matmul(A_inv,D_noisy)
    error_abs = np.linalg.norm((X-X_noisy),"fro")
    absolute_error[k-1][0] = error_abs
    error_rel = np.linalg.norm((X-X_noisy),"fro")/np.linalg.norm(X,"fro")
    relative_error[k-1][0] = error_rel
    if k == 1:
        clown_noisy = Image.fromarray(X_noisy.astype("uint8"),"L")
        clown_noisy.save("Reconstructed_Image_k1.jpg")
    if k == 5:
        clown_noisy = Image.fromarray(X_noisy.astype("uint8"),"L")
        clown_noisy.save("Reconstructed_Image_k5.jpg")
    if k == 10:
        clown_noisy = Image.fromarray(X_noisy.astype("uint8"),"L")
        clown_noisy.save("Reconstructed_Image_k10.jpg")
    if k == 15:
        clown_noisy = Image.fromarray(X_noisy.astype("uint8"),"L")
        clown_noisy.save("Reconstructed_Image_k15.jpg")
    if k == 20:
        clown_noisy = Image.fromarray(X_noisy.astype("uint8"),"L")
        clown_noisy.save("Reconstructed_Image_k20.jpg")


# Plot condition number and error vs k 
f1 = plt.figure()
plt.plot(k_vector,con_num,"r*")
plt.xlabel("k")
plt.ylabel("Condition Number")
f1.savefig("Condition_Number_vs_k.jpg")


f2 = plt.figure()
plt.plot(k_vector,absolute_error,"c*")
plt.xlabel("k")
plt.ylabel("Absolute Error")
f2.savefig("Absolute_Error_vs_k.jpg",edgecolor="k")

f3 = plt.figure()
plt.plot(k_vector,relative_error,"b*") 
plt.xlabel("k")
plt.ylabel("Relative Error")
f3.savefig("Relative_Error_vs_k.jpg",edgecolor="k")


        