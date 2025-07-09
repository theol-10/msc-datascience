import numpy as np
from numpy import genfromtxt, vstack, sqrt, std, concatenate, reshape, dot
from numpy.linalg import norm, svd
from numpy.core.fromnumeric import argmin
import pandas as pd
from pandas import read_csv, DataFrame, concat
import sys
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular, qr
import imageio 
from imageio import imread, imsave



# 1. Least Squares problem  #


def svd_LS(A, b):
    
    U, Sigma, VT = np.linalg.svd(A, full_matrices=False)
    Sigma_inv = np.diag(1 / Sigma)
    x_svd = VT.T @ Sigma_inv @ U.T @ b

    return x_svd


def qr_LS(A, b):
    
    Rank = np.linalg.matrix_rank(A)
    x_qr = None

    if Rank == A.shape[1]:
        Q_fullr, R_fullr = np.linalg.qr(A)
        y_aux = np.transpose(Q_fullr).dot(b)
        x_qr = solve_triangular(R_fullr, y_aux)

    else:
        Q, R, P = qr(A, mode='economic', pivoting=True)  
        R_def = R[:Rank, :Rank]
        c = np.transpose(Q).dot(b)[:Rank]
        u = solve_triangular(R_def, c)
        v = np.zeros((A.shape[1] - Rank))
        x_qr = np.linalg.solve(np.transpose(np.eye(A.shape[1])[:, P]), np.concatenate((u, v)))
        
    return x_qr


def dataset(degree):
    
    data = genfromtxt("dades.csv", delimiter="   ")
    points, b = data[:, 0], data[:, 1]
    A = vstack([points ** d for d in range(degree)]).T
    
    return A, b


def dataset2(degree):
    data = genfromtxt('dades_regressio.csv', delimiter=',')
    A, b = data[:, :-1], data[:, -1]
    
    return A, b


svd_errors = []
degrees=range(3,10)

for degree in range(3,10):
    A, b = dataset(degree)
    x_svd = svd_LS(A, b)
    x_qr = qr_LS(A, b)
    svd_errors.append(norm(A.dot(x_svd) - b))

min_svd_error_pos = argmin(svd_errors)
best_degree = min_svd_error_pos+3

print("Dataset 1:")
print("Best degree:", best_degree)
A, b = dataset(best_degree)
x_svd = svd_LS(A, b)
x_qr = qr_LS(A, b)
print("\n")
print("LS solution using SVD:", x_svd)
print("Solution norm:", norm(x_svd))
print("Error:", norm(A.dot(x_svd)-b))
print("\n")
print("LS solution using QR:", x_qr)
print("Solution norm:", norm(x_qr))
print("Error:", norm(A.dot(x_qr)-b))
print("\n")


print("Dataset 2:")
print("Best degree:", best_degree)
A, b = dataset2(best_degree)
x_svd = svd_LS(A, b)
x_qr = qr_LS(A, b)
print("\n")
print("LS solution using SVD:", x_svd)
print("Solution norm:", norm(x_svd))
print("Error:", norm(A.dot(x_svd)-b))
print("\n")
print("LS solution using QR:", x_qr)
print("Solution norm:", norm(x_qr))
print("Error:", norm(A.dot(x_qr)-b))
print("\n")




# 2 Graphics compression #


image1 = imageio.imread('image1.jpg')
image2 = imageio.imread('image2.jpg')
Image1 = image1[:,:,1]
Image2 = image2[:,:,1]


def compress_image(matrix, output_prefix='compressed'):
    U, sigma, V = np.linalg.svd(matrix)
    rank= [1, 5, 20, 80]

    for i in rank:
        A = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
        relative_error = np.sum(sigma[i:]**2) / np.sum(sigma**2)   #relative error
        percentage_captured = np.linalg.norm(A) / np.linalg.norm(matrix)

        output_filename = f"{output_prefix}_rank_{i}_capture_{percentage_captured:.2f}.jpg"
        imageio.imwrite(output_filename, np.clip(A, 0, 255).astype(np.uint8))
    
        print(f"Rank {i} - Percentage Captured: {percentage_captured:.2f}% - Relative Error: {relative_error:.4f}")


print("Results Image 1 (image1.jpg):")
compress_image(Image1, output_prefix='image1_compressed')
print("\n")
print("Results Image 2 (image2.jpg):")
compress_image(Image2, output_prefix='image2_compressed')        




# 3. Principal Component Analysis #


def read_dat():
    X = np.genfromtxt('example.dat', delimiter = ' ')
    return X.T

def read_csv():
    X = np.genfromtxt('RCsGoff.csv', delimiter = ',')
    return X[1:,1:].T 


def PCA(matrix_choice, file_choice):
    
    if file_choice == 1: X = read_dat()
    else: X = read_csv()
    
    X = X - np.mean(X, axis = 0)
    n = X.shape[0]
    if matrix_choice == 1: 
        # covariance matrix
        Y = (1 / (np.sqrt( n - 1))) * X.T
        U,S,VH = np.linalg.svd(Y, full_matrices = False)
        total_var = S**2 / np.sum(S**2)
        standard_dev = np.std(VH, axis = 0)
        new_expr_PCA_coord = np.matmul(VH,X).T
        
    else: 
        # correlation matrix
        X = (X.T / np.std(X, axis = 1)).T
        Y = (1 / (np.sqrt( n - 1))) * X.T
        U,S,VH = np.linalg.svd(Y, full_matrices = False)
        
        total_var = S**2 / np.sum(S**2)

        standard_dev = np.std(VH.T, axis = 0)

        new_expr_PCA_coord = np.matmul(VH,X).T
        
    return total_var, standard_dev, new_expr_PCA_coord, S


def Scree_plot(S,number_figure,matrix_type):
    
    if matrix_type == 1:
        #covariance matrix
        plt.figure(number_figure)
        plt.plot(range(len(S)), S)
        for i in range(len(S)):
            plt.scatter(i,S[i],color='red')
        plt.title('Scree plot - covariance matrix')
        plt.xlabel('Principal Components')
        plt.ylabel('Eigenvalues')
        plt.savefig("scree_plot_cov.jpg")
        plt.show()
        
    else:
        #correlation matrix
        plt.figure(number_figure)
        plt.plot(range(len(S)), S)
        for i in range(len(S)):
            plt.scatter(i,S[i],color='red')
        plt.title('Scree plot - correlation matrix')
        plt.xlabel('Principal Components')
        plt.ylabel('Eigenvalues')
        plt.savefig("scree_plot_corr.jpg")
        plt.show()    


def Kasier(S):
    
    count = 0
    for i in range(len(S)):
        if S[i]>1: count += 1
            
    return count


def rule_34(var):
    
    total_var = sum(var)
    new_var = []
    i = 0  
    while sum(new_var) < 3*total_var/4:
        new_var.append(var[i])
        i += 1

    return len(new_var)



## covariance matrix   
print('Covariance matrix')
total_var,standar_dev,new_expr,S = PCA(1,1)
print('\n')
print('Total variance in each component: ',total_var)
print('\n')
print('Standard deviation of each component: ',standar_dev)
print('\n')
print('PCA coordinates of original dataset: ',new_expr)
Scree_plot(S,1,1)
print('\n')
print('Kasier rule:',Kasier(S))
print('3/4 rule:',rule_34(total_var))
print('\n')


## Correlation matrix 
print('Correlation matrix')
total_var,standar_dev,new_expr,S = PCA(0,1)
print('\n')
print('Total variance in each principal component: ',total_var)
print('\n')
print('Standard deviation of each principal component: ',standar_dev)
print('\n')
print('PCA coordinates of original dataset: ',new_expr)
Scree_plot(S,2,0)
print('\n')
print('Kasier rule:',Kasier(S))
print('3/4 rule:',rule_34(total_var)) 
print('\n')      


# RCsGoff.csv

## Covariance matrix   
print('Covariance matrix')
total_var,standar_dev,new_expr,S = PCA(1,0)
print(new_expr.shape)
print('\n')
print('Total variance in each principal component: ',total_var)
print('\n')
print('Standard deviation of each principal component: ',standar_dev)
print('\n')
print('PCA coordinates of original dataset: ',new_expr)
Scree_plot(S,3,1)
print('\n')
print('Kasier rule:',Kasier(S))
print('3/4 rule:',rule_34(total_var))
print('\n')

X_RCsGoff = read_csv()  

data_df = pd.DataFrame(data=new_expr[:20, :].T, columns=[f"PC{i}" for i in range(1, 21)])
variance_df = pd.DataFrame(data=reshape(total_var, (20, 1)), columns=["Variance"])

if 'gene' in data_df.columns: data_df = data_df.drop('gene', axis=1)
data_df.index.name = "Sample"
data_df["Variance"] = variance_df["Variance"]

data_df.to_csv("rcsgoff_covariance.txt", sep='\t')


## correlation matrix   

print('Correlation matrix')
total_var,standar_dev,new_expr,S = PCA(0,0)
print(new_expr.shape)
print('\n')
print('Total variance in each principal component: ',total_var)
print('\n')
print('Standard deviation of each principal component: ',standar_dev)
print('\n')
print('PCA coordinates of original dataset: ',new_expr)
Scree_plot(S,4,0)
print('\n')
print('Kasier rule:',Kasier(S))
print('3/4 rule:',rule_34(total_var)) 
print('\n') 


X_RCsGoff = read_csv()  

data_df = pd.DataFrame(data=new_expr[:20, :].T, columns=[f"PC{i}" for i in range(1, 21)])
variance_df = pd.DataFrame(data=reshape(total_var, (20, 1)), columns=["Variance"])

if 'gene' in data_df.columns: data_df = data_df.drop('gene', axis=1)

data_df.index.name = "Sample"
data_df["Variance"] = variance_df["Variance"]

data_df.to_csv("rcsgoff_correlation.txt", sep='\t')