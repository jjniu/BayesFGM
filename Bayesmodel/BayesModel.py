#Simulaion funcitonal data: regular, sparse, irregularly sampled data
import numpy as np
import matplotlib.pyplot as plt
# import skfda
# import skfda.representation.basis as basis
# %matplotlib inline
import pandas as pd
import seaborn as sns
import time
import pickle
import rpy2

import rpy2.robjects as robjects

from rpy2.robjects.packages import importr
from tqdm import tqdm
# import tqdm.notebook as tq
# from tqdm.auto import tqdm
from scipy.stats import norm
from scipy.stats import invgauss
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV
from sklearn.covariance import graphical_lasso
from sklearn.metrics import confusion_matrix
from numpy import savetxt 

# class Timer:
#     def __enter__(self):
#         self.t0 = time.time()
#     def __exit__(self, *args):
#         print('Elapsed time: %0.2fs' % (time.time()-self.t0))

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def cov2cor(mat):
    ##convert the convariance matrix to correlation matrix
    D = np.diag(np.sqrt(np.diag(mat)))
    D_inv = np.linalg.inv(D)
    cor_mat = np.matmul(np.matmul(D_inv, mat), D_inv)
    return cor_mat

def catch_in_CI(samples, q, axis = 0):
    LB = np.quantile(samples, q = (1-q)/2, axis = axis)
    UB = np.quantile(samples, q = 1-(1-q)/2, axis = axis)
    return (LB>0)|(UB<0)

def F_norm(Theta, p, M):
    ##blockwise Frobunius norm for precision matrix
    # matx = np.kron(np.diag(np.ones(p, dtype=np.float32)), np.ones(M, dtype=np.float32)).transpose()
    matx = np.kron(np.diag(np.ones(p)), np.ones(M)).transpose()
    Theta_F = np.matmul(np.matmul(np.transpose(matx), Theta**2),matx)
    return np.sqrt(Theta_F)

def Bayes_fglasso(data, p, regular_parm = None, lambda_shape = 1, lambda_rate = 0.01, nBurnin = 1e3, nIter = 10e3):

    def F_norm(Theta, p, M):
        matx = np.kron(np.diag(np.ones(p)), np.ones(M)).transpose()
        Theta_F = np.matmul(np.matmul(np.transpose(matx), Theta**2),matx)
        return np.sqrt(Theta_F)

    def sampleTau(Theta_F, regular_parm):
        if np.amin(np.matrix.flatten(Theta_F))<1e-6:
            Theta_F = Theta_F + 1e-6
        tau_sq = 1/np.random.wald(regular_parm/Theta_F, regular_parm**2)
#         tau_sq = (tau_sq+tau_sq.transpose())/2
        return tau_sq

    def sampleLambda(Theta, Tau_sq):
        shape_new = lambda_shape + p*M + (M**2+1)*p*(p-1)/4
        rate_new = lambda_rate + np.sum(np.diag(Theta))/2 + np.sum(np.triu(Tau_sq, 1))/2
        lamb_sq = np.random.gamma(shape = shape_new, scale = 1/rate_new)
        return np.sqrt(lamb_sq)

    # def sampleLambda(Theta, p):
    #     M = int(Theta.shape[0]/p) 
    #     Theta_F = F_norm(Theta, p, M)
    #     shape_new = lambda_shape+p*M 
    #     rate_new = lambda_rate + np.sum(np.diag(Theta))/2 + np.sum(np.triu(Theta_F, 1))
    #     lamb = np.random.gamma(shape = shape_new, scale = 1/rate_new)
    #     return lamb

    def permut(Mat, mat_p):
        return np.matmul(np.matmul(mat_p, Mat), mat_p)

    def parti(Mat, j):
        #functional for partitioning matirx
        Mat11 = np.delete(np.delete(Mat, j, axis = 0),j, axis = 1)
        Mat12 = np.delete(Mat[:,j],j, axis = 0)
        Mat21 = Mat12.transpose()
        Mat22 = Mat[j,j]
        return (Mat11, Mat12, Mat21, Mat22)

    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    if regular_parm is None:
            sample_lambda = True
            regular_parm = 0.3
    else:
        sample_lambda = False

    N = data.shape[0]
    M = int(data.shape[1]/p)

    ##centralize data
    data = data - np.mean(data, axis = 0).reshape((1,p*M))
    # S = np.cov(data.transpose())##sample covariance/correlation matrix S:
#     S = np.real(np.corrcoef(data.transpose()))
    S = np.matmul(data.transpose(), data) #+ np.diag(np.ones(shape = p*M)*1e-8)

    ## use glasso with lambda = 0.6 for initial value
    # Theta = graphical_lasso(S, alpha = 0.6)[1]
    Theta = np.diag(np.ones(shape = p*M))##use identity matrix as initial

    Theta_F = F_norm(Theta, p,M)
    Tau = np.kron(sampleTau(Theta_F, regular_parm), np.ones(shape = (M,M)))

    samples = []
    lambda_sq = []

    total = len(range(-int(nBurnin), int(nIter)+1, 1))
    # with tqdm(total=total, position=0, leave=True) as pbar:
    for it in tqdm(range(-int(nBurnin), int(nIter)+1, 1), position=0, leave=True):
        # pbar.update()
        for i in range(p):
            ##create permutation matrix for exchange ith block and pth block
            m = np.diag(np.ones(p))
            m[:,[i,p-1]] = m[:,[p-1,i]]
            mat_p = np.kron(m, np.diag(np.ones(M)))

            #exchange the ith and pth node
            Theta_ = permut(Theta, mat_p)
            S_ = permut(S, mat_p)
            Tau_ = permut(Tau, mat_p)

            ##for every principal component
            for j_ in range(int(M)):
                j = (p-1)*M +j_

                ##partition matrices:
                (Theta11, Theta12, Theta21, Theta22) = parti(Theta_,j)
                Theta11_inv = np.linalg.inv(Theta11)[:(p-1)*M,:(p-1)*M]

                (S11, S12, S21, S22) = parti(S_,j)
                (Tau11, Tau12, Tau21, Tau22) = parti(Tau_,j)

                gamma = np.random.gamma(shape = N/2+1, scale = 2/(S22+regular_parm**2))

                Ell = np.linalg.cholesky((S22+regular_parm**2)*Theta11_inv + np.diag(1/Tau12[:(p-1)*M])).transpose()
                temp1 = np.linalg.solve(Ell.transpose(), -1*S21[:(p-1)*M])
                mu = np.linalg.solve(Ell, temp1)
                vee = np.linalg.solve(Ell, np.random.normal(size = (p-1)*M))
                beta = mu+vee
                temp = np.append(beta, np.zeros(M-1))
                Theta_[:,j][np.arange(len(Theta_))!=j] = temp
                Theta_[j,:][np.arange(len(Theta_))!=j] = temp
                Theta_[j,j] = gamma + np.sum(beta*np.matmul(Theta11_inv, beta))

            Theta = permut(Theta_, mat_p)
        
        ##update Theta_F
        Theta_F = F_norm(Theta, p, M)
        
        #update Tau
        Tau_sq = sampleTau(Theta_F, regular_parm)
        Tau = np.kron(Tau_sq, np.ones(shape = (M,M)))

        ##   update lambda
        if sample_lambda is True:
            regular_parm = sampleLambda(Theta, Tau_sq)
        #Store:
        if it>0:
            samples.append(Theta)
            
        if sample_lambda is True:
            lambda_sq.append(regular_parm)

    samples = np.stack(samples, axis=0)
    return (samples, lambda_sq) 


def Bayes_fghorse(data, p, nBurnin = 1e3, nIter = 10e3, Last_sample = None):   
    
    def sampleLambda(Theta_F, Nu, tau_sq, M):
        Lambda_sample = 1/np.random.gamma(shape = (1+M**2)/2, scale = 1/(1/Nu+ Theta_F**2/(2*tau_sq)))
#         Lambda_sample = (Lambda_sample + Lambda_sample.transpose())/2
        # return np.float32(Lambda_sample)
        return Lambda_sample

    def sampleNu(Lambda):
        Nu_sample = 1/np.random.gamma(shape = 1, scale = 1/(1+1/Lambda))
#         Nu_sample = (Nu_sample + Nu_sample.transpose())/2
        # return np.float32(Nu_sample)
        return Nu_sample

    def permut(Mat, mat_p):
        return np.matmul(np.matmul(mat_p, Mat), mat_p)

    def parti(Mat, j):
        #functional for partitioning matirx
        Mat11 = np.delete(np.delete(Mat, j, axis = 0),j, axis = 1)
        Mat12 = np.delete(Mat[:,j],j, axis = 0)
        Mat21 = Mat12.transpose()
        Mat22 = Mat[j,j]
        return (Mat11, Mat12, Mat21, Mat22)

    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)
     
    N = data.shape[0]
    M = int(data.shape[1]/p)

    ##centralize data
    data = data - np.mean(data, axis = 0).reshape((1,p*M))
    # S = np.cov(data.transpose())##sample covariance/correlation matrix S:
#     S = np.corrcoef(data.transpose())
    S = np.matmul(data.transpose(), data)

    if Last_sample is None: 
        ## use glasso with lambda = 0.6 for initial value
        # Theta = graphical_lasso(S, alpha = 0.6)[1]
        # Theta = np.float32(np.diag(np.ones(shape = p*M)))##use identity matrix as initial
        Theta = np.diag(np.ones(shape = p*M))##use identity matrix as initial    
        Theta_F = F_norm(Theta, p, M)
        tau_sq = 1
        zeta = 1
        # Nu = np.ones([p,p], dtype=np.float32)
        Nu = np.ones([p,p])
        Lambda = sampleLambda(Theta_F, Nu, tau_sq, M)
        Nu = sampleNu(Lambda)
    else:
        Theta = Last_sample[0]
        tau_sq = Last_sample[1]
        zeta = Last_sample[2]
        Nu = Last_sample[3]
        Lambda = Last_sample[4]
        nBurnin = 0

    Theta_F = F_norm(Theta, p, M)
    lambda_ = 1 ##diagonal Lambda

    samples = []

    for it in tqdm(range(-int(nBurnin), int(nIter)+1, 1)):
        for i in range(p):
            ##create permutation matrix for exchange ith block and pth block
            # m = np.diag(np.ones(p, dtype=np.float32))
            m = np.diag(np.ones(p))
            m[:,[i,p-1]] = m[:,[p-1,i]]
            # mat_p = np.kron(m, np.diag(np.ones(M, dtype=np.float32)))
            mat_p = np.kron(m, np.diag(np.ones(M)))

            #exchange the ith and pth node
            Theta_ = permut(Theta, mat_p)
            S_ = permut(S, mat_p)

            # Lambda_mat = np.kron(Lambda, np.ones([M,M], dtype=np.float32))
            Lambda_mat = np.kron(Lambda, np.ones([M,M]))
            Lambda_ = permut(Lambda_mat, mat_p)

            # Nu_mat = np.kron(Nu, np.ones([M,M], dtype=np.float32))
            Nu_mat = np.kron(Nu, np.ones([M,M]))
            Nu_ = permut(Nu_mat, mat_p)

            for j_ in range(int(M)):
                j = (p-1)*M +j_
                ##partition matrices:
                (Theta11, Theta12, Theta21, Theta22) = parti(Theta_,j)
                Theta11_inv = np.linalg.inv(Theta11)[:(p-1)*M,:(p-1)*M]

                (S11, S12, S21, S22) = parti(S_,j)
                (Lambda11, Lambda12, Lambda21, Lambda22) = parti(Lambda_, j)
                (Nu11, Nu12, Nu21, Nu22) = parti(Nu_, j)
                
                gamma = np.random.gamma(shape = N/2+1, scale = 2/(S22))

                Ell = np.linalg.cholesky((S22) * Theta11_inv+ np.diag(1/(tau_sq*Lambda12[:(p-1)*M]))).transpose()
                temp1 = np.linalg.solve(Ell.transpose(), -1*S21[:(p-1)*M])
                mu = np.linalg.solve(Ell, temp1)
                # vee = np.linalg.solve(Ell, np.random.normal(size = (p-1)*M).astype(np.float32))
                vee = np.linalg.solve(Ell, np.random.normal(size = (p-1)*M))
                beta = mu + vee
                # temp = np.append(beta, np.zeros(M-1, dtype=np.float32))
                temp = np.append(beta, np.zeros(M-1))
                Theta_[:,j][np.arange(len(Theta_))!=j] = temp
                Theta_[j,:][np.arange(len(Theta_))!=j] = temp
                Theta_[j,j] = gamma + np.sum(beta*np.matmul(Theta11_inv, beta))

            Theta = permut(Theta_, mat_p)
        ##update F_norm
        Theta_F = F_norm(Theta, p, M)

        #update Lambda
        Lambda = sampleLambda(Theta_F, Nu, tau_sq, M)

        #update Nu:
        Nu = sampleNu(Lambda)

        #update tau
        tau_sq = 1/np.random.gamma(shape = (M**2*p*(p-1)+2)/4, scale = 1/(1/zeta+ np.sum(np.triu(Theta_F**2/(2*Lambda),1))))

        ##update zeta
        zeta = 1/np.random.gamma(shape = 1, scale = 1/(1+1/tau_sq))

        if it>0:
            samples.append(Theta)

        # if it%5000 is 0:
        #     last_sample = [Theta, tau_sq, zeta, Nu, Lambda]
        #     folder = '/scratch1/jiajinn/ADNI/'
        #     filename = folder+'ADNI_Last_sample_Net_{}_{}_it{}.npz'.format(data_type, TBI, it)
        #     np.savez(filename, *lastsample)

    samples = np.stack(samples, axis=0)

    last_sample = [Theta, tau_sq, zeta, Nu, Lambda]

    return (samples, last_sample)        

