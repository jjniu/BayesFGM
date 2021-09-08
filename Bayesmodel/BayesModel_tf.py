##Set up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import norm
from sklearn.covariance import GraphicalLassoCV
from tqdm import tqdm

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
tfd = tfp.distributions

sns.set()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

class Timer:
    def __enter__(self):
        self.t0 = time.time()
    def __exit__(self, *args):
        print('Elapsed time: %0.2fs' % (time.time()-self.t0))

def Bayes_fglasso(data, p, regular_parm = None, lambda_shape = 1, lambda_rate = 0.01, nBurnin = 1e3, nIter = 10e3):

    ##blockwise Frobunius norm for precision matrix
    def F_norm(Theta, p, M): 
        matx = np.kron(np.diag(np.ones(p)), np.ones(M)).transpose()
        matx = tf.constant(matx)
        Theta_F = tf.linalg.matmul(tf.linalg.matmul(matx, tf.math.square(Theta), transpose_a = True, a_is_sparse = True), matx, b_is_sparse = True)
        return tf.math.sqrt(Theta_F)

    def sampleLambda(Theta, Tau_sq):
        shape_new = lambda_shape + p*M +(M**2+1)*p*(p-1)/4
        rate_new = lambda_rate + tf.math.reduce_sum(tf.linalg.diag_part(Theta))/2+ tf.math.reduce_sum(tf.linalg.set_diag(tf.linalg.band_part(Tau_sq,0,-1)[0,:,:], np.zeros(p)))/2
        lamb_sq = np.random.gamma(shape = shape_new, scale = 1/rate_new)
        return lamb_sq

    def sampleTau(Theta_F, regular_parm): 
        if tf.math.reduce_min(Theta_F)<1e-6:
            Theta_F = Theta_F + 1e-6
        tau_ =tfd.InverseGaussian(loc = tf.math.divide(regular_parm, Theta_F), concentration = regular_parm**2).sample(1)
        return tf.math.divide(1,tau_)

    def permut(Mat, mat_p):
        return tf.linalg.matmul(tf.linalg.matmul(mat_p, Mat), mat_p)

    def parti(Mat, j):
        #functional for partitioning matirx
        exclude_row = tf.concat([Mat[:j,:], Mat[(j+1):,:]], axis=0)
        Mat11 = tf.concat([exclude_row[:,:j], exclude_row[:,(j+1):]], axis=1)
        Mat12 = tf.concat([Mat[:,j][:j], Mat[:,j][(j+1):]], axis=0)
        Mat21 = Mat12
        Mat22 = Mat[j,j]
        return (Mat11, Mat12, Mat21, Mat22)

    # def is_pos_def(x):
    #     return np.all(np.linalg.eigvals(x) > 0)
    N = data.shape[0]
    M = int(data.shape[1]/p)

    ##centralize data
    data = data - np.mean(data, axis = 0).reshape((1,p*M))
    
    S = np.matmul(data.transpose(), data)
    S = tf.constant(S)

    ### Use glasso with CV to get initial values:
    glasso_model = GraphicalLassoCV(cv =5)## you can also use identity as initial
    glasso_model.fit(data)

    #initial values
    Theta = glasso_model.precision_
    Theta = tf.constant(Theta)

    Theta_F = F_norm(Theta, p, M)

    Tau_sq = sampleTau(Theta_F, regular_parm)

    o1 = tf.linalg.LinearOperatorFullMatrix(Tau_sq[0,:,:])
    o2 = tf.linalg.LinearOperatorFullMatrix(np.ones(shape = (M,M)))
    Tau = tf.linalg.LinearOperatorKronecker([o1, o2]).to_dense()

    samples = []
    lambda_sq = []
    
    for it in tqdm(range(-int(nBurnin), int(nIter)+1, 1)):
        for i in range(p):
            ##create permutation matrix for exchange ith block and pth block
            m = np.diag(np.ones(p))
            m[:,[i,p-1]] = m[:,[p-1,i]]
            m1 = tf.linalg.LinearOperatorFullMatrix(m)
            m2 = tf.linalg.LinearOperatorFullMatrix(np.diag(np.ones(M)))
            mat_p = tf.linalg.LinearOperatorKronecker([m1, m2]).to_dense()
            
            #exchange the ith and pth node
            Theta_ = permut(Theta, mat_p)
            S_ = permut(S, mat_p)
            Tau_ = permut(Tau, mat_p)

            ##for every principal component
            for j_ in range(int(M)):
                j = (p-1)*M +j_            
                ##partition matrices:
                (Theta11, Theta12, Theta21, Theta22) = parti(Theta_,j)
                Theta11_inv = tf.linalg.inv(Theta11)[:(p-1)*M,:(p-1)*M]

                (S11, S12, S21, S22) = parti(S_,j)
                (Tau11, Tau12, Tau21, Tau22) = parti(Tau_,j)

                gamma = np.random.gamma(shape = N/2+1, scale = 2/(S22+regular_parm**2))
                Ell = tf.linalg.cholesky((S22+regular_parm**2)*Theta11_inv + tf.linalg.diag(1/Tau12[:(p-1)*M]))
                temp1 = tf.linalg.solve(Ell,tf.expand_dims(-1*S21[:(p-1)*M],axis = 1))
                mu = tf.linalg.solve(tf.transpose(Ell), temp1)

                vee = tf.linalg.solve(tf.transpose(Ell),tf.expand_dims(tf.constant(np.random.normal(size = mu.shape[0])), axis = 1))
                beta = mu+vee

                aa = np.zeros(M)
                aa[j_] = gamma + tf.math.reduce_sum(beta*tf.linalg.matmul(Theta11_inv, beta))
                aa = tf.constant(aa)
                temp = tf.concat([beta, tf.expand_dims(aa, axis = 1)], axis = 0)

                ##update jth column and jth row of Theta_
                Theta_ = tf.concat([Theta_[:,:j], temp, Theta_[:,j+1:]], axis = 1)
                Theta_ = tf.concat([Theta_[:j,:], tf.transpose(temp), Theta_[j+1:,:]], axis = 0)
            Theta = permut(Theta_, mat_p)

        #update Tau
        Tau_sq = sampleTau(Theta_F, regular_parm)
        o1 = tf.linalg.LinearOperatorFullMatrix(Tau_sq[0,:,:])
        o2 = tf.linalg.LinearOperatorFullMatrix(np.ones(shape = (M,M)))
        Tau_o = tf.linalg.LinearOperatorKronecker([o1, o2])
        Tau = Tau_o.to_dense()

        ##update Theta_F
        Theta_F = F_norm(Theta, p, M)

        ##  update lambda
        if regular_parm is None:
            regular_parm = sampleLambda(Theta, Tau_sq)
        #Store:
        if it>0:
            samples.append(Theta)
            if regular_parm is None:
                lambda_sq.append(regular_parm)

    samples = tf.stack(samples, axis=0)
    return (samples, lambda_sq)


def Bayes_fghorse(data, p, nBurnin = 1e3, nIter = 10e3):
  
    ##blockwise Frobunius norm for precision matrix
    def F_norm(Theta, p, M): 
        matx = np.kron(np.diag(np.ones(p, dtype=np.float32)), np.ones(M, dtype=np.float32)).transpose()
        matx = tf.constant(matx, dtype = tf.float32)
        Theta_F = tf.linalg.matmul(tf.linalg.matmul(matx, tf.math.square(Theta), transpose_a = True, a_is_sparse = True), matx, b_is_sparse = True)
        return tf.math.sqrt(Theta_F)

    def sampleLambda(Theta_F, Nu, tau_sq):
        gamma_lambda =  tfd.Gamma((1+M**2)/2, tf.math.divide(1, Nu)+ tf.math.scalar_mul(1/(2*tau_sq),tf.math.square(Theta_F)))
        return tf.math.divide(1, gamma_lambda.sample(1)[0,:,:])

    def sampleNu(Lambda):
        Nu_gamma = tfd.Gamma(1, 1 + tf.math.divide(1, Lambda))
        return tf.math.divide(1, Nu_gamma.sample(1)[0,:,:])

    def permut(Mat, mat_p):
        return tf.linalg.matmul(tf.linalg.matmul(mat_p, Mat), mat_p)

    def parti(Mat, j):
        #functional for partitioning matirx
        exclude_row = tf.concat([Mat[:j,:], Mat[(j+1):,:]], axis=0)
        Mat11 = tf.concat([exclude_row[:,:j], exclude_row[:,(j+1):]], axis=1)
        Mat12 = tf.concat([Mat[:,j][:j], Mat[:,j][(j+1):]], axis=0)
        Mat21 = Mat12
        Mat22 = Mat[j,j]
        return (Mat11, Mat12, Mat21, Mat22)

    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    N = data.shape[0]
    M = int(data.shape[1]/p)

    ##centralize data
    data = data - np.mean(data, axis = 0).reshape((1,p*M))
    
    S = np.matmul(data.transpose(), data)
    S = tf.constant(S)

    ### Use glasso with CV to get initial values:
    glasso_model = GraphicalLassoCV(cv =5)
    glasso_model.fit(data)

    #initial values
    Theta = glasso_model.precision_ ##you can also use identity as initial 

    Theta = tf.constant(Theta, dtype = tf.float32)
    Theta_F = F_norm(Theta, p, M)
    tau_sq = 1
    zeta = 1
    Nu = tf.ones([p,p], dtype = tf.float32)
    Lambda = sampleLambda(Theta_F, Nu, tau_sq)
    lambda_ = 1 ##diagonal 
    Nu = sampleNu(Lambda)

    samples = []

    for it in tqdm(range(-int(nBurnin), int(nIter)+1, 1)):
        for i in range(p):
              ##create permutation matrix for exchange ith block and pth block
              m = np.diag(np.ones(p)).astype(np.float32)
              m[:,[i,p-1]] = m[:,[p-1,i]]
              m1 = tf.linalg.LinearOperatorFullMatrix(m)
              m2 = tf.linalg.LinearOperatorFullMatrix(np.diag(np.ones(M, dtype = np.float32)))
              mat_p = tf.linalg.LinearOperatorKronecker([m1, m2]).to_dense()
              
              #exchange the ith and pth node
              Theta_ = permut(Theta, mat_p)
              S_ = permut(S, mat_p)

              m1 = tf.linalg.LinearOperatorFullMatrix(Lambda)
              m2 = tf.linalg.LinearOperatorFullMatrix(np.ones([M,M], dtype=np.float32))
              Lambda_mat = tf.linalg.LinearOperatorKronecker([m1, m2]).to_dense()
              Lambda_= permut(Lambda_mat, mat_p)

              m1 = tf.linalg.LinearOperatorFullMatrix(Nu)
              m2 = tf.linalg.LinearOperatorFullMatrix(np.ones([M,M], dtype=np.float32))
              Nu_mat = tf.linalg.LinearOperatorKronecker([m1, m2]).to_dense()
              Nu_= permut(Nu_mat, mat_p)

              for j_ in range(int(M)):
                    j = (p-1)*M +j_
                    ##partition matrices:
                    (Theta11, Theta12, Theta21, Theta22) = parti(Theta_,j)
                    Theta11_inv = tf.linalg.inv(Theta11)[:(p-1)*M,:(p-1)*M]

                    (S11, S12, S21, S22) = parti(S_,j)
                    (Lambda11, Lambda12, Lambda21, Lambda22) = parti(Lambda_, j)
                    (Nu11, Nu12, Nu21, Nu22) = parti(Nu_, j)
                    
                    gamma = np.random.gamma(shape = N/2+1, scale = 2/(S22+lambda_**2))

                    Ell = tf.linalg.cholesky((S22 + lambda_**2) * Theta11_inv + tf.linalg.diag(1/(tau_sq*Lambda12[:(p-1)*M])))
                    temp1 = tf.linalg.solve(Ell, tf.expand_dims(-1*S21[:(p-1)*M], axis = 1))
                    mu = tf.linalg.solve(tf.transpose(Ell), temp1)

                    vee = tf.linalg.solve(tf.transpose(Ell),tf.expand_dims(tf.constant(np.random.normal(size = mu.shape[0]), dtype=np.float32), axis = 1))
                    beta = mu+vee
                    
                    aa = np.zeros(M, dtype=np.float32)
                    aa[j_] = gamma + tf.math.reduce_sum(beta*tf.linalg.matmul(Theta11_inv, beta))
                    aa = tf.constant(aa)
                    temp = tf.concat([beta, tf.expand_dims(aa, axis = 1)], axis = 0)
                    ##update jth column and jth row of Theta_
                    Theta_ = tf.concat([Theta_[:,:j], temp, Theta_[:,j+1:]], axis = 1)
                    Theta_ = tf.concat([Theta_[:j,:], tf.transpose(temp), Theta_[j+1:,:]], axis = 0)
              Theta = permut(Theta_, mat_p)
        
        ##update F_norm
        Theta_F = F_norm(Theta, p, M)

        #update Lambda
        Lambda = sampleLambda(Theta_F, Nu, tau_sq)

        #update Nu:
        Nu = sampleNu(Lambda)

        #update tau
        up_sum = tf.math.reduce_sum(tf.linalg.set_diag(tf.linalg.band_part(tf.math.divide(tf.math.square(Theta_F), Lambda),0,-1), np.zeros(p, dtype=np.float32)))
        scale_tau =1/(1/zeta+ up_sum.numpy()/2)
        tau_sq = 1/np.random.gamma(shape = (M**2*p*(p-1)+2)/4, scale = scale_tau)
        ##update zeta
        zeta = 1/np.random.gamma(shape = 1, scale = 1/(1+1/tau_sq))

        if it>0:
              samples.append(Theta)

    samples = np.stack(samples, axis=0)
    return samples                  


# prior = "horse"#"lasso"
# p = 100 #number of dimensions
# M = 5
# N = 100 #number of datapoints to generate
# mu = np.zeros(p*M)
# Net_true = np.diag(np.ones(p))+np.diag(0.4*np.ones(p-1),1)+np.diag(0.4*np.ones(p-1),-1)+np.diag(0.2*np.ones(p-2),2)+np.diag(0.2*np.ones(p-2),-2)

# Theta_true = np.kron(Net_true, np.diag(np.ones(M)))
# Sigma_true = np.linalg.inv(Theta_true)
# y = np.random.multivariate_normal(mean = mu, cov = Sigma_true, size = N).astype(np.float32)

# with Timer():
#     if prior == 'horse':
#         samples = Bayes_fghorse(data = y, p = p, nBurnin = 1e3, nIter = 1e3)
#     if prior == "lasso":
#         samples, lambda_sq = Bayes_fglasso(data=y, p=p, regular_parm = 0.1, lambda_shape = 1, lambda_rate = 0.01, nBurnin = 1e3, nIter = 1e3)


# plt.figure()
# plt.matshow(samples[-1,:,:]);
# plt.colorbar()
# plt.savefig("/home/jiajinn/Images/last_sample.png")
