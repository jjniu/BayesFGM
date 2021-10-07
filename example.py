import os
import sys

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter

from Bayesmodel import BayesModel
import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

robjects.r('''
    source("./sim_data_pca.R")
    '''
    )

r_SVD_dense_fun = robjects.globalenv['SVD_dense_fun']
r_pace_sparse_fun = robjects.globalenv['pace_sparse_fun']


def run(p, Net_num, prior, N, data_type):

    if data_type == "sparse":
        data_r = r_pace_sparse_fun(p, N, Net_num)
    if data_type == "dense":
        data_r = r_SVD_dense_fun(p, N, Net_num)  

    y = np.array(data_r[0])

    nBurnin = 1000
    nIter = 10000

    if prior == "lasso":
        if p == 10:
            lambda_shape = 900 ## pre-tuned parameter 
        if p == 30:
            lambda_shape = 6000
        if p == 50:
            lambda_shape = 13000

        samples, _ = BayesModel.Bayes_fglasso(data = y, p = p, lambda_shape = lambda_shape, nBurnin = nBurnin, nIter = nIter)
        
    if prior == "horse":
        
        samples, _ = BayesModel.Bayes_fghorse(data = y, p = p, nBurnin = nBurnin, nIter = nIter)

    filename = './Bayes_fg{}_samples_Net_{}_p_{}_N_{}_{}.npy'.format(prior, Net_num, p, N, data_type)
    np.save(filename, samples)

def main():
    parser = ArgumentParser("deepwalk",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')
    
    parser.add_argument('--p', default=10, type=int, help='number of node')
    
    parser.add_argument('--Net_num', default=1, type=int, help='Network')
    
    parser.add_argument('--prior', default="horse", type=str, help='method')

    parser.add_argument('--N', default=100, type=int,help='sample size')

    parser.add_argument('--data_type', default="dense", type=str, help='sampling scheme')


    args = parser.parse_args()
    
    run(args.p, args.Net_num, args.prior, args.N, args.data_type)


if __name__ == "__main__":
    sys.exit(main())
