import numpy as np
import sys
import multiprocessing
import pickle
import time
from algs_class import *

algs = ['RAGE', 'MS_GEMS']
algs_label = ['RAGE', 'Ours']

def generate_problem(the_para):
    ############ read parameters #############
    print(the_para)
    D = the_para['D']
    d_star = the_para['d_star']
    K = the_para['K']
    T = the_para['T']
    zeta = the_para['zeta']
    delta = the_para['delta']
    tol = the_para['tol']
    binary = the_para['binary']
    stddev = the_para['stddev']
    idx = the_para['instance_idx']
    ##########################################
    if idx == 0:
        return generate_problem_prop(D, d_star, K, T, delta, zeta, tol, binary, stddev)
    elif idx == 1:
        return generate_problem_unif(D, d_star, K, T, delta, zeta, tol, binary, stddev)
    else:
        raise ValueError('unknown idx for problem generation')

def generate_problem_prop(D, d_star, K, T, delta, zeta, tol, binary, stddev):
    X = np.eye(D)
    X[-1, d_star - 1] = 1 - tol
    theta = X[d_star - 1].T
    theta = theta.reshape((-1,1))
    X = X
    Z = X.copy()
    true_rewards = X@theta
    opt_reward = np.max(true_rewards)
    opt_index = np.where(opt_reward - true_rewards < tol)[0]
    opt_arm = opt_index[0]
    the_problem = { 'D': D,
                    'd_star': d_star,
                    'K_X': X.shape[0],
                    'K_Z': Z.shape[0],
                    'T': T,
                    'delta': delta,
                    'zeta': zeta,
                    'tol': tol,
                    'binary': binary,
                    'stddev': stddev,
                    'theta': theta,
                    'X': X,
                    'Z': Z,
                    'true_rewards': true_rewards,
                    'opt_reward': opt_reward,
                    'opt_arm': opt_arm,
                }
    return the_problem

def generate_problem_unif(D, d_star, K, T, delta, zeta, tol, binary, stddev):
    # tol represents a lower bound on the Delta_min
    theta_star = 1/(np.arange(1, d_star+1)**2)
    theta = np.hstack((theta_star, np.zeros(D-d_star)))
    unique = 0
    span = False
    while unique == 0 or span == False:
        X = np.random.randn(D, K)
        X /= np.linalg.norm(X, axis=0)
        span = np.linalg.matrix_rank(X,1e-5) == D
        X = X.T
        true_rewards = X@theta
        opt_reward = np.max(true_rewards)
        opt_index = np.where(opt_reward - true_rewards < tol)[0]
        if len(opt_index) <= 1:
            unique = 1
        opt_arm = opt_index[0]
        true_rewards = X@theta 
    Z = X.copy()
    the_problem = { 'D':D,
                    'd_star': d_star,
                    'K_X': X.shape[0],
                    'K_Z': Z.shape[0],
                    'T': T,
                    'delta': delta,
                    'zeta': zeta,
                    'tol': tol,
                    'binary': binary,
                    'stddev': stddev,
                    'theta': theta,
                    'X': X,
                    'Z': Z,
                    'true_rewards': true_rewards,
                    'opt_reward': opt_reward,
                    'opt_arm': opt_arm,
                }
    return the_problem

def single_sim(the_para):
    results = {}
    algs = the_para['algs']
    the_problem = generate_problem(the_para)
    alg_rage = RAGE(the_problem)
    alg_rage.run_alg()
    results[algs[0]] = alg_rage.get_results()
    alg_MS = MS_GEMS(the_problem)
    alg_MS.run_alg()
    results[algs[1]] = alg_MS.get_results()
    return results

def multi_sim(the_para, seed, idx):
    np.random.seed(seed)
    results = []
    instance_idx = the_para['instance_idx']
    if instance_idx == 0:
        tol_list = the_para['tol_list']
        for tol in tol_list:
            print('start with tol = ', tol)
            the_para_prime = the_para.copy()
            the_para_prime['tol'] = tol
            output = single_sim(the_para_prime)
            results.append(output)
    else:
        D_list = the_para['D_list']
        for D in D_list:
            print('start with D = ', D)
            the_para_prime = the_para.copy()
            the_para_prime['D'] = D
            output = single_sim(the_para_prime)
            results.append(output)
    return results

def multi_sim_prop():
    time_start = time.time()
    delta = 0.05
    zeta = 1
    d_star = 9
    D = 10
    K = 100
    T = 20000000
    tol_list = [1e-2, 1e-3, 1e-4, 1e-5]
    binary = False
    stddev = 1
    n_sims = 100
    the_para = {'instance_idx' : 0,
                    'algs' : algs,
                    'delta' : delta,
                    'zeta' : zeta,
                    'd_star' : d_star,
                    'D' : D,
                    'K' : K,
                    'T' : T,
                    'tol_list' : tol_list,
                    'binary' : binary,
                    'stddev' : stddev,
                    'n_sims' : n_sims
                }
    results = { 'data' : [],
                'the_para' : the_para
    }
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    iters = [pool.apply_async(multi_sim, args=(the_para, np.random.randint(1e6), ind)) for ind in range(n_sims)]
    for result in iters:
        results['data'].append(result.get())
    time_end = time.time()
    total_time = time_end - time_start
    print('total running time prop = ', total_time)
    results['running_time'] = total_time
    with open('results_prop.dat', 'wb') as f:
        pickle.dump(results, f)

def multi_sim_unif():
    time_start = time.time()
    delta = 0.05
    zeta = 1
    d_star = 5
    D_list = [25, 50, 75, 100]
    K = 100
    T = 50000
    tol = 0.08
    binary = False
    stddev = 1
    n_sims = 100
    the_para = {'instance_idx' : 1,
                    'algs' : algs,
                    'delta' : delta,
                    'zeta' : zeta,
                    'd_star' : d_star,
                    'D_list' : D_list,
                    'K' : K,
                    'T' : T,
                    'tol' : tol,
                    'binary' : binary,
                    'stddev' : stddev,
                    'n_sims' : n_sims
                }
    results = { 'data' : [],
                'the_para' : the_para
    }
    pool = multiprocessing.Pool(processes=2*multiprocessing.cpu_count())
    iters = [pool.apply_async(multi_sim, args=(the_para, np.random.randint(1e6), i)) for i in range(n_sims)]
    for result in iters:
        results['data'].append(result.get())
    time_end = time.time()
    total_time = time_end - time_start
    print('total running time unif = ', total_time)
    results['running_time'] = total_time
    with open('results_unif.dat', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    np.random.seed()
    instance_list = ['prop', 'unif']
    instance_idx = int(sys.argv[1])
    if instance_idx == 0:
        multi_sim_prop()
    elif instance_idx == 1:
        multi_sim_unif()
    else:
        raise ValueError('unknown instance idx')


    



