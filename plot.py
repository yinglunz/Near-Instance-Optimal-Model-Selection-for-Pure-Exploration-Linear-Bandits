import numpy as np
import sys
import pickle
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
from run_algs import algs, algs_label

line_style_list = [(0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), '-.', '-']
marker_list = ['.', '*', '^', 'h', 'P']
c_std = 0.25
font_size = 18
legend_size = 22

def preprocess_data(sim_data, sim_list):
    n_sims = len(sim_data)
    results = {}
    for alg in algs:
        results[alg] = {}
        results[alg]['acc'] = np.array([ [0. for i in range(n_sims)] for i in range(len(sim_list)) ])
        results[alg]['sample_o'] = np.array([ [0. for i in range(n_sims)] for i in range(len(sim_list)) ])
        # sample_o stores the original sample
        results[alg]['sample'] = {}
        results[alg]['time_o'] = np.array([ [0. for i in range(n_sims)] for i in range(len(sim_list)) ])
        # time_o stores the original time
        results[alg]['time'] = {}
    
    for i in range(n_sims):
        for j in range(len(sim_list)):
            for alg in algs:
                results[alg]['acc'][j,i] = sim_data[i][j][alg]['unverifiable_success']
                results[alg]['sample_o'][j,i] = sim_data[i][j][alg]['unverifiable_complexity']
                results[alg]['time_o'][j,i] = sim_data[i][j][alg]['unverifiable_time']
    
    for alg in algs:
        for j in range(len(sim_list)):
            results[alg]['sample'][sim_list[j]] = \
                 results[alg]['sample_o'][j][results[alg]['acc'][j] == 1]
            results[alg]['time'][sim_list[j]] = \
                 results[alg]['time_o'][j][results[alg]['acc'][j] == 1]
        
        results[alg]['ave_acc'] = np.zeros(len(sim_list))
        results[alg]['ave_sample'] = np.zeros(len(sim_list))
        results[alg]['std_sample'] = np.zeros(len(sim_list))
        results[alg]['ave_time'] = np.zeros(len(sim_list))
        results[alg]['std_time'] = np.zeros(len(sim_list))
        results[alg]['ave_sample_o'] = np.zeros(len(sim_list))
        results[alg]['ave_time_o'] = np.zeros(len(sim_list))

        for j in range(len(sim_list)):
            results[alg]['ave_acc'][j] = results[alg]['acc'][j].sum()/n_sims
            x = results[alg]['sample'][sim_list[j]]
            x_o = results[alg]['sample_o'][j]
            results[alg]['ave_sample'][j] = np.mean(x)
            results[alg]['std_sample'][j] = np.std(x)
            results[alg]['ave_sample_o'][j] = np.mean(x_o)
            
            y = results[alg]['time'][sim_list[j]]
            y_o = results[alg]['time_o'][j]
            results[alg]['ave_time'][j] = np.mean(y)
            results[alg]['std_time'][j] = np.std(y)
            results[alg]['ave_time_o'][j] = np.mean(y_o)

    return results

def plot_prop():
    with open('results_prop.dat', 'rb') as f:
        x = pickle.load(f)
    running_time = x['running_time']
    print('total running time is ', running_time)
    the_para = x['the_para']
    print('==== the parameter settings ====')
    print(the_para)
    T = the_para['T']
    sim_data = x['data']
    n_sims = len(sim_data)
    print('n_sims = ', n_sims)
    tol_list = the_para['tol_list']
    results = preprocess_data(sim_data, tol_list)
    for alg in algs:
        print('==== results for {} ===='.format(alg))
        print('tol list = ', tol_list)
        print('ave acc = ', results[alg]['ave_acc'])
        print('ave sample = ', results[alg]['ave_sample'])
        print('std sample = ', results[alg]['std_sample'])
        print('ave sample (all) = ', results[alg]['ave_sample_o'])
        print('ave time = ', results[alg]['ave_time'])
        print('std time = ', results[alg]['std_time'])
        print('ave time (all) = ', results[alg]['ave_time_o'])

    x_axis = range(len(tol_list))
    T_axis = T * np.ones(len(tol_list))
    x_ticks = ['{}'.format(tol) for tol in tol_list]
    x_ticks = [r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$']
    fig = plt.figure(); ax = fig.add_subplot(111)
    plt.xticks(x_axis, x_ticks)
    for i in range(len(algs)):
        ave = results[algs[i]]['ave_sample']
        std = results[algs[i]]['std_sample']
        plt.plot(x_axis, ave, label=algs_label[i], \
            marker = marker_list[i], linestyle = line_style_list[i], linewidth=3, markersize=9)
        plt.fill_between(x_axis, ave-c_std*std, ave+c_std*std, alpha=0.2)
    plt.plot(x_axis, T_axis, linestyle = line_style_list[3], linewidth=2, color='black')
    ax.set_ylabel('Sample Complexity', fontsize=font_size)
    ax.set_xlabel(r'$\epsilon$', fontsize=font_size)
    plt.xlim([x_axis[0]-0.5, x_axis[-1]+0.5])
    plt.yscale('log')
    plt.legend(loc='best', prop={'size':legend_size})
    plt.grid(alpha=0.75)
    plt.savefig('complexity_prop.pdf')
    plt.show()
    plt.close()


def plot_unif():
    with open('results_unif.dat', 'rb') as f:
        x = pickle.load(f)

    running_time = x['running_time']
    print('total running time is ', running_time)
    the_para = x['the_para']
    print('==== the parameter settings ====')
    print(the_para)
    T = the_para['T']
    sim_data = x['data']
    n_sims = len(sim_data)
    print('n_sims = ', n_sims)
    D_list = the_para['D_list']
    results = preprocess_data(sim_data, D_list)
    
    for alg in algs:
        print('==== results for {} ===='.format(alg))
        print('D list = ', D_list)
        print('ave acc = ', results[alg]['ave_acc'])
        print('ave sample = ', results[alg]['ave_sample'])
        print('std sample = ', results[alg]['std_sample'])
        print('ave sample (all) = ', results[alg]['ave_sample_o'])
        print('ave time = ', results[alg]['ave_time'])
        print('std time = ', results[alg]['std_time'])
        print('ave time (all) = ', results[alg]['ave_time_o'])

    x_axis = D_list
    T_axis = T * np.ones(len(D_list))
    x_ticks = ['{}'.format(D) for D in D_list]
    fig = plt.figure(); ax = fig.add_subplot(111)
    plt.xticks(x_axis, x_ticks)
    for i in range(len(algs)):
        ave = results[algs[i]]['ave_sample']
        std = results[algs[i]]['std_sample']
        plt.plot(x_axis, ave, label=algs_label[i], \
            marker = marker_list[i], linestyle = line_style_list[i], linewidth=3, markersize=9)
        plt.fill_between(x_axis, ave-c_std*std, ave+c_std*std, alpha=0.2)
    plt.plot(x_axis, T_axis, linestyle = line_style_list[2], linewidth=2, color='black')
    ax.set_ylabel('Sample Complexity', fontsize=font_size)
    ax.set_xlabel(r'$D$', fontsize=font_size)
    plt.legend(loc='best', prop={'size':legend_size})
    plt.grid(alpha=0.75)
    plt.savefig('complexity_unif.pdf')
    plt.show()
    plt.close()

if __name__ == '__main__':
    instance_idx = int(sys.argv[1])
    if instance_idx == 0:
        plot_prop()
    elif instance_idx == 1:
        plot_unif()
    else:
        raise ValueError('unknown instance index!')

    




