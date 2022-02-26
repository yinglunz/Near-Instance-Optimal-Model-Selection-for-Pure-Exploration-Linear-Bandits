import numpy as np
import logging
import time

max_iter = 1000
power_para = 4
l_start = 1

def get_r_d(d, zeta, support):
    return 2*support/zeta

def build_Y(Z):
    k, d = Z.shape[0], Z.shape[1]
    idxs = np.zeros((k*k,2))
    Y = np.zeros((k*k, d))
    rangeidx = np.array(list(range(k)))
    for i in range(k):
        idxs[k*i:k*(i+1),0] = rangeidx
        idxs[k*i:k*(i+1),1] = i
        Y[k*i:k*(i+1),:] = Z - Z[i,:]
    Yhat = Y
    return Yhat

def get_design(X, Z):
    # X represents the action set and Z represents the target set
    K = X.shape[0]
    K_Z = Z.shape[0]
    if K_Z < 2:
        raise ValueError('need to calc design for at least 2 arms')
    rand_idx = np.random.randint(K)
    design = np.zeros(K)
    design[rand_idx] = 1
    Yhat = build_Y(Z)
    for count in range(1, max_iter+1):
        design[design < 1e-5] = 1e-5
        A_inv = np.linalg.pinv(X.T@np.diag(design)@X)
        iota = np.einsum('ij,ji->i', Yhat, A_inv@Yhat.T)
        idx = np.argmax(iota)
        y = Yhat[idx, :, None]
        g = ((X@A_inv@y)*(X@A_inv@y)).flatten()
        g_idx = np.argmax(g)
        gamma = 2/(count+2)
        design_update = -gamma*design
        design_update[g_idx] += gamma
        relative = np.linalg.norm(design_update)/(np.linalg.norm(design))
        design += design_update
        if relative < 0.01:
                break
    del Yhat
    design[design <= 1e-5] = 0
    return iota[idx], design

def get_allocation(design, num_samples):
    num_support = (design > 0).sum()
    support_idx = np.where(design>0)[0]
    support = design[support_idx]
    n_round = np.ceil((num_samples - .5*num_support)*support)

    while n_round.sum()-num_samples != 0:
        if n_round.sum() < num_samples:
            idx = np.argmin(n_round/support)
            n_round[idx] += 1
        else:
            idx = np.argmax((n_round-1)/support)
            n_round[idx] -= 1

    allocation = np.zeros(len(design))
    allocation[support_idx] = n_round

    return allocation.astype(int)

def get_pulls_and_rewards(allocation, X, true_rewards, binary, stddev):
    pulls = np.vstack([np.tile(X[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])
    rewards = np.vstack([np.tile(true_rewards[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])
    if binary:
        rewards = np.clip(rewards, 0, 1)
        rewards = np.random.binomial(1, rewards, size=(allocation.sum(), 1))
    else:
        rewards = rewards + np.random.normal(0, stddev, size=(allocation.sum(), 1))
    return pulls, rewards

class RAGE(object):
    # RAGE algorithm from https://proceedings.neurips.cc/paper/2019/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html
    def __init__(self, the_problem):
        self.d_star = the_problem['d_star']
        self.D = the_problem['D']
        self.X = the_problem['X']
        self.Z = the_problem['Z']
        self.K_X = the_problem['K_X']
        self.K_Z = the_problem['K_Z']
        self.true_rewards = the_problem['true_rewards']
        self.opt_reward = the_problem['opt_reward']
        self.opt_arm = the_problem['opt_arm']
        self.T = the_problem['T']
        self.delta = the_problem['delta']
        self.zeta = the_problem['zeta']
        self.tol = the_problem['tol']
        self.binary = the_problem['binary']
        self.stddev = the_problem['stddev']
        self.verifiable_complexity = self.T
        self.unverifiable_complexity = self.T
        self.emp_opt_arm = (self.opt_arm - 1)%self.K_Z
        self.verifiable_success = 0
        self.unverifiable_success = 0
        self.verifiable_time = 0
        self.unverifiable_time = 0

    def run_alg(self):
        self.time_start = time.time()
        self.active_arms = list(range(len(self.Z)))
        self.K_Z_active = len(self.active_arms)
        self.Z_remaining = self.Z[self.active_arms]
        self.arm_counts = np.zeros(self.K_X)
        self.N = 0
        self.phase_idx = 1

        while (len(self.active_arms) > 1) and (self.N < self.T):
            self.delta_t = self.delta/(self.phase_idx**2)
            iota, design = get_design(self.X, self.Z_remaining)
            support = int(np.sum(design>0))
            self.r_d = get_r_d(self.D, self.zeta, support)
            num_samples = int(np.ceil(max( np.ceil(2* max((2**(2*self.phase_idx))*iota,1) *(1+self.zeta) * max(np.log(self.K_Z_active**2/self.delta_t),1) ), self.r_d)))
            if self.N + num_samples >= self.T:
                num_samples = min(self.T, num_samples)
                # we allow additional samples finish the last round
            self.N += num_samples
            allocation = get_allocation(design, num_samples)
            pulls, rewards = get_pulls_and_rewards(allocation, self.X, self.true_rewards, self.binary, self.stddev)
            self.A_inv = np.linalg.pinv(pulls.T@pulls)
            self.theta_hat = self.A_inv@pulls.T@rewards
            self.emp_opt_arm = self.active_arms[np.argmax(self.Z_remaining@self.theta_hat)]
            self.drop_arms()
            self.calc_complexity()

            print('RAGE (end of) phase = {}, total samples = {}'.format(self.phase_idx, self.N))
            print('RAGE: emp opt arm = {}, opt arm = {}'.format(self.emp_opt_arm, self.opt_arm))
            self.phase_idx += 1
            self.arm_counts += allocation
            logging.info('RAGE: finished phase %s' % str(self.phase_idx-1))
            logging.info('RAGE: active arms %s'% str(self.active_arms))

        logging.critical('RAGE: verifiable success = %s' % str(self.verifiable_success))
        logging.critical('RAGE: verifiable complexity %s' % str(self.verifiable_complexity))
        logging.critical('RAGE: verifiable time %s' % str(self.verifiable_time))
        logging.critical('RAGE: unverifiable success = %s' % str(self.unverifiable_success))
        logging.critical('RAGE: unverifiable complexity %s' % str(self.unverifiable_complexity))
        logging.critical('RAGE: unverifiable time %s' % str(self.unverifiable_time))

    def drop_arms(self):
        active_arms = self.active_arms.copy()
        K_Z_active = len(active_arms)
        for arm_idx in active_arms:
            arm = self.Z[arm_idx, :, None]
            for arm_idx_prime in active_arms:
                if arm_idx == arm_idx_prime:
                    continue
                arm_prime = self.Z[arm_idx_prime, :, None]
                y = arm_prime - arm
                if np.sqrt(2*np.abs(y.T@self.A_inv@y)* max(np.log(K_Z_active**2/self.delta_t),1)) <= y.T@self.theta_hat:
                    self.active_arms.remove(arm_idx)
                    break
        self.K_Z_active = len(self.active_arms)
        self.Z_remaining = self.Z[self.active_arms]

    def calc_complexity(self):

        if len(self.active_arms) == 1:
            self.verifiable_complexity = min(self.N, self.T)
            # count up to T samples even though additional samples are 
            # allowed to finish the last round
            self.verifiable_time = time.time() - self.time_start
            if self.active_arms[0] == self.opt_arm:
                self.verifiable_success = 1
        else: 
            self.verifiable_time = 0
            # 0 indicates that the algorithm doesn't verifiably terminate

        if self.emp_opt_arm == self.opt_arm:
            if self.unverifiable_success == 0:
                self.unverifiable_complexity = min(self.N, self.T)
                # count up to T samples even though additional samples are 
                # allowed to finish the last round
                self.unverifiable_time = time.time() - self.time_start
            self.unverifiable_success = 1
        else:
            self.unverifiable_complexity = self.T
            self.unverifiable_success = 0
            self.unverifiable_time = time.time() - self.time_start
    
    def get_results(self):
        results = {}
        results['unverifiable_success'] = self.unverifiable_success
        results['unverifiable_complexity'] = self.unverifiable_complexity
        results['verifiable_success'] = self.verifiable_success
        results['verifiable_complexity'] = self.verifiable_complexity
        results['unverifiable_time'] = self.unverifiable_time
        results['verifiable_time'] = self.verifiable_time
        return results

def test_dimension(X, Z, k, zeta, budget, d):
    X_d = X[:, :d]
    Z_d = Z[:, :d]
    value, design = get_design(X_d, Z_d)
    support = int(np.sum(design>0))
    r_d = get_r_d(d, zeta, support)
    opt_value = 2**(2*k) * value
    g_d = max(opt_value, r_d)
    filter = g_d <= budget
    return filter

def binary_search_d(X, Z, k, zeta, budget, D):
    # in experiments, we search d_k in a binary search manner
    rr = range(1, D+1)
    left, right = rr[0], rr[-1]
    def binary_search(left, right):
        if left == right - 1:
            filter = test_dimension(X, Z, k, zeta, budget, right)
            if filter:
                return right
            else:
                return left
        else:
            mid = int((left + right) / 2)
            filter = test_dimension(X, Z, k, zeta, budget, mid)
            if filter:
                return binary_search(mid, right)
            else:
                return binary_search(left, mid)
    
    d_k = binary_search(left, right)
    break_signal = 0
    filter = test_dimension(X, Z, k, zeta, budget, d_k)
    if filter == 0:
        break_signal = 1
    
    return d_k, break_signal

class MS_GEMS(object):
    def __init__(self, the_problem):
        self.X_original = the_problem['X']
        self.K_X = the_problem['K_X']
        self.Z_original = the_problem['Z']
        self.K_Z = the_problem['K_Z']
        self.true_rewards = the_problem['true_rewards']
        self.opt_reward = the_problem['opt_reward']
        self.opt_arm = the_problem['opt_arm']
        self.active_arms_D = list(range(len(self.Z_original)))
        # active arms maintained in full dimensional space
        self.K_Z_active_D = len(self.active_arms_D)
        self.D = the_problem['D']
        self.d_star = the_problem['d_star']
        self.T = the_problem['T']
        self.delta = the_problem['delta']
        self.zeta = the_problem['zeta']
        self.tol = the_problem['tol']
        self.binary = the_problem['binary']
        self.stddev = the_problem['stddev']
        self.emp_opt_arm = (self.opt_arm - 1)%self.K_Z
        self.opt_arm_located = self.emp_opt_arm

        self.unverifiable_success_s = 0
        self.unverifiable_complexity_s = self.T
        self.unverifiable_time_s = 0
        # unverifiable quantities after a singleton set is located

        self.unverifiable_success = 0
        self.unverifiable_complexity = self.T
        self.unverifiable_time = 0
        # empirical version based on empirical best arm
        # the same as the _s version after a singleton set is located

        self.break_signal = 0
        # break signal for small B_i
        self.singleton_signal = 0
        # singnal for singleton set identification

    def calc_complexity(self):

        if self.opt_arm_located == self.opt_arm:
            if self.unverifiable_success_s == 0:
                self.unverifiable_complexity_s = min(self.N_total, self.T)
                self.unverifiable_time_s = time.time() - self.time_start
            self.unverifiable_success_s = 1
        else:
            self.unverifiable_complexity_s = self.T
            self.unverifiable_success_s = 0
            self.unverifiable_time_s = time.time() - self.time_start
        
        if self.singleton_signal == 1:
            if self.opt_arm_located == self.opt_arm:
                if self.unverifiable_success == 0:
                    self.unverifiable_complexity = min(self.N_total, self.T)
                    self.unverifiable_time = time.time() - self.time_start
                self.unverifiable_success = 1
            else:
                self.unverifiable_complexity = self.T
                self.unverifiable_success = 0
                self.unverifiable_time = time.time() - self.time_start

    def calc_complexity_emp(self):

        if self.emp_opt_arm == self.opt_arm:
            if self.unverifiable_success == 0:
                self.unverifiable_complexity = min(self.N_total, self.T)
                self.unverifiable_time = time.time() - self.time_start
            self.unverifiable_success = 1
        else:
            self.unverifiable_complexity = self.T
            self.unverifiable_success = 0
            self.unverifiable_time = time.time() - self.time_start
        
    def run_alg(self):
        self.time_start = time.time()
        self.N_total = 0
        self.N_total_remaining = self.T
        self.l_start = l_start
        self.l = l_start

        while (self.N_total_remaining > 0) and (self.K_Z_active_D > 1):
            self.gamma_l = power_para ** self.l
            self.delta_l = self.delta/(2 * (self.l-self.l_start+1)**3)

            for i in range(self.l):
                self.B_i = int(power_para ** (self.l - i))
                self.n_i = int(power_para ** i)
                S = self.GEMS(self.B_i, self.n_i, self.delta_l)

                if (self.N_total_remaining <= 0) or (self.K_Z_active_D <= 1):
                    break

                if self.break_signal == 1:
                    self.break_signal = 0
                    break
                
                if len(S) == 1:
                    self.singleton_signal = 1
                    self.opt_arm_located = S[0]
                    break
            self.calc_complexity()
            
            if (self.N_total_remaining <= 0) or (self.K_Z_active_D <= 1):
                break

            logging.info('MS: finished phase l = %s' % str(self.l))
            logging.info('MS: unverifiable success = %s' % str(self.unverifiable_success))
            logging.info('MS: unverifiable complexity = %s' % str(self.unverifiable_complexity))
            logging.info('MS: unverifiable time = %s' % str(self.unverifiable_time))
            self.l += 1
        
        logging.critical('MS: unverifiable success = %s' % str(self.unverifiable_success))
        logging.critical('MS: unverifiable complexity= %s' % str(self.unverifiable_complexity))
        logging.critical('MS: unverifiable time= %s' % str(self.unverifiable_time))
        logging.critical('MS: unverifiable success (singleton) = %s' % str(self.unverifiable_success_s))
        logging.critical('MS: unverifiable complexity (singleton) = %s' % str(self.unverifiable_complexity_s))
        logging.critical('MS: unverifiable time (singleton) = %s' % str(self.unverifiable_time_s))

    def GEMS(self, budget, rounds, delta):

        self.X = self.X_original.copy()
        self.Z = self.Z_original.copy()
        self.active_arms = self.active_arms_D.copy()
        self.K_Z_active = len(self.active_arms)
        self.arm_counts = np.zeros(self.K_Z)
        self.N = 0
        self.phase_idx = 1

        while (self.phase_idx <= rounds) and (self.K_Z_active > 1) and (self.K_Z_active_D > 1) and (self.N_total_remaining > 0):
            
            self.delta_t = delta/(self.phase_idx**2)
            self.Z_remaining = self.Z_original[self.active_arms]
            self.d_k, break_signal = binary_search_d(self.X_original, self.Z_remaining, self.phase_idx, self.zeta, budget, self.D)
            if break_signal == 1:
                self.break_signal = 1
                break
            self.X = self.X_original[:, :self.d_k]
            self.Z = self.Z_original[:, :self.d_k]
            self.Z_remaining = self.Z[self.active_arms]

            iota, design = get_design(self.X, self.Z_remaining)
            support = int(np.sum(design > 0))
            r_d = get_r_d(self.d_k, self.zeta, support)
            opt_value = max(2**(2*self.phase_idx) * iota,1)
            g_d = max(opt_value, r_d)
            
            num_samples = int(np.ceil(g_d * 2 * (1+self.zeta) * \
                max(np.log(self.K_Z_active**2/self.delta_t),1) ))
            
            if num_samples > self.N_total_remaining:
                num_samples = min(num_samples, self.T)
                # we allow additional samples finish the last round
                # same treatment as RAGE
            
            self.N_total_remaining -= num_samples
            self.N += num_samples
            self.N_total += num_samples

            allocation = get_allocation(design, num_samples)
            pulls, rewards = get_pulls_and_rewards(allocation, self.X, self.true_rewards,\
                 self.binary, self.stddev)
            self.A_inv = np.linalg.pinv(pulls.T@pulls)
            self.theta_hat = self.A_inv@pulls.T@rewards
            self.emp_opt_arm = self.active_arms[np.argmax(self.Z_remaining@self.theta_hat)]
            self.drop_arms()

            pulls_D, _ = get_pulls_and_rewards(allocation, self.X_original, self.true_rewards, self.binary, self.stddev)
            self.A_inv_D = np.linalg.pinv(pulls_D.T@pulls_D + 1e-5 * np.eye(self.D))
            self.theta_hat_D = self.A_inv_D@pulls_D.T@rewards
            self.drop_arms_D()
            # additionally conduct elimination in full dimensional space
            # which is guaranteed to perserve linearity

            if self.singleton_signal == 0:
                self.calc_complexity_emp()
                # calcualte sample complexity before a singleton has been located
            
            self.phase_idx += 1
            self.arm_counts += allocation

            del pulls
            del self.theta_hat
            del pulls_D
            del self.theta_hat_D
            del self.Z_remaining

        del self.X
        del self.Z

        return self.active_arms

    def drop_arms(self):

        active_arms = self.active_arms.copy()
        K_Z_active = len(active_arms)
        for arm_idx in active_arms:
            arm = self.Z[arm_idx, :, None]
            for arm_idx_prime in active_arms:
                if arm_idx == arm_idx_prime:
                    continue
                arm_prime = self.Z[arm_idx_prime, :, None]
                y = arm_prime - arm
                if np.sqrt(2*np.abs(y.T@self.A_inv@y)* max(np.log(2*K_Z_active**2/self.delta_t),1)) <= y.T@self.theta_hat:
                    self.active_arms.remove(arm_idx)
                    break
        self.K_Z_active = len(self.active_arms)

    def drop_arms_D(self):

        active_arms = self.active_arms_D.copy()
        K_Z_active_D = len(active_arms)
        for arm_idx in active_arms:
            arm = self.Z_original[arm_idx, :, None]
            for arm_idx_prime in active_arms:
                if arm_idx == arm_idx_prime:
                    continue
                arm_prime = self.Z_original[arm_idx_prime, :, None]
                y = arm_prime - arm
                if np.sqrt(2*np.abs(y.T@self.A_inv_D@y)*max(np.log(2*K_Z_active_D**2/self.delta_t),1)) <= y.T@self.theta_hat_D:
                    self.active_arms_D.remove(arm_idx)
                    break
        self.K_Z_active_D = len(self.active_arms_D)
    
    def get_results(self):
        results = {}
        results['unverifiable_success'] = self.unverifiable_success
        results['unverifiable_complexity'] = self.unverifiable_complexity
        results['unverifiable_success_s'] = self.unverifiable_success_s
        results['unverifiable_complexity_s'] = self.unverifiable_complexity_s
        results['unverifiable_time'] = self.unverifiable_time
        results['unverifiable_time_s'] = self.unverifiable_time_s
        return results

