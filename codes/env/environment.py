import numpy as np
from math import sqrt
import sys
import os

sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))

from codes.utils import *
# GF2 tools
from codes.utils.gf2 import outer_mod2, add_mod2, to_bin
from codes.utils.gf2 import terminate_rank_approx_gf2


class Environment():
    '''
    Defines game actions, states, and rewards. 
    Supports both real field (R) and binary field (GF2) modes.
    '''
    def __init__(self,
                 S_size,
                 R_limit,
                 init_state=None,
                 T=7,
                 domain="GF2",  # Domain Selection: "GF2" or "R"
                 threshold=0.5, # Binarization threshold in GF2 mode
                 **kwargs):
        self.S_size = S_size
        self.R_limit = R_limit
        self.T = T
        self.domain = domain
        self.threshold = threshold

        # 初始化状态
        if init_state is None:
            init_state = self.get_init_state(S_size)
        self.cur_state = init_state
        self.accumulate_reward = 0
        self.step_ct = 0
        self.hist_actions = [np.zeros_like(self.cur_state) for _ in range(self.T - 1)]

    def get_init_state(self, S_size, no_base_change=False):
        '''
        Get an initial state: state
        '''
        def one_hot(idx):
            temp = np.zeros((S_size,), dtype=np.int32)
            temp[idx] = 1
            return temp

        init_state = np.zeros((S_size, S_size, S_size), dtype=np.int32)
        n = round(sqrt(S_size))

        for i in range(n):
            for j in range(n):
                z_idx = i * n + j
                z = one_hot(z_idx)
                for k in range(n):
                    x_idx = i * n + k
                    y_idx = k * n + j
                    x, y = one_hot(x_idx), one_hot(y_idx)
                    if self.domain == "GF2":
                        init_state = add_mod2(init_state, outer_mod2(x, y, z))
                    else:
                        init_state = init_state + outer(x, y, z)

        # Change of Basis
        if not no_base_change:
            p0 = .985
            P = np.random.choice([0, 1, -1], size=(S_size, S_size),
                                 p=[p0, (1 - p0) / 2, (1 - p0) / 2], replace=True)
            L = np.random.choice([0, 1, -1], size=(S_size, S_size),
                                 p=[p0, (1 - p0) / 2, (1 - p0) / 2], replace=True)
            for i in range(S_size):
                P[i, i] = np.random.choice([1, -1], size=(1,), p=[.5, .5])
                L[i, i] = np.random.choice([1, -1], size=(1,), p=[.5, .5])
            P, L = np.triu(P), np.tril(L)
            trans_mat = np.matmul(P, L)
            init_state = change_basis_tensor(tensor=init_state, trans_mat=trans_mat)

        return init_state

    def step(self, action):
        '''
        The state is transferred and the reward is changed, and it returns whether the game is over.
        '''
        u, v, w = action
        if self.domain == "GF2":
            u = to_bin(u, self.threshold)
            v = to_bin(v, self.threshold)
            w = to_bin(w, self.threshold)
            self.cur_state = add_mod2(self.cur_state, outer_mod2(u, v, w))
        else:
            self.cur_state = self.cur_state + outer(u, v, w)

        self.accumulate_reward -= 1
        self.step_ct += 1
        self.hist_actions.append(action2tensor(action))

        if self.is_terminate():
            return True
        if self.step_ct >= self.R_limit:
            self.accumulate_reward += self.terminate_reward()
            return True
        return False

    def terminate_reward(self):
        '''
        Penalty for truncation
        '''
        state = self.cur_state
        if self.domain == "GF2":
            return -terminate_rank_approx_gf2(state, threshold=self.threshold)
        else:
            return -terminate_rank_approx(state)

    def is_terminate(self):
        '''
        Determine whether cur_state is zero tensor
        '''
        if self.domain == "GF2":
            # GF2 zero check
            return np.all((self.cur_state % 2) == 0)
        else:
            return is_zero_tensor(self.cur_state)

    def reset(self, init_state=None, no_base_change=False):
        '''
        Reset the environment
        '''
        if init_state is None:
            init_state = self.get_init_state(self.S_size, no_base_change)
        self.cur_state = init_state
        self.accumulate_reward = 0
        self.step_ct = 0
        self.hist_actions = [np.zeros_like(self.cur_state) for _ in range(self.T - 1)]

    def get_network_input(self):
        '''
        Organize variables into a format for network input
        '''
        T = self.T
        S_size = self.S_size
        hist_actions = self.hist_actions[-(T - 1):]
        hist_actions.reverse()
        tensors = np.zeros((T, S_size, S_size, S_size), dtype=np.int32)
        tensors[0] = self.cur_state
        tensors[1:] = np.stack(hist_actions, axis=0)
        scalars = np.array([self.step_ct, self.step_ct, self.step_ct])
        return tensors, scalars


if __name__ == '__main__':
    test_env = Environment(S_size=4, R_limit=8, domain="GF2")
    test_action = np.array([
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0]
    ])
    for _ in range(8):
        print(test_env.step(test_action))
        print(test_env.accumulate_reward)
