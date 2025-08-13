import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
import os
import copy
import itertools
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from codes.utils import *

class TupleDataset(Dataset):
    
    def __init__(self,
                 T,
                 S_size,
                 N_steps,
                 coefficients,
                 self_data=[],
                 synthetic_data=[],
                 debug=False,
                 save_type="traj"):
        self.T = T
        self.S_size = S_size
        self.N_steps = N_steps
        self.coefficients = coefficients
        token_len = 3 * S_size // N_steps
        self.N_logits = len(coefficients) ** token_len
        self.ct = 0
        self.save_type = save_type
        
        print("Preprocessing dataset...")
        self.self_data = self_data
        self.synthetic_data = synthetic_data
        self.data = self_data + synthetic_data
        self.data_iterer = itertools.cycle(self.data)
        self.self_examples = []
        self.synthetic_examples = []
            
        if save_type == "tuple":
            for episode in tqdm(synthetic_data):
                state, action, reward = episode
                action = self.action_to_logits(canonicalize_action(action))
                self.synthetic_examples.append([state, action, reward])
            for episode in tqdm(self_data):
                state, action, reward = episode
                action = self.action_to_logits(canonicalize_action(action))
                self.self_examples.append([state, action, reward])
            self.examples = self.self_examples + self.synthetic_examples
        else:
            self._prepare_examples_from_trajs()
                   
    def _prepare_examples_from_trajs(self):
        S_size = self.S_size
        T = self.T      
        
        self_examples, synthetic_examples = [], []
        
        for traj in tqdm(self.self_data):
            _traj = copy.deepcopy(traj)
            _states, _actions, _rewards = _traj
            _states.reverse(), _actions.reverse(), _rewards.reverse()
            _traj = [_states, _actions, _rewards]
            new_traj = self.permutate_traj(_traj)
            self_examples.extend(self.traj_to_episode(new_traj))         
        
        for traj in tqdm(self.synthetic_data):
            new_traj = self.permutate_traj(traj)
            synthetic_examples.extend(self.traj_to_episode(new_traj))             
        
        self.self_examples, self.synthetic_examples = self_examples, synthetic_examples
        self.examples = self_examples + synthetic_examples
        
    def _permutate_traj(self, trajs_n=5000):
        assert self.save_type == "traj"
        print("Permutate!")
        for _ in range(trajs_n):
            self_traj = next(self.data_iterer)
            new_traj = self.permutate_traj(self_traj)
            new_episodes = self.traj_to_episode(new_traj)
            n = len(new_episodes)
            self.examples = self.examples[n:] + new_episodes
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):        
        state, action, reward = self.examples[idx]
        tensor, scalar = state
        action = self.logits_to_action(action)
        tensor, action = self.random_sign_permutation(tensor, action)
        action = canonicalize_action(action)
        action = self.action_to_logits(action)
        return [tensor, scalar], action, reward
    
    def traj_to_episode(self, traj):
        results = []
        T, S_size = self.T, self.S_size
        states, actions, rewards = traj
        states = list(reversed(states)); actions = list(reversed(actions)); rewards = list(reversed(rewards))
        actions_tensor = [action2tensor(action) for action in actions]
        for idx, state in enumerate(states):
            tensors = np.zeros((T, S_size, S_size, S_size), dtype=np.int32)
            tensors[0] = state
            if idx != 0:
                tensors[1:(idx+1)] = np.stack(list(reversed(actions_tensor[max(idx-(T-1), 0):idx])), axis=0)        
            scalars = np.array([idx, idx, idx])
            cur_state = [tensors, scalars]
            action = self.action_to_logits(canonicalize_action(actions[idx]))
            reward = rewards[idx]
            results.append([cur_state, action, reward])
        return results
    
    def permutate_traj(self, traj):
        S_size = self.S_size
        states, actions, rewards = traj
        final_state = states[0] - action2tensor(actions[0])
        new_actions = actions.copy()
        np.random.shuffle(new_actions)
        new_states = []
        new_rewards = copy.deepcopy(rewards)
        sample = final_state
        for action in new_actions:
            sample = sample + action2tensor(action)
            new_states.append(sample.copy())
        new_traj = [new_states, new_actions, new_rewards]
        return new_traj
    
    def action_to_logits(self, action):
        token_len = 3 * self.S_size // self.N_steps
        coefficients = self.coefficients
        action = action.reshape((-1, token_len))
        
        logits = []
        for token in action:
            logit = 0
            if torch.is_tensor(token):
                token = torch.flip(token, dims=(0,)).tolist()
            else:
                token = token[::-1]
            for idx, v in enumerate(token):
                v = int(v)
                if v == -1:  # 处理 np.int32(-1)
                    v = 0   # 或根据实际逻辑映射为合法值
                logit += coefficients.index(v) * (len(coefficients) ** idx)
            logits.append(logit)
        return np.array(logits, dtype=np.int32)
    
    def logits_to_action(self, logits):
        token_len = 3 * self.S_size // self.N_steps
        coefficients = self.coefficients
        action = []
        for logit in logits:
            token = []
            if logit == self.N_logits:
                raise
            for _ in range(token_len):
                idx = logit % len(coefficients)
                token.append(coefficients[idx])
                logit = logit // len(coefficients)
            token.reverse()
            action.extend(token)
        action = np.array(action, dtype=np.int32).reshape((3, -1))
        return action    
    
    def random_sign_permutation(self, tensor, action):
        trans_1, trans_2, trans_3 = \
            (np.random.binomial(1, .5, self.S_size) * 2 - 1).astype(np.int32), \
            (np.random.binomial(1, .5, self.S_size) * 2 - 1).astype(np.int32), \
            (np.random.binomial(1, .5, self.S_size) * 2 - 1).astype(np.int32)
        tensor = np.einsum('i, j, k, bijk -> bijk', trans_1, trans_2, trans_3, tensor,
                           dtype=np.int32)
        action = np.stack([action[0]*trans_1, action[1]*trans_2, action[2]*trans_3], axis=0)
        return tensor, action
    
if __name__ == '__main__':
    dataset = TupleDataset(T=7,
                           S_size=4,
                           N_steps=6,
                           coefficients=[0, 1, -1],
                           synthetic_data=np.load("data/traj_data/100000_S4T7_scalar3.npy", allow_pickle=True).tolist(),
                           debug=True)
    import pdb; pdb.set_trace()
