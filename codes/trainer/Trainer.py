import time
import yaml
import copy
import random
import shutil
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from codes.trainer.loss import *
from codes.trainer.Player import *
from codes.env import *
from codes.mcts import *
from codes.utils import *
from codes.dataset import *
from codes.multi_runner import *

class Trainer():
    '''
    For training the network
    '''
    
    def __init__(self,
                 net: Net,
                 env: Environment,
                 mcts: MCTS,
                 S_size=4,
                 T=7,
                 coefficients=[0, 1, -1],
                 batch_size=1024,
                 iters_n=50000,
                 exp_dir="exp",
                 exp_name="debug",
                 device="cuda:0",
                 self_play_device="cuda:1",
                 lr=5e-3,
                 weight_decay=1e-5,
                 step_size=40000,
                 gamma=.1,
                 a_weight=.5,
                 v_weight=.5,
                 save_freq=10000,
                 temp_save_freq=2500,
                 self_play_freq=10,
                 self_play_buffer=100000,
                 grad_clip=4.0,
                 val_freq=2000,
                 all_kwargs=None):
        '''
        Initialize a Trainer.
        Contains net, env and MCTS
        '''
        self.env = env
        self.net = net
        self.mcts = mcts
        self.S_size = S_size
        self.T = T
        self.coefficients = coefficients
        
        self.self_examples = []
        self.synthetic_examples = []
        
        self.entropy_loss = torch.nn.CrossEntropyLoss()
        self.quantile_loss = QuantileLoss()
        self.a_weight = a_weight
        self.v_weight = v_weight
        
        self.optimizer_a = torch.optim.AdamW(net.parameters(),
                                             weight_decay=weight_decay,
                                             lr=lr)
        self.scheduler_a = torch.optim.lr_scheduler.StepLR(self.optimizer_a,
                                                           step_size=step_size,
                                                           gamma=gamma)
        self.optimizer_v = torch.optim.AdamW(net.parameters(),
                                             weight_decay=weight_decay,
                                             lr=lr)
        self.scheduler_v = torch.optim.lr_scheduler.StepLR(self.optimizer_v,
                                                           step_size=step_size,
                                                           gamma=gamma)
        
        self.batch_size = batch_size
        self.iters_n = iters_n
        self.grad_clip = grad_clip
        self.save_freq = save_freq
        self.temp_save_freq = temp_save_freq
        self.self_play_freq = self_play_freq
        self.self_play_buffer = self_play_buffer
        self.val_freq = val_freq
        
        self.exp_dir = exp_dir
        self.save_dir = os.path.join(exp_dir, exp_name, str(int(time.time())))  
        self.log_dir = os.path.join(self.save_dir, "log")
        self.data_dir = os.path.join(self.save_dir, "data")
        
        self.device = device
        self.self_play_device = self_play_device
        self.net.to(device)
        self.all_kwargs = all_kwargs
        
        # Initialize training related indicators
        self.final_train_loss = None
        self.train_steps = 0
        self.peak_train_memory_MB = None

        # Reasoning-related metrics
        self.peak_infer_memory_MB = None

    def save_model(self, filename, step):
        save_path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_a_state_dict': self.optimizer_a.state_dict(),
            'optimizer_v_state_dict': self.optimizer_v.state_dict(),
            'step': step
        }, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, path, only_weight=False):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        if not only_weight:
            self.optimizer_a.load_state_dict(checkpoint['optimizer_a_state_dict'])
            self.optimizer_v.load_state_dict(checkpoint['optimizer_v_state_dict'])
        step = checkpoint.get('step', 0)
        print(f"Model loaded from {path}, step={step}")
        return step    

    
    def generate_synthetic_examples(self,
                                    prob=[.8, .1, .1],
                                    samples_n=10000,
                                    R_limit=12,
                                    save_path=None,
                                    save_type="traj") -> list:
        '''
        Generate artificially synthesized Tensor examples
        Return: results
        '''
        assert save_type in ["traj", "tuple"]
        
        S_size = self.S_size
        coefficients = self.coefficients
        T = self.T
        
        total_results = []
        for _ in tqdm(range(samples_n)):
            R = random.randint(1, R_limit)
            for _ in range(10000):
                sample = np.zeros((S_size, S_size, S_size), dtype=np.int32)
                states = []        
                actions = []
                rewards = []                
                for r in range(1, (R+1)):
                    ct = 0
                    while True:
                        u = np.random.choice(coefficients, size=(S_size,), p=prob, replace=True)
                        v = np.random.choice(coefficients, size=(S_size,), p=prob, replace=True)
                        w = np.random.choice(coefficients, size=(S_size,), p=prob, replace=True)
                        ct += 1
                        if not is_zero_tensor(outer(u, v, w)):
                            break
                        if ct > 100000:
                            raise Exception("Oh my god...")
                    sample = sample + outer(u, v, w)
                    action = np.stack([u, v, w], axis=0)
                    actions.append(canonicalize_action(action))
                    states.append(sample.copy())
                    rewards.append(-r)
                
                # Check redundancy.
                red_flag = False
                for (i, j) in [[0,1], [1,2], [2,0]]:
                    _mat = np.zeros((S_size ** 2, R), dtype=np.int32)
                    for idx, action in enumerate(actions):
                        _mat[:, idx] = np.outer(action[i], action[j]).reshape((-1,))
                    if np.linalg.matrix_rank(_mat) < R:
                        red_flag = True
                        break
                
                if red_flag:
                    continue
                break
                
            # Reformulate the results.
            if save_type == "tuple":
                states.reverse(); actions.reverse(); rewards.reverse()
                actions_tensor = [action2tensor(action) for action in actions]
                for idx, state in enumerate(states):
                    tensors = np.zeros((T, S_size, S_size, S_size), dtype=np.int32)
                    tensors[0] = state            # state.
                    if idx != 0:
                        # History actions.
                        tensors[1:(idx+1)] = np.stack(reversed(actions_tensor[max(idx-(T-1), 0):idx]), axis=0)        
                    scalars = np.array([idx, idx, idx])     #FIXME: Havn't decided the scalars.
                    
                    cur_state = [tensors, scalars]
                    action = actions[idx]
                    reward = rewards[idx]
                    total_results.append([cur_state, action, reward])
            
            else:
                traj = [states, actions, rewards]          # Note: Synthesis order...
                total_results.append(traj)
                
        if save_path is not None:
            np.save(save_path, np.array(total_results, dtype=object))
            
        return total_results
        
    
    def learn_one_batch(self, batch_example):
        s, a_gt, v_gt = batch_example  # s: [tensor, scalar]

        # Move the two elements of s to the GPU as well
        s_tensor, s_scalar = s
        s_tensor = s_tensor.to(self.device)
        s_scalar = s_scalar.to(self.device)

        a_gt = a_gt.long().to(self.device)
        v_gt = v_gt.float().to(self.device)

        self.net.set_mode("train")
        output = self.net([s_tensor, s_scalar, a_gt])
        o, q = output
        
        v_loss = self.quantile_loss(q, v_gt)
        o = o.transpose(1,2)
        a_loss = self.entropy_loss(o, a_gt)
        loss = self.v_weight * v_loss + self.a_weight * a_loss

        del a_gt, v_gt
        return loss, v_loss, a_loss

    
    def val_one_episode(self,
                        episode):
        '''
        Verify a tuple and print it out
        '''
        
        state, action, reward = episode
        tensor, scalar = state
        
        self.net.set_mode("infer")
        self.net.set_samples_n(4)
        output = self.net(state)
        a, p, _ = output
        a = a.detach().cpu().numpy()
        q, v = self.net.value(output)
        policy, p = self.net.policy(output)
        
        policy, p, q, v = policy[0], p[0], q[0], v[0]
        
        self.net.set_mode("train")
        tensor, scalar, action = tensor[None], scalar[None], action[None]
        state = [tensor, scalar]
        o, _ = self.net([*state, action])
        o = o.detach().cpu().numpy()[0]         # o: [N_steps, N_logits]
        
        log_txt = "\n".join(
            ["\nState: \n", str(state[0][0, 0]),
            "\nGt action: \n", str(self.net.logits_to_action(action[0])),
            "\nGt logit: \n", str(action[0]),
            "\nInfer actions: \n", str(policy),
            "\nInfer logits: \n", str(a),
            "\nprob: \n", str(p),
            "\nGt value: \n", str(reward),
            "\nInfer value: \n", str(v),
            "\nquantile: \n", str(q),
            *["\nTop 5 logit for step %d\n: " % step + str(np.argsort(o[step])[-5:]) for step in range(self.net.N_steps)]]
        )
        
        del a, o, p, _, output, q, v
        
        return log_txt
        
        
    def learn(self,
              resume=None,
              only_weight=False,
              example_path=None,
              self_example_path=None,
              save_type="traj",
              self_play=False):
        '''
        The main function of training
        '''
        import torch
        import time
        from torch.utils.tensorboard import SummaryWriter

        optimizer_a = self.optimizer_a
        scheduler_a = self.scheduler_a
        optimizer_v = self.optimizer_v
        scheduler_v = self.scheduler_v
        batch_size = self.batch_size
        self_play_freq = self.self_play_freq
        self_play_buffer = self.self_play_buffer

        # Tensorboard.
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_writer = SummaryWriter(self.log_dir)

        # Save config.
        all_kwargs = self.all_kwargs
        cfg_path = os.path.join(self.save_dir, "config.yaml")
        with open(cfg_path, 'w') as f:
            yaml.dump(all_kwargs, f)

        if resume is not None:
            # Load model.
            old_iter = self.load_model(resume, only_weight)
            # Copy log file.
            old_exp_dir = os.path.join(os.path.dirname(resume), '..')
            for log_f in os.listdir(os.path.join(old_exp_dir, "log")):
                shutil.copy(os.path.join(old_exp_dir, "log", log_f), self.log_dir)
        else:
            old_iter = 0

        # Save initial checkpoint.
        ckpt_name = "latest.pth"
        self.save_model(ckpt_name, old_iter)

        # 1. Get synthetic examples.
        if example_path is not None:
            self.synthetic_examples.extend(self.load_examples(example_path))
        else:
            self.synthetic_examples.extend(self.generate_synthetic_examples(samples_n=3000))

        if self_example_path is not None:
            self.self_examples.extend(self.load_examples(self_example_path))

        # Dataloader.
        dataset = TupleDataset(T=self.T,
                               S_size=self.S_size,
                               N_steps=self.net.N_steps,
                               coefficients=self.coefficients,
                               self_data=self.self_examples,
                               synthetic_data=self.synthetic_examples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        loader = iter(dataloader)
        epoch_ct = 0

        # Initialization of training statistics variables
        total_loss_accum = 0.0
        total_samples = 0

        # Reset peak memory statistics
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        # Record training start time
        train_start_time = time.time()

        for i in tqdm(range(old_iter, self.iters_n)):

            try:
                batch_example = next(loader)
            except StopIteration:
                dataloader.dataset._permutate_traj()
                if self_play:
                    self_examples = self.get_self_examples()   # New self-play data.
                    if self_examples is not None:
                        print("Detect new self-data!")
                        self.self_examples.extend(self_examples)
                        self.self_examples = self.self_examples[-self_play_buffer:]
                        np.save(os.path.join(self.data_dir, "total_self_data.npy"), np.array(self.self_examples, dtype=object))
                        synthetic_examples_n = 2000 if i > 50000 else 100000
                        dataset = TupleDataset(T=self.T,
                                               S_size=self.S_size,
                                               N_steps=self.net.N_steps,
                                               coefficients=self.coefficients,
                                               self_data=self.self_examples,
                                               synthetic_data=random.sample(self.synthetic_examples, synthetic_examples_n))
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                    else:
                        print("No detect new self-data...")

                loader = iter(dataloader)
                batch_example = next(loader)
                print("Epoch: %d finish." % epoch_ct)
                epoch_ct += 1

            # Optimizer clear
            optimizer_a.zero_grad()
            optimizer_v.zero_grad()

            # Training single step
            loss, v_loss, a_loss = self.learn_one_batch(batch_example)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.grad_clip)
            optimizer_a.step()
            optimizer_v.step()
            scheduler_a.step()
            scheduler_v.step()

            # The accumulated loss is used to calculate the average training loss at the end
            total_loss_accum += loss.item() * len(batch_example) 
            total_samples += len(batch_example)

            self.train_steps = i + 1

            # Recording logs
            if i % 20 == 0:
                self.log_writer.add_scalar("loss", loss.detach().cpu().item(), global_step=i)
                self.log_writer.add_scalar("v_loss", v_loss.detach().cpu().item(), global_step=i)
                self.log_writer.add_scalar("a_loss", a_loss.detach().cpu().item(), global_step=i)

            if i % self.save_freq == 0:
                ckpt_name = "it%07d.pth" % i
                self.save_model(ckpt_name, i)

            if i % self.temp_save_freq == 0:
                ckpt_name = "latest.pth"
                self.save_model(ckpt_name, i)

            if i % self.val_freq == 0:
                val_episode = dataset[random.randint(0, len(dataset)-1)]
                log_txt = self.val_one_episode(val_episode)
                self.log_writer.add_text("Infer", log_txt, global_step=i)

        # After training, save the final model
        self.save_model("final.pth", i)

        # Training end time
        train_end_time = time.time()
        self.train_time_sec = train_end_time - train_start_time

        # Calculate the average training loss
        self.final_train_loss = total_loss_accum / total_samples if total_samples > 0 else float('nan')

        # Peak video memory for training, in MB
        if self.device.startswith("cuda"):
            peak_mem_bytes = torch.cuda.max_memory_allocated()
            self.peak_train_memory_MB = peak_mem_bytes / (1024 ** 2)
        else:
            self.peak_train_memory_MB = "NA"

        print(f"Training finished: total steps={self.train_steps}, "
              f"final loss={self.final_train_loss:.6f}, "
              f"train time={self.train_time_sec:.2f}s, "
              f"peak train memory={self.peak_train_memory_MB} MB")

            
    
    def infer(self,
              init_state=None,
              no_base_change=True,
              mcts_simu_times=1000,
              mcts_samples_n=16,
              step_limit=12,
              resume=None,
              vis=False,
              noise=False,
              log=True):

        import torch
        import time
        from tqdm import tqdm
        import numpy as np
        import os

        log_actions = []

        assert resume is not None, "No meaning for random init infer."
        self.load_model(resume)

        if log:
            exp_dir = os.path.join(os.path.dirname(resume), '..')
            infer_log_dir = os.path.join(exp_dir, "infer")
            os.makedirs(infer_log_dir, exist_ok=True)
            infer_log_f = os.path.join(infer_log_dir, str(int(time.time())) + '.txt')

        net = self.net
        env = self.env
        mcts = self.mcts

        env.reset(init_state, no_base_change)
        net.set_mode("infer")
        net.set_samples_n(mcts_samples_n)
        net.eval()
        mcts.reset(env.cur_state, simulate_times=mcts_simu_times, R_limit=step_limit)
        env.R_limit = step_limit + 1

        step_ct = 0

        # Reset the memory peak value before inference starts
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        infer_start_time = time.time()

        for step in tqdm(range(step_limit)):
            print(f"Current state is (step{step}):")
            print(env.cur_state)

            action, actions, pi, log_txt = mcts(env.cur_state, net, log=True, noise=noise)
            if vis:
                mcts.visualize()
            print(f"We choose action(step{step}):")
            print(action)
            terminate_flag = env.step(action)  # Will change self.cur_state.
            mcts.move(action)  # Move MCTS forward.
            log_actions.append(action)

            if log:
                with open(infer_log_f, "a") as f:
                    f.write(log_txt)
                    f.write("\n\n\n")

            if terminate_flag:
                step_ct = step + 1
                print("We get to the end!")
                break

        infer_end_time = time.time()

        print("Final result:")
        print(env.cur_state)

        print("Actions are:")
        print(np.stack(log_actions, axis=0))

        if log:
            with open(infer_log_f, "a") as f:
                f.write("\n\n\n")
                f.write("\nFinal result:\n")
                f.write("\n" + str(env.cur_state) + "\n")
                f.write("\nActions are:\n")
                f.write("\n" + str(np.stack(log_actions, axis=0)) + "\n")
                f.write("\n\n\n")
                f.write(f"\nStep ct: {step_ct}\n")

        # Record inference time
        self.infer_time_sec = infer_end_time - infer_start_time

        # Record the peak value of video memory, in MB
        if self.device.startswith("cuda"):
            peak_mem_bytes = torch.cuda.max_memory_allocated()
            self.peak_infer_memory_MB = peak_mem_bytes / (1024 ** 2)
        else:
            self.peak_infer_memory_MB = "NA"

        print(f"Inference finished: steps={step_ct}, time={self.infer_time_sec:.2f}s, peak memory={self.peak_infer_memory_MB} MB")

        return step_ct


        
        
    def filter_train_data(self,
                          n=100,
                          example_path=None,
                          mcts_simu_times=10000,
                          mcts_samples_n=16,
                          step_limit=12,
                          resume=None):
        
        assert resume is not None, "No meaning for random init infer."
        assert example_path is not None
        
        synthetic_examples = self.load_examples(example_path)
        test_data = random.sample(list(synthetic_examples), n)
        
        exp_dir = os.path.join(os.path.dirname(resume), '..')
        infer_log_dir = os.path.join(exp_dir, "infer")
        os.makedirs(infer_log_dir, exist_ok=True)
        infer_log_f = os.path.join(infer_log_dir, str(int(time.time()))+'.txt')        
        
        for traj in (test_data):
            # import pdb; pdb.set_trace()
            states, _, _ = traj
            raw_r = len(states)
            init_state = states[-1]
            
            result_r = self.infer(init_state=init_state,
                                  no_base_change=True,
                                  mcts_samples_n=mcts_samples_n,
                                  mcts_simu_times=mcts_simu_times,
                                  step_limit=step_limit,
                                  resume=resume,
                                  log=False)
            
            with open(infer_log_f, "a") as f:
                f.write("%d %d\n" % (raw_r, result_r))
        
        
    
    def save_model(self, ckpt_name, iter):
        save_dir = os.path.join(self.save_dir, "ckpt")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save({'model': self.net.state_dict(),
                    'iter': iter,
                    'optimizer_a': self.optimizer_a.state_dict(),
                    'optimizer_v': self.optimizer_v.state_dict(),
                    'scheduler_a': self.scheduler_a.state_dict(),
                    'scheduler_v': self.scheduler_v.state_dict()}, save_path)


    def load_model(self, ckpt_path, only_weight=False, to_device="cuda:0"):
        ckpt = torch.load(ckpt_path)
        self.net.load_state_dict(ckpt['model'])
        if not only_weight:
            self.optimizer_a.load_state_dict(ckpt['optimizer_a'])
            self.optimizer_v.load_state_dict(ckpt['optimizer_a'])
            self.scheduler_a.load_state_dict(ckpt['scheduler_v'])
            self.scheduler_v.load_state_dict(ckpt['scheduler_v'])
        return ckpt['iter']

    
    def load_examples(self, example_path):
        return np.load(example_path, allow_pickle=True)
    
    
    def get_self_examples(self):
        newest_data_path = os.path.join(self.data_dir, "self_data.npy")   # New data from player.
        old_data_path = os.path.join(self.data_dir, "self_data_old.npy")
        if os.path.exists(newest_data_path):
            ct = 0
            while True:
                if ct > 10000000:
                    raise Exception
                try:
                    self_examples = self.load_examples(newest_data_path)
                    os.system("mv %s %s" % (newest_data_path, old_data_path))
                    return self_examples
                except:
                    ct += 1
                    continue
        return None
            
          
            
if __name__ == '__main__':
    
    conf_path = "./config/my_conf.yaml"
    with open(conf_path, 'r', encoding="utf-8") as f:
        kwargs = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    net = Net(**kwargs["net"])
    mcts = MCTS(**kwargs["mcts"],
                init_state=None)
    env = Environment(**kwargs["env"],
                      init_state=None)
    trainer = Trainer(**kwargs["trainer"],
                      net=net, env=env, mcts=mcts,
                      all_kwargs=kwargs)
    
    
    # import pdb; pdb.set_trace()
    # res = trainer.play()
    # res = trainer.generate_synthetic_examples()
    # trainer.learn(example_path="./data/100000_T5_scalar3.npy")
    # import pdb; pdb.set_trace()
    # trainer.load_model("./exp/debug/1680630182/ckpt/it0020000.pth")
    # trainer.infer()
    # trainer.infer(resume="./exp/debug/1680764978/ckpt/it0002000.pth")
    # import pdb; pdb.set_trace()
    # trainer.generate_synthetic_examples(samples_n=3000, save_path="./data/3000_T5_scalar3.npy")
    # trainer.learn(resume=None,
    #               example_path="./data/100000_T5_scalar3.npy")
    # trainer.learn(resume=None)
    trainer.learn(resume=None,
                  example_path="./data/3000_T5_scalar3.npy")