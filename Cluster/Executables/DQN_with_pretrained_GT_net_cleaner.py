#!/usr/bin/env python
# coding: utf-8

# DEEP REINFORCEMENT LEARNING EXPLAINED - 15 - 16 - 17
# # **Deep Q-Network (DQN)**

# OpenAI Pong

# In[1]:


import gym
import gym.spaces

DEFAULT_ENV_NAME = "PongNoFrameskip-v4" 




import warnings
warnings.filterwarnings('ignore')


# ## OpenAI Gym Wrappers

# In[7]:


# Taken from 
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py

import cv2
import numpy as np
import collections
import numpy as np

import torch
import torch.nn as nn        # Pytorch neural network package
import torch.optim as optim  # Pytorch optimization package
import numpy
import random



features = 4
Prepro_device = "cuda"
class GT_Net(nn.Module):
    def __init__(self, num_channels):
        super(GT_Net, self).__init__()
        
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=4, stride=2, padding=1)  # 42 x 42
        self.conv2 = nn.Conv2d(32, 32, 2, 2, 1)  # 21 x 21
        self.conv3 = nn.Conv2d(32, 64, 2, 2, 1)  # 11 x 11
        self.conv4 = nn.Conv2d(64, 64, 2, 2, 1)  # 6 x 6
        self.flat1 = nn.Flatten()
        self.dense1 = nn.Linear(3136, 256) # 6x6x 64 = 2304
        self.dense2 = nn.Linear(256, 4*num_channels)
        
        self.BN0 = nn.BatchNorm1d(256)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        h = self.act(self.conv1(x))
        #print("conv1: " + str(h.size()))
        h = self.act(self.conv2(h))
        #print("conv2: " + str(h.size()))
        h = self.act(self.conv3(h))
        #print("conv3: " + str(h.size()))
        h = self.act(self.conv4(h))
        #print("conv4: " + str(h.size()))
        h = self.flat1(h)
        #print(h.size())
        h = self.act(self.BN0(self.dense1(h)))
        #print(h.size())
        return self.dense2(h)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], 
                                old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class Scale01(gym.ObservationWrapper):
    def observation(self, obs):
        obs[obs < 0.342] = 0
        obs[:, 83, :] = 0 #unterste pixelreihe war immer 1 also hat keine Infos. Daher wird auf 0 gesetzt.
        obs[obs > 0.1] = 1
        return obs
class PreproNet(gym.ObservationWrapper):
    def __init__(self, env):
        super(PreproNet, self).__init__(env)
        self.model = GT_Net(4).to(Prepro_device)
        self.model.load_state_dict(torch.load('/itet-stor/ericschr/net_scratch/BA/GT_from_Images_Buffer_400E_3-5'))
        self.model.eval()
        self.index = 0
        print("prepro model")
        print(self.model)
    
    def observation(self, obs):
        obs = obs[None,:,:,:]
        res = self.model(torch.from_numpy(obs).to(Prepro_device)).detach().cpu().numpy()
        print(res)
        if self.index > 20:
            np.save("/itet-stor/ericschr/net_scratch/BA/savedImagesDQN_GTAnfang", obs)
            print("saved images")
            exit()
        self.index += 1
        return res
        # ret = np.zeros((4,4))
        #print(f"obs: {obs.shape}")
        # obs = obs[:,None,None,:,:]
        #print(f"obs: {obs.shape}")
        #print(f"obs: {obs[0].dtype}")
        # with torch.no_grad():
        #     ret[0] = self.model(torch.from_numpy(obs[0]).to(Prepro_device)).detach().cpu().numpy()
        #     print(f"reto 0: {ret[0]}")
        #     ret[1] = self.model(torch.from_numpy(obs[0]).to(Prepro_device)).detach().cpu().numpy()
        #     ret[2] = self.model(torch.from_numpy(obs[0]).to(Prepro_device)).detach().cpu().numpy()
        #     ret[3] = self.model(torch.from_numpy(obs[0]).to(Prepro_device)).detach().cpu().numpy()
        # return ret
    
def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    env = Scale01(env)
    return PreproNet(env)


# ## The DQN model
# 

# In[11]:



device = torch.device("cuda")
#device = torch.device("cpu")


random_seed = 0 
numpy.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(random_seed)
torch.random.manual_seed(random_seed)

# In[12]:


# Taken from 
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/dqn_model.py



class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # self.conv = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )
        # self.gt_net = GT_Net(1)

        # conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(16, 3136),
            nn.ReLU(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    # def _get_conv_out(self, shape):
    #     o = (torch.zeros(1, *shape)).view(1, -1)
    #     return int(np.prod(o.size()))

    def forward(self, x):
        # print(f"convinsize: {x.size()}")
        #conv_out = self.conv(x).view(x.size()[0], -1)
        conv_out = x.view(x.size()[0], -1).to(torch.float32)
        # print(f"convoutsize: {conv_out.size()}")
        # print(f"convoutsize: {conv_out.dtype}")
        return self.fc(conv_out)


# In[13]:


test_env = make_env(DEFAULT_ENV_NAME)
test_net = DQN(test_env.observation_space.shape, test_env.action_space.n).to(device)
print(test_net)


# ## Training

# Load Tensorboard extension

# In[15]:


from torch.utils.tensorboard import SummaryWriter



# Import required modules and define the hyperparameters

# In[16]:


import time
import numpy as np
import collections
import datetime
x = datetime.datetime.now()


VISUALIZEtraining = False
MEAN_REWARD_BOUND = 1   #Change to 19.0  !!not used!!
max_frame_idx   = 1000000 #1Mio  

gamma = 0.99                   
batch_size = 32                
replay_size = 10000            
learning_rate = 1e-4           
sync_target_frames = 1000      
replay_start_size = 10000      

eps_start=1.0
eps_decay=.999985
eps_min=0.02


# Experience replay buffer

# In[17]:


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),                np.array(dones, dtype=np.uint8), np.array(next_states)


# Agent

# In[18]:


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        
        if VISUALIZEtraining:
            env.render()
            
        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


# In[19]:


import datetime
import math
print(">>>Training starts at ",datetime.datetime.now())


# Main training loop

# In[20]:


env = make_env(DEFAULT_ENV_NAME)

net = DQN(env.observation_space.shape, env.action_space.n).to(device)
target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)

writer = SummaryWriter(log_dir="/itet-stor/ericschr/net_scratch/BA/DQN_runs/summary", comment=f"longRun-{x.day}_{x.month}" + DEFAULT_ENV_NAME)
 
buffer = ExperienceReplay(replay_size)
agent = Agent(env, buffer)

epsilon = eps_start

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
total_rewards = []
#*******Change************* That way every imprvement counts
for i in range(100):
    total_rewards.append(-21.000)

frame_idx = 0  

best_mean_reward = None
save = -19


# In[21]:


#model=r'C:\Users\erics\Documents\Programme\Bachelorarbeit\PongNoFrameskip-v4-best-Kopie-11-300.dat'
#net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))


# In[22]:



while True:
        frame_idx += 1
        epsilon = max(epsilon*eps_decay, eps_min)
        
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)

            mean_reward = np.mean(total_rewards[-10:]) #changed from 100 to have a quicker downwards trend as well           
            
            print("%d:  %d games, mean reward %.3f, (epsilon %.3f)" % (
                frame_idx, len(total_rewards), mean_reward, epsilon))
            
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("mean_reward", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                if mean_reward > save: # save @-19, -17, -15, -13, etc
                    torch.save(net.state_dict(), "/itet-stor/ericschr/net_scratch/BA/DQN_runs/" + DEFAULT_ENV_NAME + f"-best{x.day}_{x.month}-{save}.dat")
                    save += 2
                    
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f" % (best_mean_reward))

            # if mean_reward > MEAN_REWARD_BOUND:
            #     print("Solved in %d frames!" % frame_idx)
            #     break

            if frame_idx > max_frame_idx:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < replay_start_size:
            continue

        batch = buffer.sample(batch_size)
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

        state_action_values = net(states_v).gather(1, actions_v.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        #For Linux use: state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        next_state_values = target_net(next_states_v).max(1)[0]

        next_state_values[done_mask] = 0.0

        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * gamma + rewards_v

        loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)
                
        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()
        if frame_idx % sync_target_frames == 0:
            target_net.load_state_dict(net.state_dict())
       
writer.close()


# In[23]:


print(">>>Training ends at ",datetime.datetime.now())


