#!/usr/bin/env python
# coding: utf-8

# DEEP REINFORCEMENT LEARNING EXPLAINED - 15 - 16 - 17
# # **Deep Q-Network (DQN)**

# OpenAI Pong

# In[1]:
import sys
#arguments: path to model, features, likeDQN, namefolder, random seed
assert(len(sys.argv)==5)

namefolder = str(sys.argv[1])
random_seed = int(sys.argv[2])
eps_decay = float(sys.argv[3])
bigAgent = str(sys.argv[4])

namefolder = namefolder + "_" + bigAgent + "_" + str(eps_decay) + "_" + str(random_seed)

print(f"namefolder: {namefolder}, random_seed: {random_seed}, eps_decay: {eps_decay}, bigAgent: {bigAgent}")


import gym
import gym.spaces

DEFAULT_ENV_NAME = "PongNoFrameskip-v4" 



import warnings
warnings.filterwarnings('ignore')


# ## OpenAI Gym Wrappers



# Taken from 
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py

import cv2
import numpy as np
import collections

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

def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return Scale01(env)


# ## The DQN model
# 



import torch
import torch.nn as nn        # Pytorch neural network package
import torch.optim as optim  # Pytorch optimization package
import torch.nn.functional as F
from torchtyping import TensorType
import numpy
import random


device = torch.device("cuda")
BVAEtoDevice = False #should prepro happen on GPU or CPU False becuse not with net
linear = False

# random_seed = 43 
numpy.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(random_seed)
torch.random.manual_seed(random_seed)



# Taken from 
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/dqn_model.py

import numpy as np
features = 4

class DQNnormal(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNnormal, self).__init__()

  
        self.fc = nn.Sequential(
            nn.Linear(features*4, 3136), #buffer von 4 Bilder nacheinander als input (features*2 kommt vom training vom VAE)
            nn.Linear(3136, 512),
            # #***Buffered BVAE ***
            # nn.Linear(features*2, 512), #kein buffer von 4 Bilder nacheinander als input (features*2 kommt vom training vom VAE)
            # #***Buffered BVAE ende ***

            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    
    def forward(self, x):
        
        x = torch.div(x, 84) #normalise the data

        return self.fc(x)

class DQNbig(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNbig, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(features*4, 3136), 
            nn.InstanceNorm1d(3136),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(3136, 3136), 
            nn.InstanceNorm1d(3136),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        

    def forward(self, x):
        x = torch.div(x, 84) #normalise the data

        if x.shape[0] == 32:
            return self.fc(x)
        else:
            x = x[None,:]
            return self.fc(x)[0]

class DQNsuperBig(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNsuperBig, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(features*4, 3136), 
            nn.InstanceNorm1d(3136),
            nn.ReLU(),
            # nn.Dropout(p=0.1),

            nn.Linear(3136, 3136), 
            nn.InstanceNorm1d(3136),
            nn.ReLU(),
            # nn.Dropout(p=0.1),

            nn.Linear(3136, 3136), 
            nn.InstanceNorm1d(3136),
            nn.ReLU(),
            # nn.Dropout(p=0.1),

            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
       
    def forward(self, x):
        x = torch.div(x, 84) #normalise the data
        
        
        if x.shape[0] == 32:
            return self.fc(x)
        else:
            x = x[None,:]
            return self.fc(x)[0]
       


# In[9]:


test_env = make_env(DEFAULT_ENV_NAME)
if bigAgent == 'normal':
    test_net = DQNnormal(test_env.observation_space.shape, test_env.action_space.n).to(device)
elif bigAgent == 'big':
    test_net = DQNbig(test_env.observation_space.shape, test_env.action_space.n).to(device)
elif bigAgent == 'superbig':
    test_net = DQNsuperBig(test_env.observation_space.shape, test_env.action_space.n).to(device)
print(test_net)


# Load pretrained BVAE



features = 4
# define a simple net to train on the ground truths
class Net(nn.Module):
    def __init__(self, num_channels):
        super(Net, self).__init__()
        
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
        h = self.dense2(h)
        print(f"states: {h}")
        return h

#calculate the ground truths from the images
def getGT(img):
    if len(np.where(img[:,9]==1)[0]) == 0: #no left paddle
           leftpaddle = 0
    else:
           leftpaddle = np.average(np.where(img[:,9]==1)[0])#[0]
                         
    if len(np.where(img[:,74]==1)[0]) == 0: #no right paddle
           rightpaddle = 0
    else:
           rightpaddle = np.average(np.where(img[:,74]==1)[0])#[0]
                         
                         
    
    img[:,8:11]=0
    img[:,73:76]=0
    
    
    
    if len(np.where(img==1)[1]) == 0: #no x ball
           xBall = 0
    else:
           xBall = np.average(np.where(img==1)[1])#[0]
            
    if len(np.where(img==1)[0]) == 0: #no y ball
           yBall = 0
    else:
           yBall = np.average(np.where(img==1)[0])#[0]
    
   
    
    ret = np.zeros(4)
    ret[0] = leftpaddle
    ret[1] = rightpaddle
    ret[2] = xBall
    ret[3] = yBall
    
    #return leftpaddle, rightpaddle, xBall, yBall
    return ret

def getGTBuffer(imgbuf):
    ret = np.zeros(16)
    ret[0:4] = getGT(imgbuf[0])
    ret[4:8] = getGT(imgbuf[1])
    ret[8:12] = getGT(imgbuf[2])
    ret[12:] = getGT(imgbuf[3])
    return ret

def getGTStack(imgstack):
    ret = np.zeros((imgstack.shape[0],16))
    for i in range(len(imgstack)):
        ret[i] = getGTBuffer(imgstack[i])
    return ret






from torch.utils.tensorboard import SummaryWriter


# Import required modules and define the hyperparameters



import time
import numpy as np
import collections
import datetime
x = datetime.datetime.now()

VISUALIZEtraining = False
MEAN_REWARD_BOUND = -19   #Change to 19.0 !!not used!!
max_frame_idx   = 1000000 #1Mio

gamma = 0.99                   
batch_size = 32                
replay_size = 10000            
learning_rate = 1e-4           
sync_target_frames = 1000      
replay_start_size = 10000      

eps_start=1.0
eps_min=0.02

print(f"randSeed {random_seed} gamma {gamma} batch_size {batch_size} replay_size {replay_size} learning_rate {learning_rate} sync_target_frames {sync_target_frames} replay_start_size {replay_start_size} eps_start {eps_start} eps_decay {eps_decay} eps_min{eps_min}")
# Experience replay buffer



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
            state_a = getGTBuffer(state_a[0]) 
            state_v = torch.tensor(state_a, dtype=torch.float)
            
            if BVAEtoDevice == True:
                state_v = state_v.to(device)
            
            if BVAEtoDevice == False:
                state_v = state_v.to(device)
            q_vals_v = net(state_v)
           
            _, act_v = torch.max(q_vals_v, dim=0)
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



import datetime
import math
print(">>>Training starts at ",datetime.datetime.now())


# Main training loop




env = make_env(DEFAULT_ENV_NAME)

if bigAgent == 'normal':
    net = DQNnormal(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQNnormal(env.observation_space.shape, env.action_space.n).to(device)
elif bigAgent == 'big':
    net = DQNbig(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQNbig(env.observation_space.shape, env.action_space.n).to(device)
elif bigAgent == 'superbig':
    net = DQNsuperBig(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQNsuperBig(env.observation_space.shape, env.action_space.n).to(device)

writer = SummaryWriter(log_dir=f"/itet-stor/ericschr/net_scratch/BA/DQN_runs/With_pretrained_GT_net/summary/{namefolder}", comment=f"-GT_NET_Buff{x.day}_{x.month}_EPSdec{eps_decay}" + DEFAULT_ENV_NAME) 

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

while True:
        frame_idx += 1
        epsilon = max(epsilon*eps_decay, eps_min)
        
        # reward = agent.play_step(BVAE, net, epsilon, device=device)
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
               
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    torch.save(net.state_dict(), f"/itet-stor/ericschr/net_scratch/BA/DQN_runs/With_pretrained_GT_net/summary/{namefolder}/" + DEFAULT_ENV_NAME + f"GT_NET-{x.day}_{x.month}.dat")
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
        
        #BVAE
        states_TOpreprocess = torch.tensor(states)
        next_states_TOpreprocess = torch.tensor(next_states)
        # np.save("/itet-stor/ericschr/net_scratch/BA/savedImagesDQN_GT", states_TOpreprocess)
        # print("saved images")
        # exit()
        

        if BVAEtoDevice == True:
            states_TOpreprocess = states_TOpreprocess.to(device)
            next_states_TOpreprocess = next_states_TOpreprocess.to(device)
        else:
            states_TOpreprocess = states_TOpreprocess.to('cpu')
            next_states_TOpreprocess = next_states_TOpreprocess.to('cpu')

        states_preprocessed = torch.tensor(getGTStack(states_TOpreprocess), dtype=torch.float)
        next_states_preprocessed = torch.tensor(getGTStack(next_states_TOpreprocess), dtype=torch.float)

        
            
            
            
        if BVAEtoDevice == False:
            states_preprocessed = states_preprocessed.to(device)
            next_states_preprocessed = next_states_preprocessed.to(device)


        
        states_v = states_preprocessed.view(-1, 4*features) #oder batchsize [batchsize, 4* features]
        
        next_states_v = next_states_preprocessed.view(-1, 4*features)
            
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




print(">>>Training ends at ",datetime.datetime.now())

