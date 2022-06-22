#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
#arguments: path to model, features, likeDQN, namefolder, random seed, epsilon decay, big Agent [normal, big, superbig]
assert(len(sys.argv)==8)

modeldir = str(sys.argv[1])
features = int(sys.argv[2])
likeDQN =  str(sys.argv[3])          #likeDQN oder NotDQN oder NotDQNWOBN oder likeDQNWOBN
namefolder = str(sys.argv[4])
random_seed = int(sys.argv[5])
eps_decay = float(sys.argv[6])
bigAgent = str(sys.argv[7])

assert(bigAgent=='normal' or bigAgent=='big' or bigAgent=='superbig')

print(f"modeldir: {modeldir}, features: {features}, likeDQN: {likeDQN}, namefolder: {namefolder}, random_seed: {random_seed}, eps_decay: {eps_decay}, bigAgent: {bigAgent}")

namefolder = namefolder + "_" + bigAgent + "_" + eps_decay + "_" + random_seed


import gym
import gym.spaces

DEFAULT_ENV_NAME = "PongNoFrameskip-v4" 



import warnings
warnings.filterwarnings('ignore')


# ## OpenAI Gym Wrappers

# In[6]:


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

# In[7]:


import torch
import torch.nn as nn        # Pytorch neural network package
import torch.optim as optim  # Pytorch optimization package
import torch.nn.functional as F
from torchtyping import TensorType
import numpy
import random


device = torch.device("cuda")
BVAEtoDevice = True #should prepro happen on GPU or CPU
linear = False

#random_seed = 0
numpy.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(random_seed)
torch.random.manual_seed(random_seed)

# In[8]:


# Taken from 
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/dqn_model.py

import numpy as np
# features = 64
# likeDQN = "NotDQNWOBN" #likeDQN oder NotDQN oder NotDQNWOBN oder likeDQNWOBN
isSum = "sum" # "sum" or "mean"
assert(isSum=="mean" or isSum=="sum")
# namefolder = f"testOldLong_Seed{random_seed}"

class DQNnormal(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNnormal, self).__init__()

   #     self.conv = nn.Sequential(
   #         nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
   #         nn.ReLU(),
   #         nn.Conv2d(32, 64, kernel_size=4, stride=2),
   #         nn.ReLU(),
   #         nn.Conv2d(64, 64, kernel_size=3, stride=1),
   #         nn.ReLU()
   #     )
        
        #conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            #nn.Linear(conv_out_size, 512),
            nn.Linear(features*2*4, 3136), #buffer von 4 Bilder nacheinander als input (features*2 kommt vom training vom VAE)
            # nn.BatchNorm1d(3136),
            nn.Linear(3136, 512),
            # #***Buffered BVAE ***
            # nn.Linear(features*2, 512), #kein buffer von 4 Bilder nacheinander als input (features*2 kommt vom training vom VAE)
            # #***Buffered BVAE ende ***

            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    #def _get_conv_out(self, shape):
    #    o = self.conv(torch.zeros(1, *shape))
    #    return int(np.prod(o.size()))

    def forward(self, x):
        #conv_out = self.conv(x).view(x.size()[0], -1)
        #return self.fc(conv_out)
        #print(x.size())

        return self.fc(x)

class DQNbig(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNbig, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(features*2*4, 3136), 
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

        if x.shape[0] == 32:
            return self.fc(x)
        else:
            x = x[None,:]
            return self.fc(x)[0]

class DQNsuperBig(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNsuperBig, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(features*2*4, 3136), 
            nn.InstanceNorm1d(3136),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(3136, 3136), 
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

# In[10]:



# define a simple linear VAE #until now normal VAE without Beta
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
 
        # encoder 84*84 = 7’056
        self.enc0 = nn.Linear(in_features=84*84, out_features=1024)
        self.enc1 = nn.Linear(in_features=1024, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=features*2)
 
        # decoder 
        self.dec0 = nn.Linear(in_features=features, out_features=512)
        self.dec1 = nn.Linear(in_features=512, out_features=1024)
        self.dec2 = nn.Linear(in_features=1024, out_features=84*84)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
 #   def forward(self, x):
 #       # encoding
 #       x = F.relu(self.enc0(x))
 #       x = F.relu(self.enc1(x))

 #       x = self.enc2(x).view(-1, 2, features)

        # get `mu` and `log_var`
 #       mu = x[:, 0, :] # the first feature values as mean
 #       log_var = x[:, 1, :] # the other feature values as variance

        # get the latent vector through reparameterization
 #       z = self.reparameterize(mu, log_var)
 
        # decoding
 #       x = F.relu(self.dec0(z))
 #       x = F.relu(self.dec1(x))
 #       reconstruction = torch.sigmoid(self.dec2(x))
 #       return reconstruction, mu, log_var
    
    def encode(self, x):
        x = F.relu(self.enc0(x))
        x = F.relu(self.enc1(x))
        x = self.enc2(x)
        return x


# Convolutional VAEs

# In[11]:


class Encoder(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=4, stride=2, padding=1)  # 42 x 42
        self.BN2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 2, 2, 1)  # 21 x 21
        self.BN3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 2, 2, 1)  # 11 x 11
        self.BN4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 2, 2, 1)  # 6 x 6
        self.BN5 = nn.BatchNorm2d(64)
        self.flat1 = nn.Flatten()
        self.dense1 = nn.Linear(3136, 256) # 6x6x 64 = 2304
        self.BN1 = nn.BatchNorm1d(256)
        self.dense_means_logVar = nn.Linear(256, latent_dim*2)
        #self.dense_log_var = nn.Linear(256, latent_dim)

        self.act = nn.ReLU(inplace=True)
    
    
    
    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "output_dim"], TensorType["batch", "output_dim"]):
        #print("encoder: ")
        #print(x.size())
        h = self.act(self.BN2(self.conv1(x)))
        #print("conv1: " + str(h.size()))
        h = self.act(self.BN3(self.conv2(h)))
        #print("conv2: " + str(h.size()))
        h = self.act(self.BN4(self.conv3(h)))
        #print("conv3: " + str(h.size()))
        h = self.act(self.BN5(self.conv4(h)))
        #print("conv4: " + str(h.size()))
        
        h = self.flat1(h)
        #print(h.size())
        h = self.act(self.BN1(self.dense1(h)))
        #print(h.size())
        #means = self.dense_means(h)
        #print(means.size())
        #log_var = self.dense_log_var(h)
        #print(log_var.size())
        return self.dense_means_logVar(h)
        
        #sample = self.reparameterize(means, log_var)
        
        #return sample, means, log_var
        #return means, log_var

class EncoderWOBN(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, latent_dim: int):
        super(EncoderWOBN, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=4, stride=2, padding=1)  # 42 x 42
        self.conv2 = nn.Conv2d(32, 32, 2, 2, 1)  # 21 x 21
        self.conv3 = nn.Conv2d(32, 64, 2, 2, 1)  # 11 x 11
        self.conv4 = nn.Conv2d(64, 64, 2, 2, 1)  # 6 x 6
        self.flat1 = nn.Flatten()
        self.dense1 = nn.Linear(3136, 256) # 6x6x 64 = 2304
        self.dense_means_logVar = nn.Linear(256, latent_dim*2)
        #self.dense_log_var = nn.Linear(256, latent_dim)

        self.act = nn.ReLU(inplace=True)
    
    
    
    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "output_dim"], TensorType["batch", "output_dim"]):
        #print("encoder: ")
        #print(x.size())
        h = self.act((self.conv1(x)))
        #print("conv1: " + str(h.size()))
        h = self.act((self.conv2(h)))
        #print("conv2: " + str(h.size()))
        h = self.act((self.conv3(h)))
        #print("conv3: " + str(h.size()))
        h = self.act((self.conv4(h)))
        #print("conv4: " + str(h.size()))
        
        h = self.flat1(h)
        #print(h.size())
        h = self.act((self.dense1(h)))
        #print(h.size())
        #means = self.dense_means(h)
        #print(means.size())
        #log_var = self.dense_log_var(h)
        #print(log_var.size())
        return self.dense_means_logVar(h)
        
        #sample = self.reparameterize(means, log_var)
        
        #return sample, means, log_var
        #return means, log_var

# In[5]:


class EncoderLikeDQN(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, latent_dim: int):
        super(EncoderLikeDQN, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0)  # 20 x 20
        # self.BN2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 10 x 10
        # self.BN3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)  # 10 x 10
        # self.BN4 = nn.BatchNorm2d(64)
        self.flat1 = nn.Flatten()        
        self.dense1 = nn.Linear(6400, 512) # 10x10x 64 = 6400
        self.BN1 = nn.BatchNorm1d(512)
        self.dense_means_logVar = nn.Linear(512, latent_dim*2)
        #self.dense_log_var = nn.Linear(256, latent_dim)

        self.act = nn.ReLU(inplace=True)
    
    
    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "output_dim"], TensorType["batch", "output_dim"]):
        #print("encoder: ")
        #print(x.size())
        # h = self.act(self.BN2(self.conv1(x)))
        h = self.act(self.conv1(x))

        #print(h.size())
        # h = self.act(self.BN3(self.conv2(h)))
        h = self.act((self.conv2(h)))

        #print(h.size())
        # h = self.act(self.BN4(self.conv3(h)))
        h = self.act((self.conv3(h)))

        #print(h.size())
        
        h = self.flat1(h)
        #print(h.size())
        h = self.act(self.BN1(self.dense1(h)))
        #print(h.size())
        #means = self.dense_means(h)
        #print(means.size())
        #log_var = self.dense_log_var(h)
        #print(log_var.size())
        return self.dense_means_logVar(h)
        
        #sample = self.reparameterize(means, log_var)
        
        #return sample, means, log_var
        #return means, log_var


# In[6]:


class Decoder(nn.Module):
    def __init__(self, input_dim: int, num_channels: int, latent_dim: int):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels

        self.dense1 = nn.Linear(latent_dim, 256)
        self.BN1 = nn.BatchNorm1d(256)
        self.dense2 = nn.Linear(256, 3136)
        self.BN2 = nn.BatchNorm1d(3136)


        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=1)
        self.BN3 = nn.BatchNorm2d(64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=1)
        self.BN4 = nn.BatchNorm2d(32)
        self.upconv3 = nn.ConvTranspose2d(32, 32, 2, stride=2, padding=1)
        self.BN5 = nn.BatchNorm2d(32)
        self.upconv4 = nn.ConvTranspose2d(32, num_channels, 4, stride=2, padding=1)

        self.act = nn.ReLU(inplace=True)
        

    def forward(self, z: TensorType["batch", "input_dim"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        #print("decoder: ")
        h = self.act(self.BN1(self.dense1(z)))
        h = self.act(self.BN2(self.dense2(h)))
        h = h.view(-1, 64, 7, 7)
        #print(h.size())
        h = self.act(self.BN3(self.upconv1(h)))
        #print("Transpose 1: " + str(h.size()))
        h = self.act(self.BN4(self.upconv2(h)))
        #print("Transpose 2: " + str(h.size()))
        h = self.act(self.BN5(self.upconv3(h)))
        #print("Transpose 3: " + str(h.size()))
        img = self.upconv4(h)
        #print("Transpose 4: " + str(img.size()))
        return img


class DecoderWOBN(nn.Module):
    def __init__(self, input_dim: int, num_channels: int, latent_dim: int):
        super(DecoderWOBN, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels

        self.dense1 = nn.Linear(latent_dim, 256)
        self.dense2 = nn.Linear(256, 3136)


        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(32, 32, 2, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(32, num_channels, 4, stride=2, padding=1)

        self.act = nn.ReLU(inplace=True)
        

    def forward(self, z: TensorType["batch", "input_dim"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        #print("decoder: ")
        h = self.act((self.dense1(z)))
        h = self.act((self.dense2(h)))
        h = h.view(-1, 64, 7, 7)
        #print(h.size())
        h = self.act((self.upconv1(h)))
        #print("Transpose 1: " + str(h.size()))
        h = self.act((self.upconv2(h)))
        #print("Transpose 2: " + str(h.size()))
        h = self.act((self.upconv3(h)))
        #print("Transpose 3: " + str(h.size()))
        img = self.upconv4(h)
        #print("Transpose 4: " + str(img.size()))
        return img

# In[8]:


class DecoderLikeDQN(nn.Module):
    def __init__(self, input_dim: int, num_channels: int, latent_dim: int):
        super(DecoderLikeDQN, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels

        self.dense1 = nn.Linear(latent_dim, 512)
        self.BN1 = nn.BatchNorm1d(512)
        self.dense2 = nn.Linear(512, 6400)
        self.BN2 = nn.BatchNorm1d(6400)        
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        # self.BN3 = nn.BatchNorm2d(32)
        self.upconv2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        # self.BN4 = nn.BatchNorm2d(32)
        self.upconv3 = nn.ConvTranspose2d(32, num_channels, 8, 4, 0)

        self.act = nn.ReLU(inplace=True)
        

    def forward(self, z: TensorType["batch", "input_dim"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        #print("encoder: ")
        h = self.act(self.BN1(self.dense1(z)))
        h = self.act(self.BN2(self.dense2(h)))
        h = h.view(-1, 64, 10, 10)
        #print(h.size())
        # h = self.act(self.BN3(self.upconv1(h)))
        h = self.act((self.upconv1(h)))

        #print(h.size())
        # h = self.act(self.BN4(self.upconv2(h)))
        h = self.act((self.upconv2(h)))

        #print(h.size())
        img = self.upconv3(h)
        #print(img.size())
        return img


class EncoderLikeDQNWOBN(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, latent_dim: int):
        super(EncoderLikeDQNWOBN, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0)  # 20 x 20
        # self.BN2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 10 x 10
        # self.BN3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)  # 10 x 10
        # self.BN4 = nn.BatchNorm2d(64)
        self.flat1 = nn.Flatten()        
        self.dense1 = nn.Linear(6400, 512) # 10x10x 64 = 6400
        self.dense_means_logVar = nn.Linear(512, latent_dim*2)
        #self.dense_log_var = nn.Linear(256, latent_dim)

        self.act = nn.ReLU(inplace=True)
    
    
    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "output_dim"], TensorType["batch", "output_dim"]):
        #print("encoder: ")
        #print(x.size())
        # h = self.act(self.BN2(self.conv1(x)))
        h = self.act(self.conv1(x))

        #print(h.size())
        # h = self.act(self.BN3(self.conv2(h)))
        h = self.act((self.conv2(h)))

        #print(h.size())
        # h = self.act(self.BN4(self.conv3(h)))
        h = self.act((self.conv3(h)))

        #print(h.size())
        
        h = self.flat1(h)
        #print(h.size())
        h = self.act((self.dense1(h)))
        #print(h.size())
        #means = self.dense_means(h)
        #print(means.size())
        #log_var = self.dense_log_var(h)
        #print(log_var.size())
        return self.dense_means_logVar(h)
        
        #sample = self.reparameterize(means, log_var)
        
        #return sample, means, log_var
        #return means, log_var

class DecoderLikeDQNWOBN(nn.Module):
    def __init__(self, input_dim: int, num_channels: int, latent_dim: int):
        super(DecoderLikeDQNWOBN, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels

        self.dense1 = nn.Linear(latent_dim, 512)
        self.dense2 = nn.Linear(512, 6400)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        # self.BN3 = nn.BatchNorm2d(32)
        self.upconv2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        # self.BN4 = nn.BatchNorm2d(32)
        self.upconv3 = nn.ConvTranspose2d(32, num_channels, 8, 4, 0)

        self.act = nn.ReLU(inplace=True)
        

    def forward(self, z: TensorType["batch", "input_dim"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        #print("encoder: ")
        h = self.act((self.dense1(z)))
        h = self.act((self.dense2(h)))
        h = h.view(-1, 64, 10, 10)
        #print(h.size())
        # h = self.act(self.BN3(self.upconv1(h)))
        h = self.act((self.upconv1(h)))

        #print(h.size())
        # h = self.act(self.BN4(self.upconv2(h)))
        h = self.act((self.upconv2(h)))

        #print(h.size())
        img = self.upconv3(h)
        #print(img.size())
        return img



class EncoderLikeDQNWBN(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, latent_dim: int):
        super(EncoderLikeDQNWBN, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0)  # 20 x 20
        self.BN2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 10 x 10
        self.BN3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)  # 10 x 10
        self.BN4 = nn.BatchNorm2d(64)
        self.flat1 = nn.Flatten()        
        self.dense1 = nn.Linear(6400, 512) # 10x10x 64 = 6400
        self.BN1 = nn.BatchNorm1d(512)
        self.dense_means_logVar = nn.Linear(512, latent_dim*2)
        #self.dense_log_var = nn.Linear(256, latent_dim)

        self.act = nn.ReLU(inplace=True)
    
    
    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "output_dim"], TensorType["batch", "output_dim"]):
        #print("encoder: ")
        #print(x.size())
        h = self.act(self.BN2(self.conv1(x)))
        #print(h.size())
        h = self.act(self.BN3(self.conv2(h)))
        #print(h.size())
        h = self.act(self.BN4(self.conv3(h)))
        #print(h.size())
        
        h = self.flat1(h)
        #print(h.size())
        h = self.act(self.BN1(self.dense1(h)))
        #print(h.size())
        #means = self.dense_means(h)
        #print(means.size())
        #log_var = self.dense_log_var(h)
        #print(log_var.size())
        return self.dense_means_logVar(h)
        
        #sample = self.reparameterize(means, log_var)
        
        #return sample, means, log_var
        #return means, log_var

class DecoderLikeDQNWBN(nn.Module):
    def __init__(self, input_dim: int, num_channels: int, latent_dim: int):
        super(DecoderLikeDQNWBN, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels

        self.dense1 = nn.Linear(latent_dim, 512)
        self.BN1 = nn.BatchNorm1d(512)
        self.dense2 = nn.Linear(512, 6400)        
        self.BN2 = nn.BatchNorm1d(6400)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.BN3 = nn.BatchNorm2d(32)
        self.upconv2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.BN4 = nn.BatchNorm2d(32)
        self.upconv3 = nn.ConvTranspose2d(32, num_channels, 8, 4, 0)

        self.act = nn.ReLU(inplace=True)
        

    def forward(self, z: TensorType["batch", "input_dim"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        #print("encoder: ")
        h = self.act(self.BN1(self.dense1(z)))
        h = self.act(self.BN2(self.dense2(h)))
        h = h.view(-1, 64, 10, 10)
        #print(h.size())
        h = self.act(self.BN3(self.upconv1(h)))
        #print(h.size())
        h = self.act(self.BN4(self.upconv2(h)))
        #print(h.size())
        img = self.upconv3(h)
        #print(img.size())
        return img

class EncoderWrong(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, latent_dim: int):
        super(EncoderWrong, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=4, stride=2, padding=1)  # 42 x 42
        self.conv2 = nn.Conv2d(32, 32, 2, 2, 1)  # 21 x 21
        self.conv3 = nn.Conv2d(32, 64, 2, 2, 1)  # 11 x 11
        self.conv4 = nn.Conv2d(64, 64, 2, 2, 1)  # 6 x 6
        self.flat1 = nn.Flatten()
        self.dense1 = nn.Linear(3136, 256) # 6x6x 64 = 2304
        self.dense_means_logVar = nn.Linear(256, latent_dim*2)
        #self.dense_log_var = nn.Linear(256, latent_dim)

        self.act = nn.ReLU(inplace=True)
    
    
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
    
    
    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "output_dim"], TensorType["batch", "output_dim"]):
        #print("encoder: ")
        #print(x.size())
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
        h = self.act(self.dense1(h))
        #print(h.size())
        #means = self.dense_means(h)
        #print(means.size())
        #log_var = self.dense_log_var(h)
        #print(log_var.size())
        return self.dense_means_logVar(h)
        
        #sample = self.reparameterize(means, log_var)
        
        #return sample, means, log_var
        #return means, log_var

# In[9]:


class VAE(nn.Module):
    def __init__(self, z_dim, num_channels, device, latent_dim):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.device = device
        if likeDQN == "likeDQN":
            self.encoder = EncoderLikeDQN(z_dim, num_channels, latent_dim) # use "wrong" encoder
            self.decoder = DecoderLikeDQN(z_dim, num_channels, latent_dim)
        elif likeDQN == "NotDQN":
            self.encoder = Encoder(z_dim, num_channels, latent_dim) # use "wrong" encoder
            self.decoder = Decoder(z_dim, num_channels, latent_dim)
        elif likeDQN == "NotDQNWOBN":
            self.encoder = EncoderWOBN(z_dim, num_channels, latent_dim) # use "wrong" encoder
            self.decoder = DecoderWOBN(z_dim, num_channels, latent_dim)
        elif likeDQN == "likeDQNWOBN":
            self.encoder = EncoderLikeDQNWOBN(z_dim, num_channels, latent_dim) # use "wrong" encoder
            self.decoder = DecoderLikeDQNWOBN(z_dim, num_channels, latent_dim)
        elif likeDQN == "likeDQNWBN":
            # self.encoder = EncoderLikeDQNWBN(z_dim, num_channels, latent_dim) # use "wrong" encoder
            self.encoder = EncoderWrong(z_dim, num_channels, latent_dim) # use "wrong" encoder
            self.decoder = DecoderLikeDQNWBN(z_dim, num_channels, latent_dim)
        else:
            print("like DQN not correct!!")
            exit()

            
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0
        self.mse = 0
        self.bce = 0
        self.tc = 0
        self.to(device)
        #self.rec_loss = nn.MSELoss() #try BCE Loss
        self.rec_loss = nn.BCELoss(reduction=isSum) #() like that we have a high BCE loss maybe we can go higher with beta
        #self.rec_loss = nn.BCEWithLogitsLoss() #clamp input values betweeen 0 & 1 use without sigmoid after last output
        # self.logsigmoid = nn.LogSigmoid()
        # self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        
        
    def gaussian_log_density(self, z_sampled: TensorType["batch", "num_latents"],
                         z_mean: TensorType["batch", "num_latents"],
                         z_logvar: TensorType["batch", "num_latents"]):
        normalization = torch.log(torch.tensor(2. * numpy.pi))
        inv_sigma = torch.exp(-z_logvar)
        tmp = (z_sampled - z_mean)
        return -0.5 * (tmp * tmp * inv_sigma + z_logvar + normalization)    

    def total_correlation(self, z: TensorType["batch", "num_latents"],
                      z_mean: TensorType["batch", "num_latents"],
                      z_logvar: TensorType["batch", "num_latents"]) -> torch.Tensor:
    
        batch_size = z.size(0)
        log_qz_prob = self.gaussian_log_density(z.unsqueeze(1), z_mean.unsqueeze(0), z_logvar.unsqueeze(0))

        log_qz_product = torch.sum(
            torch.logsumexp(log_qz_prob, dim=1),
            dim=1
        )
        log_qz = torch.logsumexp(
            torch.sum(log_qz_prob, dim=2),
            dim=1
        )
        if isSum == "sum":
            return torch.sum(log_qz - log_qz_product)
        else:
            return torch.mean(log_qz - log_qz_product)

    
        
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
       
        
    def num_channels(self):
        return self.encoder.num_channels

    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        z = self.encoder(x).view(x.size(0), self.z_dim, 2)
        if torch.isnan(z).any():
            print("z has NaN")
            print(z)
            # print("*************************************input saved***********")
            # x = x.cpu().detach().numpy()
            # numpy.save( "faulty_batch", x)

            
            
        mu = z[:, :, 0]
        logvar = z[:, :, 1]
        sigma = torch.exp(z[:, :, 1])

        reparam_z = mu + sigma*self.N.sample(mu.shape)

        if isSum == "mean":
            self.kl = 0.5 * (sigma**2 + mu**2 - 2*torch.log(sigma) - 1).mean()
        else:
            self.kl = 0.5 * (sigma**2 + mu**2 - 2*torch.log(sigma) - 1).sum()


        self.tc = self.total_correlation(reparam_z, mu, logvar)
        
        x_t = self.decoder(reparam_z).sigmoid()
        #x_t = self.decoder(reparam_z) #No sigmoid if BCEWithLogitsLoss
        #x_t = self.logsigmoid(x_t) #funktioniert nicht wie gedacht!!

        #if torch.isnan(x_t).any():
            #print(x_t)
        #pred = x_t.clamp(0, 1) #push values between 0 and 1
        #pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred) #vlt muss das noch rein
        
        #self.mse = self.rec_loss(x_t, x)
        self.bce = self.rec_loss(x_t, x)
        return x_t
    
    # TODO: Passe diese Klasse noch an. Vlt geht damit das Kopieren zurück
    
    def encode(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        z = self.encoder(x).view(x.size(0), self.z_dim, 2)            
        mu = z[:, :, 0]
        logvar = z[:, :, 1]
        sigma = torch.exp(z[:, :, 1])
        z[:, :, 1] = sigma
        return z


# In[16]:


if linear == False:
    latentDim = features #für conv model
    BVAE = VAE(latentDim, 4, device, latentDim)
    # modeldir = "/itet-stor/ericschr/net_scratch/BA/models/TCBVAE/ConvTC0.0001_Beta0.00192Lat64lr0.0001-best29_3.dat"
    BVAE.load_state_dict(torch.load(modeldir))
    #BVAE.load_state_dict(torch.load('/itet-stor/ericschr/net_scratch/BA/models/TCBVAE/ConvTC0.0001_Beta0.00192Lat64lr0.0001-best29_3.dat'))
    # BVAE = VAE(latentDim, 4, device, latentDim)
    # BVAE.load_state_dict(torch.load('/itet-stor/ericschr/net_scratch/BA/models/TCBVAE/Buffered/BuffConvB0_TC0_Lat64VAE/ConvB0_TC0_Lat64VAE3-4'))
    BVAE.eval()
    print(f"Model: {modeldir}")
    print(BVAE)
else:
    BVAE = LinearVAE()
    BVAE.load_state_dict(torch.load('/itet-stor/ericschr/net_scratch/BA/models/TCBVAE/linearB5_TC0_Lat16lr0.0001VAEMAR23'))

if BVAEtoDevice == True:
    BVAE.to(device)


# ## Training

# Load Tensorboard extension

# In[17]:


from torch.utils.tensorboard import SummaryWriter


# Import required modules and define the hyperparameters

# In[18]:


import time
import numpy as np
import collections
import datetime
x = datetime.datetime.now()

VISUALIZEtraining = False
MEAN_REWARD_BOUND = -19   #Change to 19.0 !!!not used!!!
max_frame_idx   = 1000000 #1Mio


gamma = 0.99                   
batch_size = 32                
replay_size = 10000            
learning_rate = 1e-4           
sync_target_frames = 1000      
replay_start_size = 10000      

eps_start=1.0
# eps_decay=.9999985 #.999985
eps_min=0.02


# Experience replay buffer

# In[19]:
print(f"randSeed {random_seed} gamma {gamma} batch_size {batch_size} replay_size {replay_size} learning_rate {learning_rate} sync_target_frames {sync_target_frames} replay_start_size {replay_start_size} eps_start {eps_start} eps_decay {eps_decay} eps_min{eps_min}")


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

# In[20]:


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, BVAE, net, epsilon=0.0, device="cpu"):
        
        if VISUALIZEtraining:
            env.render()
            
        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a)
            if BVAEtoDevice == True:
                state_v = state_v.to(device)
            #print(state_v.size(1)) # buffersize
            if linear == True:
                state_v = BVAE.encode((state_v[0]).view(state_v.size(1), -1)) #preprocess with beta vae with bunch of 4
                state_v = state_v.view(1, -1)
            else:
                #state_v = state_v.view(, , state_v.size(2), -1)

                # #***Buffered BVAE ***
                # state_v = state_v.view(1, 4, state_v.size(2), -1)
                # #***Buffered BVAE ende ***

                state_v = BVAE.encode((state_v))
                state_v = state_v.view(-1)
                

            #print(f"state_v: {state_v.size()}")
            if BVAEtoDevice == False:
                state_v = state_v.to(device)
            q_vals_v = net(state_v)
           # print("q_vals_v size: " ) 
           # print(q_vals_v.size())
            
            # _, act_v = torch.max(q_vals_v, dim=1)
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


# In[21]:


def preproBVAElinear(states_TOpreprocess):
    #print("states_TOpreprocess size: ")
    #print(states_TOpreprocess.size())

    for i in range(states_TOpreprocess.size(0)):
        states_processing = states_TOpreprocess[i]
        #print("states_processing size: ")
        #print(states_processing.size())

        #print(i)

        #print("viewed: ")
        #print(states_processing.view(states_processing.size(0), -1).size())
        temp = BVAE.encode(states_processing.view(states_processing.size(0), -1)) #preprocess with beta vae with bunch of 4
        temp = temp[None, :] #expand by an axis [1, 128]
        try:
            states_preprocessed = torch.cat((temp , states_preprocessed), dim=0) #concatinate to finish tensor
        except:
            states_preprocessed = temp
            
    return states_preprocessed


# In[22]:


import datetime
import math
print(">>>Training starts at ",datetime.datetime.now())


# Main training loop

# In[23]:


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

# net = DQN(env.observation_space.shape, env.action_space.n).to(device)
# target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)

writer = SummaryWriter(log_dir=f"/itet-stor/ericschr/net_scratch/BA/DQN_runs/With_Pretrained_conv_Buffer_VAE/summary/{namefolder}", comment=f"-Buff{x.day}_{x.month}_EPSdec{eps_decay}" + DEFAULT_ENV_NAME) 

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
        
        reward = agent.play_step(BVAE, net, epsilon, device=device)
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
        

        if BVAEtoDevice == True:
            states_TOpreprocess = states_TOpreprocess.to(device)
            next_states_TOpreprocess = next_states_TOpreprocess.to(device)
        else:
            states_TOpreprocess = states_TOpreprocess.to('cpu')
            next_states_TOpreprocess = next_states_TOpreprocess.to('cpu')

        if linear == True:
            states_preprocessed = preproBVAElinear(states_TOpreprocess)
            next_states_preprocessed = preproBVAElinear(next_states_TOpreprocess)

        else: #conv layers
            #Ich weiss nicht ob die BIlder so in der korrekten Reihenfolge sind oder die 4 Bilder in Blöcken von 32 nacheinander
            # states_TOpreprocess = states_TOpreprocess.view(-1, 1, states_TOpreprocess.size(2), states_TOpreprocess.size(3))
            # next_states_TOpreprocess = next_states_TOpreprocess.view(-1, 1, next_states_TOpreprocess.size(2), next_states_TOpreprocess.size(3))

            # #***Buffered BVAE***
            # states_TOpreprocess = states_TOpreprocess.view(-1, 4, states_TOpreprocess.size(2), states_TOpreprocess.size(3))
            # next_states_TOpreprocess = next_states_TOpreprocess.view(-1, 4, next_states_TOpreprocess.size(2), next_states_TOpreprocess.size(3))
            # # print(f"states_TOpreprocess: {states_TOpreprocess.size()}")
            # #***Buffered BVAE ende***

            states_preprocessed = BVAE.encode(states_TOpreprocess)
            next_states_preprocessed = BVAE.encode(next_states_TOpreprocess)
            #Wieder dientanglen: (batchsize, Buffer, 84, 84)
            states_preprocessed = states_preprocessed.view(-1, 4, states_preprocessed.size(1), states_preprocessed.size(2))
            next_states_preprocessed = next_states_preprocessed.view(-1, 4, next_states_preprocessed.size(1), next_states_preprocessed.size(2))

            #  #***Buffered BVAE***
            # states_preprocessed = states_preprocessed.view(-1, 1, states_preprocessed.size(1), states_preprocessed.size(2))
            # next_states_preprocessed = next_states_preprocessed.view(-1, 1, next_states_preprocessed.size(1), next_states_preprocessed.size(2))
            # #***Buffered BVAE ende***
            
            #print("states_preprocessed size: ")
            #print(states_preprocessed.size())
            
            
            
        if BVAEtoDevice == False:
            states_preprocessed = states_preprocessed.to(device)
            next_states_preprocessed = next_states_preprocessed.to(device)

        #print("states_preprocessed size: ")
        #print(states_preprocessed.size())
        
        states_v = states_preprocessed.view(states_preprocessed.size(0), -1) #oder batchsize [batchsize, 4* features]
        #print("states_v size: ")
        #print(states_v.size())
        next_states_v = next_states_preprocessed.view(next_states_preprocessed.size(0), -1)
            
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
torch.save(net.state_dict(), f"/itet-stor/ericschr/net_scratch/BA/DQN_runs/With_Pretrained_conv_Buffer_VAE/summary/{namefolder}/" + DEFAULT_ENV_NAME + f"Buff-bestreward{best_mean_reward}-{x.day}_{x.month}.dat")


# In[ ]:


print(">>>Training ends at ",datetime.datetime.now())

