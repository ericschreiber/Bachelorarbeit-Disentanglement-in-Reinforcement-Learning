#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1" #for debugging


# In[2]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torch.profiler
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchtyping import TensorType

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

import random
import numpy
import datetime


matplotlib.style.use('ggplot')
#torch.set_printoptions(profile="full") #print full tensor
torch.set_printoptions(profile="default")

likeDQN = "LikeDQNWBN" #likeDQN oder NotDQN oder LikeDQNWBN

# In[3]:


random_seed = 0 
numpy.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(random_seed)
torch.random.manual_seed(random_seed)


# In[4]:


class Encoder(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, latent_dim: int):
        super(Encoder, self).__init__()
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


# In[5]:


class EncoderLikeDQN(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, latent_dim: int):
        super(EncoderLikeDQN, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0)  # 20 x 20
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 10 x 10
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)  # 10 x 10
        self.flat1 = nn.Flatten()        
        self.dense1 = nn.Linear(6400, 512) # 10x10x 64 = 6400
        self.dense_means_logVar = nn.Linear(512, latent_dim*2)
        #self.dense_log_var = nn.Linear(256, latent_dim)

        self.act = nn.ReLU(inplace=True)
    
    
    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "output_dim"], TensorType["batch", "output_dim"]):
        #print("encoder: ")
        #print(x.size())
        h = self.act(self.conv1(x))
        #print(h.size())
        h = self.act(self.conv2(h))
        #print(h.size())
        h = self.act(self.conv3(h))
        #print(h.size())
        
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

class EncoderLikeDQNWBN(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, latent_dim: int):
        super(EncoderLikeDQNWBN, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0)  # 20 x 20
        self.BN1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 10 x 10
        self.BN2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)  # 10 x 10
        self.BN3 = nn.BatchNorm2d(64)
        self.flat1 = nn.Flatten()        
        self.dense1 = nn.Linear(6400, 512) # 10x10x 64 = 6400
        self.BN4 = nn.BatchNorm1d(512)
        self.dense_means_logVar = nn.Linear(512, latent_dim*2)
        #self.dense_log_var = nn.Linear(256, latent_dim)

        self.act = nn.ReLU(inplace=True)
    
    
    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "output_dim"], TensorType["batch", "output_dim"]):
        #print("encoder: ")
        #print(x.size())
        h = self.act(self.BN1(self.conv1(x)))
        #print(h.size())
        h = self.act(self.BN2(self.conv2(h)))
        #print(h.size())
        h = self.act(self.BN3(self.conv3(h)))
        #print(h.size())
        
        h = self.flat1(h)
        #print(h.size())
        h = self.act(self.BN4(self.dense1(h)))
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
        self.dense2 = nn.Linear(256, 3136)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(32, 32, 2, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(32, num_channels, 4, stride=2, padding=1)

        self.act = nn.ReLU(inplace=True)
        

    def forward(self, z: TensorType["batch", "input_dim"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        #print("decoder: ")
        h = self.act(self.dense1(z))
        h = self.act(self.dense2(h))
        h = h.view(-1, 64, 7, 7)
        #print(h.size())
        h = self.act(self.upconv1(h))
        #print("Transpose 1: " + str(h.size()))
        h = self.act(self.upconv2(h))
        #print("Transpose 2: " + str(h.size()))
        h = self.act(self.upconv3(h))
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
        self.dense2 = nn.Linear(512, 6400)        
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.upconv3 = nn.ConvTranspose2d(32, num_channels, 8, 4, 0)

        self.act = nn.ReLU(inplace=True)
        

    def forward(self, z: TensorType["batch", "input_dim"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        #print("encoder: ")
        h = self.act(self.dense1(z))
        h = self.act(self.dense2(h))
        h = h.view(-1, 64, 10, 10)
        #print(h.size())
        h = self.act(self.upconv1(h))
        #print(h.size())
        h = self.act(self.upconv2(h))
        #print(h.size())
        img = self.upconv3(h)
        #print(img.size())
        return img

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


# In[9]:


class VAE(nn.Module):
    def __init__(self, z_dim, num_channels, device, latent_dim):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.device = device
        if likeDQN == "likeDQN":
            self.encoder = EncoderLikeDQN(z_dim, num_channels, latent_dim) # use "wrong" encoder
            self.decoder = EncoderLikeDQNWBN(z_dim, num_channels, latent_dim)
        elif likeDQN == "NotDQN":
            self.encoder = Encoder(z_dim, num_channels, latent_dim) # use "wrong" encoder
            self.decoder = Decoder(z_dim, num_channels, latent_dim)
        elif likeDQN == "LikeDQNWBN":
            self.encoder = EncoderLikeDQNWBN(z_dim, num_channels, latent_dim) # use "wrong" encoder
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
        self.rec_loss = nn.BCELoss(reduction='sum')
        #self.rec_loss = nn.BCEWithLogitsLoss() #clamp input values betweeen 0 & 1 use without sigmoid after last output
        # self.logsigmoid = nn.LogSigmoid()
        
        
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
        return torch.sum(log_qz - log_qz_product)
        # return torch.mean(log_qz - log_qz_product)


    
        
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
        # self.kl = 0.5 * (sigma**2 + mu**2 - 2*torch.log(sigma) - 1).mean()
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


# Data

# In[10]:


# train_data = numpy.load('/itet-stor/ericschr/net_scratch/BA/train_data100kBufferMAR29.npy')
# val_data = numpy.load('/itet-stor/ericschr/net_scratch/BA/val_data20kBufferMAR29.npy')

train_data = numpy.load('/itet-stor/ericschr/net_scratch/BA/train_data30kBufferBallbetween.npz')['data']
val_data = numpy.load('/itet-stor/ericschr/net_scratch/BA/val_data8kBufferBallbetween.npz')['data']


# Learning Setup

# In[11]:


# leanring parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}")


latentDim = 40 #4*10
epochs = 75
batch_size = 64
kl_wheight = 1 #0.00064
beta = 0.00192

if beta != 0 :
    tc_wheight = beta - 1
else: 
    tc_wheight = 0
tc_wheight = 0.0001

beta = 0.01
tc_wheight = 0

lr = 0.0001


# In[14]:

x = datetime.datetime.now()
#newpath = f"C:/Users/erics/Documents/Programme/Bachelorarbeit/beat_VAE_Pong_runs/runLIKEDQNBeta{beta}Lat{latentDim}"
#newpath = f"C:/Users/erics/Documents/Programme/Bachelorarbeit/beat_VAE_Pong_runs/run1Beta{beta}Lat{latentDim}"
newpath = f"/itet-stor/ericschr/net_scratch/BA/VAE_runs/Buffered/runConv{likeDQN}_TC{tc_wheight}_Beta{beta}Lat{latentDim}Lr{lr}klWheight{kl_wheight}"
newpath = newpath + f"/outputBeta{x.day}-{x.month}"

if not os.path.exists(newpath):
    os.makedirs(newpath)
    
savingDir = newpath + "/epoch"

# In[15]:


train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True, #this instructs DataLoader to use pinned memory and enables faster and asynchronous memory copy from the host to the GPU.
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
)

# In[16]:


#enc = Encoder(latentDim, 1, latentDim).to(device)
#dec = Decoder(latentDim, 1, latentDim).to(device)
#optEnc = optim.Adam(enc.parameters(), lr=lr)
#optDec = optim.Adam(dec.parameters(), lr=lr)

#print(enc)
#print(dec)


vae = VAE(latentDim, 4, device, latentDim).to(device)
opt = optim.Adam(vae.parameters(), lr=lr)

#criterion = nn.MSELoss(reduction='sum').to(device)

print(vae)


# In[17]:


def final_loss(mse_loss, mu, logvar, beta, kl_wheight):
    """
    This function will add the reconstruction loss (MSELoss) and the (one could also take the mse loss instead of bce then we get a kind of PCA)
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    MSE = mse_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta*kl_wheight*KLD


# In[18]:


def reparameterize(mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample


# Training Loop

# In[19]:


#def fit(enc, dec, dataloader):
def fit(vae, dataloader, frame_idx, writer):
    #enc.train()
    #dec.train()
    vae.train
    running_loss = 0.0
   # with torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=10),
   #                             on_trace_ready=torch.profiler.tensorboard_trace_handler('C:/Users/erics/Documents/Programme/Bachelorarbeit/Profiler/BVAE/bestermann_MAR9_VAE_Class_Run8_runningLoss/'),
   #                             record_shapes=True,
   #                             profile_memory=True,
   #                             with_stack=True) as prof: 
        
   #     prof.start()
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data = data.to(device)
        data = data[:,  :, :]

        #optEnc.zero_grad()
        #optDec.zero_grad()
        opt.zero_grad(set_to_none=True)
        #opt.zero_grad()

      #  interData = enc(data)
      #  sample, mu, logvar = interData
        #interData = reparameterize(mu, logvar)
      #  reconstruction = dec(sample)
        vae(data)        

        #mse_loss = criterion(reconstruction, data)
        #loss = final_loss(mse_loss, mu, logvar, beta, kl_wheight = dataloader.batch_size/len(train_data))
        #kl_wheight = dataloader.batch_size/len(train_data) fixer hyperparameter
        #loss = kl_wheight * beta * vae.kl + vae.mse
        loss = kl_wheight * beta * vae.kl + vae.bce + tc_wheight * vae.tc
        
        writer.add_scalar("wheighted loss", loss, frame_idx)
        writer.add_scalar("KL", vae.kl, frame_idx)
        writer.add_scalar("BCE", vae.bce, frame_idx)
        writer.add_scalar("TC", vae.tc, frame_idx)

        #running_loss += loss.item()
        running_loss += loss.detach().cpu().numpy() # faster with detach().cpu().numpy() but double the copied amount just for plotting purposes
        loss.backward()
      #  optEnc.step()
      #  optDec.step() 

        #cont = False
        for param in vae.parameters():    
            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=None, neginf=None)
        #     if torch.isnan(param.grad).any() or torch.isinf(param.grad).any(): #if there is a faulty gradient ignore whole gradient
        #         cont = True
        # if cont:
        #    continue

        #torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=2.0, norm_type=2.0) #args.clip) #clipping gradient
        opt.step()

        frame_idx += 1
        #print(opt.param_groups[0]['lr'])

       #     prof.step()
       #     if(i > 100):
       #         break
       # prof.stop()
    writer.flush()
        
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss, frame_idx, writer


# In[20]:


#def validate(enc, dec, dataloader):
def validate(vae, dataloader):
    #enc.eval()
    #dec.eval()
    vae.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data = data.to(device)
            data = data[:, :, :]
            
          #  interData = enc(data)
          #  sample, mu, logvar = interData
          #  reconstruction = dec(sample)
                
          #  mse_loss = criterion(reconstruction, data)
          #  loss = final_loss(mse_loss, mu, logvar, beta, kl_wheight = dataloader.batch_size/len(val_data))
          #  running_loss += loss.item()
            reconstruction = vae(data)        
            loss = kl_wheight * beta * vae.kl + vae.bce + tc_wheight * vae.tc
            
            running_loss += loss.detach().cpu()
            
            # save the last batch input and output of every epoch
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 4, 84, 84)[:8], 
                                  reconstruction.view(batch_size, 4, 84, 84)[:8]))
                save_image(both.cpu(), savingDir + f"{epoch}.png", nrow=num_rows)
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss


# In[21]:


train_loss = []
val_loss = []
torch.backends.cudnn.benchmark = True #choose best kernel for computation

writer = SummaryWriter(log_dir="/itet-stor/ericschr/net_scratch/BA/VAE_runs/Buffered/summary/" + f"Buf{likeDQN}convTC{tc_wheight}_Beta{beta}Lat{latentDim}lr{lr}/{x.day}-{x.month}", comment="-" )
writer2 = SummaryWriter(log_dir="/itet-stor/ericschr/net_scratch/BA/VAE_runs/Buffered/summary/" + f"_EPOCHS_Buf{likeDQN}convTC{tc_wheight}_Beta{beta}Lat{latentDim}lr{lr}/{x.day}-{x.month}", comment="-" )
frame_idx = 0

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    #train_epoch_loss = fit(enc, dec, train_loader)
    #val_epoch_loss = validate(enc, dec, val_loader)
    
    train_epoch_loss, frame_idx, writer = fit(vae, train_loader, frame_idx, writer)
    val_epoch_loss = validate(vae, val_loader)

    writer2.add_scalar("train epoch loss", train_epoch_loss, frame_idx)
    writer2.add_scalar("validation epoch loss", val_epoch_loss, frame_idx)
           
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")

writer.close()
writer2.close()


# * Auslastung GPU: copy ~90%, vram 100% ( 2GB), 3D 0%
# * Auslastung CPU: ~25 %

# In[ ]:


newpath = f"/itet-stor/ericschr/net_scratch/BA/models/TCBVAE/Buffered/BuffConvB{beta}_TC{tc_wheight}_Lat{latentDim}VAE/"
if not os.path.exists(newpath):
    os.makedirs(newpath)

torch.save(vae.state_dict(), newpath + f"ConvB{beta}_TC{tc_wheight}_Lat{latentDim}VAE{x.day}-{x.month}")


