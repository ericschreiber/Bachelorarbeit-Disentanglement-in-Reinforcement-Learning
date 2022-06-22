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

import sys
#arguments: path to model, features, likeDQN, namefolder, random seed, epsilon decay, big Agent [normal, big, superbig]
assert(len(sys.argv)==4)

latentDim = int(sys.argv[1])
random_seed = int(sys.argv[2])
likeDQN = str(sys.argv[3])#"likeDQN" #likeDQN oder NotDQN

isSum = "sum" # "sum" or "mean" #sum or mean for the BCE Loss


matplotlib.style.use('ggplot')
#torch.set_printoptions(profile="full") #print full tensor
torch.set_printoptions(profile="default")


assert(isSum=="mean" or isSum=="sum")



# In[3]:


#random_seed = 71 
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
        self.dense_means_logVar = nn.Linear(256, latent_dim)
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


# In[5]:


class EncoderLikeDQN(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, latent_dim: int):
        super(EncoderLikeDQN, self).__init__()
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
        self.dense_means_logVar = nn.Linear(512, latent_dim)
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
            self.decoder = DecoderLikeDQN(z_dim, num_channels, latent_dim)
        elif likeDQN == "NotDQN":
            self.encoder = Encoder(z_dim, num_channels, latent_dim) # use "wrong" encoder
            self.decoder = Decoder(z_dim, num_channels, latent_dim)
        else:
            print("like DQN not correct!!")
            exit()

            
        
        self.bce = 0
        self.to(device)
        #self.rec_loss = nn.MSELoss() #try BCE Loss
        self.rec_loss = nn.BCELoss(reduction=isSum) #() like that we have a high BCE loss maybe we can go higher with beta
        #self.rec_loss = nn.BCEWithLogitsLoss() #clamp input values betweeen 0 & 1 use without sigmoid after last output
        # self.logsigmoid = nn.LogSigmoid()
        # self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        
       
        
    def num_channels(self):
        return self.encoder.num_channels

    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        z = self.encoder(x)

        x_t = self.decoder(z).sigmoid()
        
        self.bce = self.rec_loss(x_t, x)
        return x_t
    




print("start loading data")
train_data = numpy.load('/itet-stor/ericschr/net_scratch/BA/train_data100kBufferWithLabelsBallbetweenPaddles.npz')['data']
val_data = numpy.load('/itet-stor/ericschr/net_scratch/BA/val_data100kBufferWithLabelsBallbetweenPaddles.npz')['data']
train_dataBall = numpy.load('/itet-stor/ericschr/net_scratch/BA/train_data100kBufferWithLabelsONLYball.npz')['data']
val_dataBall = numpy.load('/itet-stor/ericschr/net_scratch/BA/val_data100kBufferWithLabelsONLYball.npz')['data']

print("loaded data")
# Learning Setup

# In[11]:


# leanring parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}")


# latentDim = 32
epochs = 75
batch_size = 32
lr = 0.0001

print(f"Best_BN_firstBall_{likeDQN} Lat{latentDim} Lr{lr} batchsize{batch_size} randomSeed{random_seed}")

# In[14]:


x = datetime.datetime.now()
#newpath = f"C:/Users/erics/Documents/Programme/Bachelorarbeit/beat_VAE_Pong_runs/runLIKEDQNBeta{beta}Lat{latentDim}"
#newpath = f"C:/Users/erics/Documents/Programme/Bachelorarbeit/beat_VAE_Pong_runs/run1Beta{beta}Lat{latentDim}"
# newpath = f"/itet-stor/ericschr/net_scratch/BA/VAE_runs/findBest_BADdis_firstBall_runConv{likeDQN}_FullBN_{isSum}TC{tc_wheight}_Beta{beta}Lat{latentDim}Lr{lr}klWheight{kl_wheight}randomSeed{random_seed}"
newpath = f"/itet-stor/ericschr/net_scratch/BA/VAE_runs/Breakout_findBest_AEncoder_FullBN{likeDQN}_{isSum}Lat{latentDim}Lr{lr}randomSeed{random_seed}"
newpath = newpath + f"/findBest{x.day}-{x.month}"

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

train_loaderBall = DataLoader(
    train_dataBall,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True, #this instructs DataLoader to use pinned memory and enables faster and asynchronous memory copy from the host to the GPU.
)
val_loaderBall = DataLoader(
    val_dataBall,
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


vae = VAE(latentDim, 1, device, latentDim).to(device)
opt = optim.Adam(vae.parameters(), lr=lr)


print(vae)



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
        data = data[:, None, :, :]

       
        vae.zero_grad()
        opt.zero_grad(set_to_none=True)
        
        vae(data) 

        #vae.kl = torch.nan_to_num(vae.kl, nan=0.0, posinf=None, neginf=None) #put to largest number maybe then training can continue [nicht der Fall bleibt einfach stecken]

        
        loss =  vae.bce 

        writer.add_scalars('Losses', {'wheighted loss':loss,
                                    'BCE': vae.bce,
                                    }, frame_idx)
        



        running_loss += loss.detach().cpu().numpy() # faster with detach().cpu().numpy() but double the copied amount just for plotting purposes
        loss.backward()

        # cont = False
        for param in vae.parameters():    
            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=None, neginf=None)
        #     if torch.isnan(param.grad).any() or torch.isinf(param.grad).any(): #if there is a faulty gradient ignore whole gradient
        #         cont = True
        # if cont:
        #    continue

        #torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=2.0, norm_type=2.0) #args.clip) #clipping gradient
        opt.step()

        frame_idx += 1
        
    writer.flush()
        
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss, frame_idx, writer




def validate(vae, dataloader):
    vae.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data = data.to(device)
            data = data[:, None, :, :]
            
          
            reconstruction = vae(data)        
            loss = vae.bce 
            
            running_loss += loss.detach().cpu()
            
            # save the last batch input and output of every epoch
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 1, 84, 84)[:8], 
                                  reconstruction.view(batch_size, 1, 84, 84)[:8]))
                save_image(both.cpu(), savingDir + f"{epoch}.png", nrow=num_rows)
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss


# In[21]:


train_loss = []
val_loss = []
torch.backends.cudnn.benchmark = True #choose best kernel for computation

writer = SummaryWriter(log_dir="/itet-stor/ericschr/net_scratch/BA/VAE_runs/summary/" + f"Breakout_findBest_AEncoder_FullBN{isSum}_E{epochs}_{likeDQN}conv_Lat{latentDim}lr{lr}randomSeed{random_seed}/{x.day}-{x.month}", comment="-" )
#writer2 = SummaryWriter(log_dir="/itet-stor/ericschr/net_scratch/BA/VAE_runs/summary/" + f"SQUARES_EPOCHS_{likeDQN}convTC{tc_wheight}_Beta{beta}Lat{latentDim}lr{lr}/{x.day}-{x.month}", comment="-" )
frame_idx = 0

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}", flush=True)
    #train_epoch_loss = fit(enc, dec, train_loader)
    #val_epoch_loss = validate(enc, dec, val_loader)
    if epoch < 10:
        train_epoch_loss, frame_idx, writer = fit(vae, train_loaderBall, frame_idx, writer)
        val_epoch_loss = validate(vae, val_loaderBall)
    else:
        train_epoch_loss, frame_idx, writer = fit(vae, train_loader, frame_idx, writer)
        val_epoch_loss = validate(vae, val_loader)

    # writer2.add_scalar("train epoch loss", train_epoch_loss, frame_idx)
    # writer2.add_scalar("validation epoch loss", val_epoch_loss, frame_idx)
    # writer.add_scalar("train epoch loss", train_epoch_loss, frame_idx)
    # writer.add_scalar("validation epoch loss", val_epoch_loss, frame_idx)
    # writer.flush()
           
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}", flush=True)

writer.close()
#writer2.close()



# In[ ]:


newpath = f"/itet-stor/ericschr/net_scratch/BA/models/Autoencoder/Breakout_findBest_AEncoder_FullBN{likeDQN}Conv_{isSum}Lat{latentDim}_Epochs{epochs}randomSeed{random_seed}/"
newpath = newpath + f"/output{x.day}-{x.month}/"
if not os.path.exists(newpath):
    os.makedirs(newpath)

torch.save(vae.state_dict(), newpath + f"AE_{likeDQN}_Lat{latentDim}_Epochs{epochs}randomSeed{random_seed}{x.day}-{x.month}")


