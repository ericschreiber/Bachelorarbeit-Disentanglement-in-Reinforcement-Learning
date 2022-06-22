#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchtyping import TensorType
from torch.utils.tensorboard import SummaryWriter


import os
import random
import numpy as np
import datetime


matplotlib.style.use('ggplot')




random_seed = 0 
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(random_seed)
torch.random.manual_seed(random_seed)




features = 16
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
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc0(x))
        x = F.relu(self.enc1(x))

        x = self.enc2(x).view(-1, 2, features)

        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = F.relu(self.dec0(z))
        x = F.relu(self.dec1(x))
        reconstruction = torch.sigmoid(self.dec2(x))
        return z,reconstruction, mu, log_var




features = 16
# define a simple linear VAE #until now normal VAE without Beta
class Linear2VAE(nn.Module):
    def __init__(self):
        super(Linear2VAE, self).__init__()
 
        # encoder 84*84 = 7’056
        self.enc0 = nn.Linear(in_features=84*84, out_features=1024)
        self.enc1 = nn.Linear(in_features=1024, out_features=features*2)
 
        # decoder 
        self.dec0 = nn.Linear(in_features=features, out_features=1024)
        self.dec1 = nn.Linear(in_features=1024, out_features=84*84)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc0(x))
        x = self.enc1(x).view(-1, 2, features)

        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = F.relu(self.dec0(z))
        reconstruction = torch.sigmoid(self.dec1(x))
        return z,reconstruction, mu, log_var




features = 16
# define a simple linear VAE #until now normal VAE without Beta
class Linear1VAE(nn.Module):
    def __init__(self):
        super(Linear1VAE, self).__init__()
 
        # encoder 84*84 = 7’056
        self.enc0 = nn.Linear(in_features=84*84, out_features=features*2)
 
        # decoder 
        self.dec0 = nn.Linear(in_features=features, out_features=84*84)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc0(x))
        x = x.view(-1, 2, features)

        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        reconstruction = torch.sigmoid(self.dec0(z))
        return z,reconstruction, mu, log_var


# Parameters for training



# leanring parameters
layers = 3
epochs = 40
batch_size = 64
beta = 5
kl_wheight = 0.00064
lr = 0.0001

if beta != 0 :
    tc_wheight = beta - 1
    beta = 1
else: 
    tc_wheight = 0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"linear Layers{layers} Beta{beta} Lat{features} Lr{lr} klWheight{kl_wheight} tc_wheight{tc_wheight} batchsize{batch_size}")

x = datetime.datetime.now()
newpath = f"/itet-stor/ericschr/net_scratch/BA/VAE_runs/TCBlinear/runLinearTC{tc_wheight}_Beta{beta}Lat{features}klwh{kl_wheight}Layers{layers}"
newpath = newpath + f"/outputBeta{x.day}-{x.month}"

if not os.path.exists(newpath):
    os.makedirs(newpath)
    
savingDir = newpath + "/epoch"




def gaussian_log_density(z_sampled: TensorType["batch", "num_latents"],
                         z_mean: TensorType["batch", "num_latents"],
                         z_logvar: TensorType["batch", "num_latents"]):
    normalization = torch.log(torch.tensor(2. * np.pi))
    inv_sigma = torch.exp(-z_logvar)
    tmp = (z_sampled - z_mean)
    return -0.5 * (tmp * tmp * inv_sigma + z_logvar + normalization)




def total_correlation(z: TensorType["batch", "num_latents"],
                      z_mean: TensorType["batch", "num_latents"],
                      z_logvar: TensorType["batch", "num_latents"]) -> torch.Tensor:
    
    batch_size = z.size(0)
    log_qz_prob = gaussian_log_density(z.unsqueeze(1), z_mean.unsqueeze(0), z_logvar.unsqueeze(0))

    log_qz_product = torch.sum(
        torch.logsumexp(log_qz_prob, dim=1),
        dim=1
    )
    log_qz = torch.logsumexp(
        torch.sum(log_qz_prob, dim=2),
        dim=1
    )
    return torch.abs(torch.mean(log_qz - log_qz_product))


# In[10]:


def final_loss(reconstruction_loss, mu, logvar, z_sampled, beta, kl_wheight, tc_wheight):
    """
    This function will add the reconstruction loss (MSELoss) and the (one could also take the mse loss instead of bce then we get a kind of PCA)
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    :param z_sampled: sample that will be inputed into the decoder
    """
    REC = reconstruction_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    TC = total_correlation(z_sampled, mu, logvar)
    return REC + beta*kl_wheight*KLD + tc_wheight * TC


# Load Data



train_data = np.load('/itet-stor/ericschr/net_scratch/BA/train_data100kMAR22.npy') #hardcoded random data
val_data = np.load('/itet-stor/ericschr/net_scratch/BA/val_data20kMAR22.npy')


# Model



# transforms why do i need a transform?
transform = transforms.Compose([
    transforms.ToTensor(),
])




train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False
)



if layers == 3:
    model = LinearVAE().to(device)
elif layers == 2:
    model = Linear2VAE().to(device)
elif layers == 1:
    model = Linear1VAE().to(device)
else:
    print(f"wrong layers number: {layers}")

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='sum')
#criterion = torch.nn.MSELoss(reduction = 'sum')
print(model)







# Training Loop (we train the autoencoder on one image in the buffer not on the total buffer. This could also be a nice feature)



def fit(model, dataloader, frame_idx, writer):
    model.train()
    running_loss = 0.0
   # with torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=10),
   #                             on_trace_ready=torch.profiler.tensorboard_trace_handler('C:/Users/erics/Documents/Programme/Bachelorarbeit/Profiler/BVAE/Linear_MAR8/'),
   #                             record_shapes=True,
   #                             profile_memory=True,
   #                             with_stack=True) as prof: 
        
   #     prof.start()
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        #data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        z_sampled, reconstruction, mu, logvar = model(data)
        mse_loss = criterion(reconstruction, data)
        loss = final_loss(mse_loss, mu, logvar, z_sampled, beta, kl_wheight, tc_wheight)

        writer.add_scalar("wheighted loss", loss, frame_idx)

        running_loss += loss.item()
        loss.backward()
        optimizer.step()
     #       prof.step()

     #   prof.stop()
    writer.flush()

    train_loss = running_loss/len(dataloader.dataset)
    return train_loss, frame_idx, writer


# In[19]:


def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            #data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            z_sampled, reconstruction, mu, logvar = model(data)
            mse_loss = criterion(reconstruction, data)
            loss = final_loss(mse_loss, mu, logvar, z_sampled, beta, kl_wheight, tc_wheight)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 1, 84, 84)[:8], 
                                  reconstruction.view(batch_size, 1, 84, 84)[:8]))
                save_image(both.cpu(), savingDir + f"{epoch}.png", nrow=num_rows)
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss


# In[22]:


train_loss = []
val_loss = []

writer = SummaryWriter(log_dir="/itet-stor/ericschr/net_scratch/BA/VAE_runs/TCBlinear/summary/" + f"linearTC{tc_wheight}_Beta{beta}Lat{features}lr{lr}/{x.day}-{x.month}", comment="-" )
writer2 = SummaryWriter(log_dir="/itet-stor/ericschr/net_scratch/BA/VAE_runs/TCBlinear/summary/" + f"_EPOCHS_linearTC{tc_wheight}_Beta{beta}Lat{features}lr{lr}/{x.day}-{x.month}", comment="-" )
frame_idx = 0


for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, frame_idx, writer = fit(model, train_loader, frame_idx, writer)
    val_epoch_loss = validate(model, val_loader)

    writer2.add_scalar("train epoch loss", train_epoch_loss, frame_idx)
    writer2.add_scalar("validation epoch loss", val_epoch_loss, frame_idx)

    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")

writer.close()
writer2.close()
# In[17]:

newpath = f"/itet-stor/ericschr/net_scratch/BA/models/TCBVAE/linearB{beta}_TC{tc_wheight}_Lat{features}VAELayers{layers}"
if not os.path.exists(newpath):
    os.makedirs(newpath)

torch.save(model.state_dict(), newpath + f"linearB{beta}_TC{tc_wheight}_Lat{features}VAE{x.day}-{x.month}")






