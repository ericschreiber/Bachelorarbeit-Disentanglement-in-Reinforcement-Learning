#!/usr/bin/env python
# coding: utf-8



import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1" #for debugging




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

import random
import numpy


matplotlib.style.use('ggplot')
#torch.set_printoptions(profile="full") #print full tensor
torch.set_printoptions(profile="default")


likeDQN = "likeDQN" #likeDQN oder NotDQN oder L1Convs




random_seed = 0 
numpy.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(random_seed)
torch.random.manual_seed(random_seed)




class EncoderL1(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, latent_dim: int):
        super(EncoderL1, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels

           
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=4, stride=2, padding=1)  # 84 x 84
        self.conv2 = nn.Conv2d(32, 32, 2, 2, 0)  # 43 x 43
        self.conv3 = nn.Conv2d(32, 64, 2, 2, 1)  # 22 x 22
        self.conv4 = nn.Conv2d(64, 64, 2, 2, 1)  # 12 x 12
        self.flat1 = nn.Flatten()
        self.dense1 = nn.Linear(9216, 3136) # 12x12x 64 = 9216
        self.dense2 = nn.Linear(3136, 256) 
        self.dense_means_logVar = nn.Linear(256, latent_dim*2)
        
        self.act = nn.ReLU(inplace=True)
    
    
    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "output_dim"], TensorType["batch", "output_dim"]):
        #print("encoder: ")
        print(f"inputsize: {x.size()}")
        h = self.act(self.conv1(x))
        print(f"conv 1: {h.size()}")
        h = self.act(self.conv2(h))
        print(f"conv 2: {h.size()}")
        h = self.act(self.conv3(h))
        print(f"conv 3: {h.size()}")
        h = self.act(self.conv4(h))
        print(f"conv 4: {h.size()}")
        
        h = self.flat1(h)
        h = self.act(self.dense1(h))
        h = self.act(self.dense2(h))
        
        
        return self.dense_means_logVar(h)
        
        
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
        
        return self.dense_means_logVar(h)
        
      


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
       
        return self.dense_means_logVar(h)
        


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



class DecoderL1(nn.Module):
    def __init__(self, input_dim: int, num_channels: int, latent_dim: int):
        super(DecoderL1, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels

       
        self.dense1 = nn.Linear(latent_dim, 256)
        self.dense2 = nn.Linear(256, 3136)
        self.dense3 = nn.Linear(3136, 9216)  
        
        
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0)
        self.upconv4 = nn.ConvTranspose2d(32, num_channels, 4, stride=2, padding=1)
        
        self.act = nn.ReLU(inplace=True)
        

    def forward(self, z: TensorType["batch", "input_dim"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        #print("encoder: ")
        h = self.act(self.dense1(z))
        h = self.act(self.dense2(h))
        h = self.act(self.dense3(h))
        #h = h.view(-1, 64, 20, 20)
        h = h.view(-1, 64, 12, 12)
        #print(h.size())
        h = self.act(self.upconv1(h))
        #print(h.size())
        h = self.act(self.upconv2(h))
        #print(h.size())
        h = self.act(self.upconv3(h))
        img = self.upconv4(h)        
        #print(img.size())
        return img
# In[8]:


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
        elif likeDQN == "L1Convs":
            self.encoder = Encoder(z_dim, num_channels, latent_dim) 
            self.decoder = Decoder(z_dim, num_channels, latent_dim)
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
        self.rec_loss = nn.BCELoss(reduction='sum')
        #self.rec_loss = nn.BCEWithLogitsLoss() #clamp input values betweeen 0 & 1 use without sigmoid after last output
        
        
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
    
    def decode(self, reparam_z):
        
        return self.decoder(reparam_z)

    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        z = self.encoder(x).view(x.size(0), self.z_dim, 2)
        if torch.isnan(z).any():
            print("z has NaN")
            print(z)
            print("*************************************input saved***********")
            x = x.cpu().detach().numpy()
            numpy.save( "faulty_batch", x)

            
            
        mu = z[:, :, 0]
        logvar = z[:, :, 1]
        sigma = torch.exp(z[:, :, 1])
        reparam_z = mu + sigma*self.N.sample(mu.shape)
        self.kl = 0.5 * (sigma**2 + mu**2 - 2*torch.log(sigma) - 1).sum() #.mean()
        self.tc = self.total_correlation(reparam_z, mu, logvar)
        
        x_t = self.decoder(reparam_z).sigmoid()
        
        self.bce = self.rec_loss(x_t, x)
        return mu, logvar, x_t
    


# Data


train_data = numpy.load('/itet-stor/ericschr/net_scratch/BA/train_data100kBreakout.npz')['data']
val_data = numpy.load('/itet-stor/ericschr/net_scratch/BA/val_data20kBreakout.npz')['data']


#downsample the data [I leave it maybe one day you can use this as well. Its for downsampling to 64x64]
# print("downsampling")

# from torch.nn.functional import interpolate
# train_dataPy= torch.from_numpy(train_data)
# val_dataPy= torch.from_numpy(val_data)

# train_dataPy= train_dataPy[:, None, :, :]
# val_dataPy= val_dataPy[:, None, :, :]

# train_data = interpolate(train_dataPy, scale_factor=(0.5,0.5), mode='bilinear')
# val_data = interpolate(val_dataPy, scale_factor=(0.5,0.5), mode='bilinear')

# train_data = train_data.squeeze()
# val_data = val_data.squeeze()

# print(f"train_data: {train_data.size()}")
# print(f"val_data: {val_data.size()}")

# del train_dataPy
# del val_dataPy
# torch.cuda.empty_cache()
# Learning Setup



# leanring parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latentDim = 32
epochs = 75
batch_size = 64
kl_wheight = 1
beta = 0.5
gamma = 0.005
tc_wheight = 0

lr = 0.0001

print(f"findBest_Breakout_ L1{likeDQN} Gamma{gamma} Beta{beta} Lat{latentDim} Lr{lr} klWheight{kl_wheight} tc_wheight{tc_wheight} batchsize{batch_size}")


import datetime
x = datetime.datetime.now()
newpath = f"/itet-stor/ericschr/net_scratch/BA/VAE_runs/Jacobian_L1/findBest_Breakout_runL1{likeDQN}_Gamma{gamma}_Beta{beta}Lat{latentDim}Lr{lr}klWheight{kl_wheight}"
newpath = newpath + f"/output{x.day}-{x.month}"

if not os.path.exists(newpath):
    os.makedirs(newpath)
    
savingDir = newpath + "/epoch"




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




vae = VAE(latentDim, 1, device, latentDim).to(device)
opt = optim.Adam(vae.parameters(), lr=lr)


print(vae)




def compute_generator_jacobian_image_optimized(model, embedding, epsilon_scale = 0.001, device="cpu"):
    raw_jacobian = compute_generator_jacobian_optimized(model, embedding, epsilon_scale, device)
    # shape is (latent_size, batch_size, numchannels = 1, im_size, im_size)
    jacobian = torch.sum(raw_jacobian, dim=2,keepdim = True)
    return(jacobian)

# output shape is (latent_dim, batch_size, model_output_shape)
def compute_generator_jacobian_optimized(model, embedding, epsilon_scale = 0.001, device="cpu"):
    batch_size = embedding.shape[0]
    latent_dim = embedding.shape[1]
    # repeat "tiles" like ABCABCABC (not AAABBBCCC)
    # note that we detach the embedding here, so we should hopefully
    # not be pulling our gradients further back than we intend
    encoding_rep = embedding.repeat(latent_dim + 1,1).detach().clone() #why hier kopie??
    # define our own repeat to work like "AAABBBCCC"
    delta = torch.eye(latent_dim).reshape(latent_dim, 1, latent_dim)                .repeat(1, batch_size, 1)                .reshape(latent_dim*batch_size, latent_dim)
    delta = torch.cat((delta, torch.zeros(batch_size,latent_dim))).to(device)
    # we randomized this before up to epsilon_scale,
    # but for now let's simplify and just have this equal to epsilon_scale.
    # I'd be _very_ impressed if the network can figure out to make the results
    # periodic with this frequency in order to get around this gradient check.
    epsilon = epsilon_scale     
    encoding_rep += epsilon * delta
    recons = model.decode(encoding_rep)
    temp_calc_shape = [latent_dim+1,batch_size] + list(recons.shape[1:])
    recons = recons.reshape(temp_calc_shape)
    recons = (recons[:-1] - recons[-1])/epsilon
    return(recons)


def jacobian_loss_function(model, mu, logvar, device="cpu"):
    # jacobian shape is (latent_dim, batch_size, output_model_shape)
    # where the first batch_size rows correspond to the first latent dimension, etc.
    jacobian = compute_generator_jacobian_optimized(model, mu, epsilon_scale = 0.001, device=device)
    #print(jacobian.shape)
    latent_dim = jacobian.shape[0]
    batch_size = jacobian.shape[1]
    jacobian = jacobian.reshape((latent_dim, batch_size, -1))
    obs_dim = jacobian.shape[2]
    loss = torch.sum(torch.abs(jacobian))/batch_size
    assert len(loss.shape)==0, "loss should be a scalar"
    return(loss)


# Training Loop



#def fit(enc, dec, dataloader):
def fit(vae, dataloader):
    
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


        opt.zero_grad(set_to_none=True)
        
        mu, logvar, reconstruction = vae(data)        

       
        loss = kl_wheight * beta * vae.kl + vae.bce + tc_wheight * vae.tc
        loss += gamma * jacobian_loss_function(vae, mu, logvar, device)

        running_loss += loss.detach().cpu().numpy() # faster with detach().cpu().numpy() but double the copied amount just for plotting purposes
        loss.backward()
     
        for param in vae.parameters():    
            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=None, neginf=None)

        opt.step()
        #print(opt.param_groups[0]['lr'])

       #     prof.step()
       #     if(i > 100):
       #         break
       # prof.stop()
        
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss




def validate(vae, dataloader):
    #enc.eval()
    #dec.eval()
    vae.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data = data.to(device)
            data = data[:, None, :, :]
            
         
            mu, logvar, reconstruction = vae(data)        
            loss = kl_wheight * beta * vae.kl + vae.bce + tc_wheight * vae.tc
            loss += gamma * jacobian_loss_function(vae, mu, logvar, device)
            
            running_loss += loss.detach().cpu()
            
            # save the last batch input and output of every epoch
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 1, 84, 84)[:8], 
                                  reconstruction.view(batch_size, 1, 84, 84)[:8]))
                save_image(both.cpu(), savingDir + f"{epoch}.png", nrow=num_rows)
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss


print("start training")

train_loss = []
val_loss = []
torch.backends.cudnn.benchmark = True #choose best kernel for computation

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    #train_epoch_loss = fit(enc, dec, train_loader)
    #val_epoch_loss = validate(enc, dec, val_loader)
    
    train_epoch_loss = fit(vae, train_loader)
    val_epoch_loss = validate(vae, val_loader)
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")






newpath = f"/itet-stor/ericschr/net_scratch/BA/models/Jacobian_L1/findBest_Breakout_runL1{likeDQN}_Gamma{gamma}_Beta{beta}Lat{latentDim}Lr{lr}klWheight{kl_wheight}/"
if not os.path.exists(newpath):
    os.makedirs(newpath)

torch.save(vae.state_dict(), newpath + f"runL1{likeDQN}_Gamma{gamma}_Beta{beta}VAE{x.day}-{x.month}")