# Disentanglement for Reinforcement Learning
Bachelor Thesis of Eric Schreiber FS 2022. 

PDF of the [Project Description](https://tik-db.ee.ethz.ch/file/5157fc98c588fe3b8d9e639ea9238f0c/DisentanglementRLBait.pdf) and the [Bachelor Thesis](Bachelor_Thesis.pdf)

Generally I did everything on the cluster expect constructing the datasets and plots. Those are python notebooks which i ran on my laptop. These Python notebooks can be found in the notebooks folder. 

Please check all location where you import a dataset. I moved all the datasets into one folder. You may have to change those paths.
## Installation & Usage
Install the conda environment with
```bash
conda env create --file=PyTorchRLclean.yml
```
The PyTorchRL environment has all neccerssary packages for training the VAEs and the reinforcement learning parts. If this env does not work try the one from the server branch.

## What is Where
- All scripts are in the Notebooks folder. 
- All Datasets are zipped in one file if you need them. If you want a single one you can take them as well from the server branch.
- Some nice bash commands are in the file commands.txt That was just a collection for myself.

## Executable Python Scripts
In general use the scripts of the server branch. These scripts are mostly not on the server. Those in bold might be usefull. If something is missing it might be on the dev branch. But I should have taken everything important to the main or server branch.

### Plots
**make_Plots.ipynb** -> Notebook to import tensorboard data which is uploaded online and make plots from it. 

*latent spaces Beta changing conv.ipynb* -> Looking at the KL Divergence of the latent space while changing beta. 

*latent spaces Beta changing.ipynb* -> Looking at the KL Divergence of the latent space while changing beta but with fully linear VAEs.

*latent spaces.ipynb* -> Looks at the KL Divergence of a trained linear VAE.

**latent spaces conv.ipynb** -> Latent spaces traversal and some other plots of the KL divergence in conv VAEs

### Generate Data

**Generate_Data.ipynb** -> Generates a dataset from the environment. The problem is you have to use a pretrained DQN. Should work for all Atari games

**Generate_Data_hardcode.ipynb** -> Generate images that look like it comes from the environment. It can also produce the corresponding ground truth labels. Works for both Pong and Breakout.

**Generate_Data_downsampling84to64.ipynb** -> Downsamples a dataset to 64x64. This might be helpfull if you take the VAEs from Benjamin

### Ground Truths
**Get_GT_from_images.ipynb**      ->    Calculate where in an image is the ball and paddle. This should work fine but is quite computationally intense.

*GT_from_image.ipynb*     ->    Traines a NN to predict the location of paddles and the ball. This did not work fine because the form coming from the environment always changes. You have to train it differently.

## Cluster side
The most important scripts are here up to date. They can be found in the cluster folder.
## Installation & Usage
Install the conda environment on the cluster with
```bash
conda env create --file=PyTorchRLFINAL.yml
conda env create --file=DisMetricsFINAL.yml
```
The PyTorchRL environment has all neccerssary packages for training the VAEs and the reinforcement learning parts. To get the metrics of a model one additionally needs TF which is contained in the DisMetrics environment.

You don't need to download all files. The folders /models, /runs, /DQN_runs and /VAE_runs just contain Information about training runs and trained models. There are also quite a lot of datasets which you probably don't need all.

To start a file run the bash file like
```bash
bash /path_to_/Executables/name.sh 
```
and add the commented neccessary hyperparameters at the end. In all bash files there should be an example of how to do it.
## What is Where
- All training scripts are in the Executables folder. Each script has its own bash file that lets it run on the cluster. 
- Models are all saved in the models folder. The Metrics are also setup in a way that the scores are saved into the corresponding models folder. 
- The images taken during VAE training each epoch are saved in the VAE_runs folder. 
- The Metrics can be computed by running /Executables/Metrics.sh. All Metrics related stuff is in the Dsentanglement_metrics folder. However I am not happy with the way I set up. So its best if you just download it from Benjamin again. 

## Executable Python Scripts
In general LikeDQN means that it has the same structure as the Deep-Q net given in the [Playing Atari games](https://arxiv.org/abs/1312.5602) Paper. NotDQN is another architecture.

### AE and VAE
*Autoencoder.py*     ->    Trains a normal autoencoder on the provided dataset.

*Buffered_Conv_Beta_TC_VAE.py*     ->    Can train a Beta VAE and a Beta TC VAE on a buffer of 4 input images given as the color chanel. This was excluded in my thesis since it did not provide a gain in comparison to just plainly encode all images individually. The code is not up to date and should be adapted to match the Conv_Beta_TC_VAE.py if you want to use it.

*Conv_Beta_TC_VAE.py*     ->    Can train a Beta VAE and a Beta TC VAE. Beta and TC wheight are seperate hyperparameters. Keep in mind that if you want to train a TCB VAE you should put Beta = 1 and TC = your_Beta -1.

*Gridsearch_Beta_TC_VAE.py* -> A script to make a gridsearch over some range of Beta and TC values. This can also be done as single jobs on the cluster itself but this way I dont flood the whole queue (which some students did on a regular basis and it's extremly annoying for the others so don't do it)

*L1_VAE_Conv.py* -> Training script for the JL1-VAE.      

*linear_Beta_TC_VAE_Pong.py* -> Training script for a linear VAE. This script was not used since convolutional VAEs performed better. But it might still be worth a try.

### Reinforcement Learning

*DQN_with_pretrained_Buffer_VAE_Pong.py* -> Using a buffer VAE. Performed worse then just encoding each image individually and is therefore not important. (so far only works for Pong)

*DQN_with_pretrained_BVAE_Pong.py* -> Trains a DQN with a pretrained VAE encoder as a preprocessing step. 

*DQN_with_pretrained_GT_net.py* -> Calculates the ground truth factors from an image and uses them as the input to the DQN. (so far only works for Pong)

*DRL_TowardsDataScience_Pong.py* -> Original implementation of the DQN. 
