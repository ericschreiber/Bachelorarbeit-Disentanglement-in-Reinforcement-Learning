Cart-Pole project is done with '100%' accuracy (200/200 points every time)
/*In the Pong project I am having trouble importing the environment with gym. It only seems to work with ale.
But I could not find a source for training with ale environment.
I finally got it to work. Since it is late in the evening i will postpone doing the rest. I have to set it up in a new file.*/
Pong is working. But training takes a long time. I am not surprised, that model still looses. But it should be somewhat better.

Quellen: 

MIT Lecture: 						https://www.youtube.com/watch?v=93M1l_nrhpQ
Arxiv Insights: 					https://www.youtube.com/watch?v=JgvyzIkgxF0
Intro into RL: 						https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

Environment: 						https://gym.openai.com/envs/#classic_control
Keras RL Docs: 						https://keras-rl.readthedocs.io/en/latest/
Tutorial: 						https://www.youtube.com/watch?v=cO5g5qLrLSo
Playing Atari with Deep Reinforcement Learning (2013): 	https://arxiv.org/pdf/1312.5602.pdf
Playing Pong based on the above: 			https://towardsdatascience.com/deep-q-network-dqn-i-bce08bdf2af
Bad implemtation of pong:				https://medium.com/gradientcrescent/fundamentals-of-reinforcement-learning-automating-pong-in-using-a-policy-model-an-implementation-b71f64c158ff
Flappy Birds with RL: 					https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial

Autoencoder:
Tutorial (old): 					https://github.com/bnsreenu/python_for_microscopists/blob/master/178_179_variational_autoencoders_mnist.py
Vlt sehr gut: 						https://github.com/alexbooth/Beta-VAE-Tensorflow-2.0/blob/master/model.py
code based on: 						https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/


Maybe RL with autoencoder: https://github.com/navneet-nmk/pytorch-rl


Conda gym:  conda install -c conda-forge gym-atari 
 	conda install -c conda-forge atari_py 
	pip install ale-py
	pip install gym[atari] #ich glaube das ist das einzige Package das man wirklich braucht.
	#piglet bracuht man auch ( pip install pyglet )


Fragen:
	Ich verliere Präzision im mean_loss & epsilon Why?? (DRL_TowardsDataScience_Pong.ipynb)
	Dimensionen bei VAE stimmen nicht überein
	Gewicht whights aufsummiert gleich 1 (also auch BCE gewichten)
	Verschiedene Betas versuchen
	
	R^2 wert nutzen um correlation zu finden -> minimale anzahl features (Beta reduzieren bis schwellenwert von r^2 erreicht. R^2 gibt prozent der eingefangenen varianz der Trainingsdaten durch das model an.)
	PCA statt Beta VAE nutzen mit R^2

	Lerne BVAE mit buffered input data weil sonst preprocessing viel zu lange dauert

	
