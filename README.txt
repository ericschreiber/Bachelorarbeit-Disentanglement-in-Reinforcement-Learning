Cart-Pole project is done with '100%' accuracy (200/200 points every time)
/*In the Pong project I am having trouble importing the environment with gym. It only seems to work with ale.
But I could not find a source for training with ale environment.
I finally got it to work. Since it is late in the evening i will postpone doing the rest. I have to set it up in a new file.*/
Pong is working. But training takes a long time. I am not surprised, that model still looses. But it should be somewhat better.
Das zweite DQN file versucht das preprocessing schneller zu machen und alles in batches auf der GPU auszuführen.

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
Beta VAE Paper						https://openreview.net/pdf?id=Sy2fzU9gl
ConvVAE debuggercoffe:					https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/
ConvVAE DS:						https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
possible Improvement of VAE:				https://openreview.net/pdf?id=ryxOvH86SH
BCE Loss & KL divergence erklärt			https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a			

Other Approaches:		
ATC Learning: 						http://proceedings.mlr.press/v139/stooke21a/stooke21a.pdf
"Almost perfect dis-
entanglement has been shown to be helpful for down-
stream performance and OOD1 generalization even with
MLP downstream task"					https://openreview.net/pdf?id=I8rHTlfITWC (linked to this paper https://arxiv.org/abs/2010.14407v2)

Log Traininginformation & Speed-up:
Pytorch Profiler 					https://opendatascience.com/optimizing-pytorch-performance-batch-size-with-pytorch-profiler/
tensorboard --logdir=C:\Users\erics\Documents\Programme\Bachelorarbeit\Profiler\BVAE

Auflistung zum noch durchlesen:				https://neptune.ai/blog/best-tools-to-log-and-manage-ml-model-building-metadata



Maybe RL with autoencoder: https://github.com/navneet-nmk/pytorch-rl


Conda gym:  conda install -c conda-forge gym-atari 
 	conda install -c conda-forge atari_py 
	pip install ale-py
	pip install gym[atari] #ich glaube das ist das einzige Package das man wirklich braucht.
	#piglet bracuht man auch ( pip install pyglet )


Fragen:
	Ich verliere Präzision im mean_loss & epsilon Why?? (DRL_TowardsDataScience_Pong.ipynb)
	Setze kleine Gewichte = 0 Wie kann man das machen?mit copying
	Conv 2.5x langsamer extrem aufwändig 
	mean and Log Variance in autoencoders (ist dies eine schätzung und wie lernt man dies)
	Benjamin Code:
	    Train Gan:	disc.train()		Was macht das discriminator network?
            		enc.train()
            		dec.train()
				Müssen die nicht miteinander trainiert werden? (Nein da gegeneinander arbeiten)


Ideen:
	Gewicht wheights aufsummiert gleich 1 (also auch BCE gewichten)
	Verschiedene Betas versuchen
	
	R^2 wert nutzen um correlation zu finden -> minimale anzahl features (Beta reduzieren bis schwellenwert von r^2 erreicht. R^2 gibt prozent der eingefangenen varianz der Trainingsdaten durch das model an.)
	PCA statt Beta VAE nutzen mit R^2 -> ****Arbeit vergleiche disentanglement mit PCA.****

	Ein Layer weniger in VAE für speed up
	Versuche Conv BVAE for speed up
	Lerne BVAE mit buffered input data weil sonst preprocessing viel zu lange dauert

	

Notizen Besprechung: Stand 2.3. BVAE auf GPU 14 Games in 6 min 20 sec
		    ursprüngliches Netz: 2 min

	(X) Latent space finden: decoder mit unterschiedlichen inputs füttern
	(X) class definieren für layers
	Layer freezen(.requires_gradfield = false oder  & gewichte reinladen)
	aufschreiben, was ich analysieren möchte.
	anfangsgewichte aufschreiben (set random seeds)
Notizen Besprechung: 10.3.
	(X) Conv mit BCE machen 
	(X) klwheight als fixer Hyperparameter ~1-16
	(X) Cluster testen conv
	(X) latent spaces: dimensionwise kl divergence anschauen 0 space kl < 0.01
	(X) letzter kernel nicht 4 sondern kleiner dafür mehr lin layer oder 1. kernel grösser und 1. grösserer stride.
	(X) 1. Mal trainieren ohne kl divergence. Danach erst beta einführen. Ohne Kl muss gut lernen aber halt alles entglet

Notizen & Fragen: 10.3. - 17.3.
	(X) Im Moment loss function berechnet in vae class. könnte es besser sein criterion zu benutzen? Vlt weniger kopieren?
	Möchtest du Zugrif auf mein Github? Soll ich auf Gitlab hochladen? -> Gitlab
	(X) Pfad bei Job funktioniert nicht. Kein Zugriff?? -> Files executable machen!!
	Brauche unterschiedliche gute Agents um Daten zu sammeln -> Muss noch neue Daten sammeln
	Wie loggt man? Wie loggt man print ausgaben? Gestern 7h laufen lassen aber entweder nicht gelernt oder nicht gespeichert.
	Multiple GPUs 
	Wieviel soll ich nachfragen? Deutlich schneller oder selbständiger?
	Wie spiechert man Anfangs random seeds?
	TODO Vorschläge: Daten sammeln
			Conv trainieren
			

Notizen Besprechung: 17.3.
	(X) Logging mit Tensorboard
	(X) conv nochmals auf cluster
	(X) Beta-TCVAE (anderer loss) wheights ~1-20 = beta
	(X) Gitlab hochladen
	Rsync anstatt scp 
	(X) Funktion schreiben um Daten zu generireen dann random sampeln f(p1, p2, xBall ,yBall) damit kann man dann auch untersuchungen machen
	(X) Wie spiechert man Anfangs random seeds?  aaae/aaae/train_dislibvae.py Zeile 2355
	Überlege setup vergleich 

Notizen & Fragen: 17.3. - 24.3.
	(X) Output relativ spät ins File geschrieben. Gibt es einen Force Command sodass alles zeitnah geschrieben wird?
	(X) gitlab main branch protected ich pushe zu dev
	(X) Fragen ob random_seed korrekt gespeichert und was man damit machen soll?
	(X) 2x2 Ball zu klein für linear netzwerk (beta = tc = 0) auch 3x3 -> mit kleinem Beta geht es 
	(X) TC oder nur bce lässt Werte NaN werden mit beta = 10 stable
	(X) Soll TC negativ sein? Ich nehme absolutwert korrekt?
	(X) Wie soll ich angehen um gute Vergleiche zu machen? Statistisch signifikant schon angesprochen was muss ich dafür machen?
		(5 oder mehr verschiedene random seeds, verschiedene models, )


Notizen Besprechung: 24.03.
	(X) gradient clipping gegen NaN (bei loss backward) Hat ohne funktioniert weiss aber nicht warum :/
		exploding gradient, step auslassen anstatt auf 0 setzen 
	(X) Tc = beta -1
	TC untersuchen
	(X) TC negativ lassen (je negativ desto besser)
	(X) 0&1 werte in game
	(X) Wie generalisiert model auf nicht 0&1 werte? -> analyse (gar nicht)
	(X) lineares model Grösse 2
	(X) DQN mit pretrained (gleichen conv layers)
	falls Zeit: loss aus Paper L1
	(X) genau spezifizieren wie der vergleich ist. (Trainingszeit (performance per step) , random seeds, etc)
	(X) Paper lesen zu Metrics 3 metrics
	(X) falls Zeit mal anschauen Disentaglement metrics (disentanglement lib auf tensorflow)
	

	
Notizen & Fragen: 24.3. - 31.3.
	TC Training nochmals anschauen
	Wie soll ich mit Buffer lernen? Original macht 4 input chanels im Bild. Dies ist schwierig zum lernen, da meine Bilder nicht nacheinander 		sind. -> neuen Datensatz mit 4 Bilder als chanel inputs
	Wie lernt RL -> siehe Tensorboard DQN
	Disentanglement metrics lernen die nochmals ein Model?
	NaN Problem: 	The issue is that Sigmoid (in particular, the so-call logistic function)
			uses exponential to map (-inf, inf) to (0.0, 1.0). But then BCELoss
			turns around and uses log() to map (0.0, 1.0) back to (-inf, inf).
			Mathematically, this is fine, but numerically, using floating-point
			arithmetic, the exponentials can start to saturate, leading to loss
			of precision, and can underflow to 0.0 and overflow to inf, leading
			to infs and nans in your loss function and backpropagation. [https://discuss.pytorch.org/t/bce-loss-vs-cross-entropy/97437/3]

	NaN Problem unterschen aber files gelöscht 
	Github aufgesetzt :/
	
	Neues Dataset mit Buffer von 4 Bilder
	Conv trainieren auf 4 Bilder


Notizen Besprechung: 31.03.
	gradient clipping gegen NaN (bei loss backward) Hat ohne funktioniert weiss aber nicht warum :/
		exploding gradient, step auslassen anstatt auf 0 setzen 
	! Gridsearch TC & Beta
	Längere DQN Trainingruns
	4 chanel auch laufen lassen. Sollte gleich schnell wie normal sein
	Prüfen ob alle loss mean oder alle loss sum sind
	NaN with logistic Log Digits aber unbedingt bei encode() forward wieder sigmoid anwenden!
	new dataset: npz array machen (mit labels und bilder)
	Library für metrics installieren & laufen
	falls Zeit: loss aus Paper L1