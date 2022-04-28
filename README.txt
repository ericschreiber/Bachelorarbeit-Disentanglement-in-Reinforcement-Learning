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
JL1:							https://proceedings.neurips.cc/paper/2021/file/bfd2308e9e75263970f8079115edebbd-Paper.pdf / https://proceedings.neurips.cc/paper/2021/file/bfd2308e9e75263970f8079115edebbd-Supplemental.pdf
Github JL1:						https://github.com/travers-rhodes/jlonevae/blob/main/jlonevae_lib/architecture/vae_jacobian.py#L105
TC VAE (relevance factor):				https://proceedings.neurips.cc/paper/2018/hash/1ee3dfcd8a0645a25a35977997223d22-Abstract.html
Cyclical Annealing Schedule for beta:			https://medium.com/mlearning-ai/a-must-have-training-trick-for-vae-variational-autoencoder-d28ff53b0023

RL:
Decoupling Representation Learning from Reinforcement Learning: http://proceedings.mlr.press/v139/stooke21a/stooke21a.pdf (nicht ganz gelesen)
Atari with RL DQN					https://arxiv.org/pdf/1312.5602.pdf


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


Quotes
"disentangled representation can be defined as one where single
latent units are sensitive to changes in single generative factors, while being relatively invariant to
changes in other factors (Bengio et al., 2013)."   	https://openreview.net/pdf?id=Sy2fzU9gl

We believe that using our approach as an unsupervised pretraining
stage for supervised or reinforcement learning will produce significant improvements for scenarios
such as transfer or fast learning.			https://openreview.net/pdf?id=Sy2fzU9gl (original beta VAE)



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
	(X) gradient clipping gegen NaN (bei loss backward) Hat ohne funktioniert weiss aber nicht warum :/
		exploding gradient, step auslassen anstatt auf 0 setzen 
	(X) ! Gridsearch TC & Beta (5x5 Gridsearch 10 Epochs, log value increase 10, 1, 0.1, 0.01, 0.001)
	(X) Längere DQN Trainingruns
	(X) 4 chanel auch laufen lassen. Sollte gleich schnell wie normal sein [Das war der Grund für die längere Laufzeit]
	(X) Prüfen ob alle loss mean oder alle loss sum sind [Alles sum]
	(X) NaN with logistic Log Digits aber unbedingt bei encode() forward wieder sigmoid anwenden! [War nicht besser als mit grad clipping]
	(X) new dataset: npz array machen (mit labels und bilder)
	Library für metrics installieren & laufen
	falls Zeit: loss aus Paper L1

Notizen & Fragen: 31.3. - 7.4.
	Step auslassen funktioniert bei BCE aber nicht bei LogitsLoss. nan_to_num funktioniert bei beiden. Wobei besser bei Logitsloss als bei BCE
		Nur Logitsloss wird plötzlich alles 0. Weiss nicht warum aber mit nan_to_num oder Continue geht es 
		Training und validation loss werden trotzdem nan ( müssten Loss buffern und nur schreiben falls nichts falsch)
		-> Decide BCE mit nan_to_num (ohne continue) Haltet 20 Epochs durch
	Mit beta > 0.001 kommt immer auf gleichen finalen Trainingsloss Grid_train_losses:
		[-1900.121363046875, -4.094250633544922, -4.094250617980957, -4.09425067565918, -4.0942505978393555, -4.094250721435547]
	Gridsearch eher ernüchternd, aber verschiedene Lat Dimensions deuten darauf hin, dass 6 Dim möglich sind
	Empfelungsschreiben Data Science
	Einstellung, dass log-file während execution gefüllt wird sodass auch gefüllt wenn abgebrochen -> unbuffer

	TODO: Check dimesions after BVAE in DQN with pretrained BVAE
		Letztes DQN run hat nicht gelernt warum??

Notizen Besprechung 7.4.:
	Darla Paper gleiches Problem mit grossem Beta nichts gelernt [reconstruction und input in pretrainied denoising AE dann erst loss berechnen]
	(X) Versuche mal 10 latent dimensions mit Beta grid [ähnliche Resultate]
	(X) Deutlich mehr Epochs ~50-100 [gaussian KL wird nur noch grösser]
	(X) ! L1 Loss implementieren
	(X) ! metrics (naming anpassen, dataset (dim bider, dim color, x, y))

Notizen & Fragen: 7.4. - 14.4.:
	conda install -c eumetsat expect [funktioniert nicht]
	DCI braucht 30 min. Wird alles aus versehen auf cpu gemacht? [4 cpus geben]
	L1 langsam. L1 auf Buffer anwenden (lesen wie sie mit time series umgegangen sind)[wirklich als ganzes Bild??] [4 cpus geben]
	0_1 Lernt viel schneller aber den Ball nicht mit lin Netzwerk
	Beta im korrekten Bereich mit sum bei allem.


Notizen Besprechung 14.4.:
	(X) Ist BCE Loss eine Matrix? [ja innerhalb von BCEloss() torch.Size([64, 1, 84, 84])]
	(X) Datenset Benjamin schicken
	! KL plotten und loss etc. KL muss kleiner werden über zeit nicht grösser! (Besterma code hat noch gaussian plotting im tensorbaord)
	(X) sanity check 6x6 quadrat auf verschiedene Positionen
	(X) ! Batch normalization zwischen den Layers (könnte auch schlechter werden) [auch in den convolutions? effektiv schönere ränder]

Notizen & Fragen: 14.4. - 21.4.:
	untersuche disentanglement bei neuem einfachem Datensatz. Ob mein Code irgendwo falsch ist? [ich glaube nicht aber weiss nicht wo das Problem 				liegt]
	(X) Idee nächste Woche mal nur schreiben? Dann wieder mehr Lust auf verschieden Dinge zu probieren.
		Danke vielmals fürs Durchlesen.
		War gut um nochmals die Basics zu lesen und zu verstehen. 
	Code mit Schieber um latent dimensions zu prüfen? [loggen was aus encoder kommt]
	Steigende KL Divergence why? [zu kleines beta]
	Wieso gibt es in der KL Divergence einen Spike Bei mean und bei sum
		Einfach Loss anpassung auslassen geht nicht dann werden alle zukünftigen KL so gross bleiben eg ausgelassen werden
		Wahrscheinlich okay wenn 75 Epochs log geplotted gibt es immer wieder spikes trotzdem ist der Trend downwards
	Wieso reduziert sich bei mir entweder BCE oder KL Divergence? Sollte sich nicht beides reduzieren?
		In welcher Grössenordnung sollten BCE und KL Divergence zu einander sein? 1:1?
	Scatterplot mit latent dimensions. Mean über latent dimensions sollte nicht auf einem Haufen sein damit es für decoder einfacher wird. ?
	Muss ich bei Encoder in DQN nach dem Encoder noch aus einer prob distribution samplen oder kann ich einfach mean & log var reingeben? (ich glaub ich 		sollte noch samplen da man aber nur samplet um backpropagatin zu nutzen, was wir hier nicht brauchen glaube ich vlt doch nicht)
	Empfehlungsschreiben Wattenhofer einfach angeben den Namen?
	 
Notizen Besprechung 28.4.:
	dataset downscalen auf 64x64 und benjamin schicken
	VAE trainieren und bester nehmen
	groundtruth aus Bild lernen
	DQN mit VAE, normal, und groundtruth lernen. 5 verschiedene random seeds
	
TODO: Fragen überlegen bei den Metrics
	KL divergence wird nan wenn zu gross wird (glaube ich) daher tritt nur bei sum auf
	torch.distributions.kl_div läuft gerade

Ideen: Neural net trainieren mit position & richtung dann das nutzen für DQN
	DQN trainieren mit dem was ich habe