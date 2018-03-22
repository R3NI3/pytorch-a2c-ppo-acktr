# pytorch-a2c-ppo-acktr

## Update 10/06/2017: added enjoy.py and a link to pretrained models!
## Update 09/27/2017: now supports both Atari and MuJoCo/Roboschool!

This is a PyTorch implementation of
* Advantage Actor Critic (A2C), a synchronous deterministic version of [A3C](https://arxiv.org/pdf/1602.01783v1.pdf)
* Proximal Policy Optimization [PPO](https://arxiv.org/pdf/1707.06347.pdf)
* Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation [ACKTR](https://arxiv.org/abs/1708.05144)

Also see the OpenAI posts: [A2C/ACKTR](https://blog.openai.com/baselines-acktr-a2c/) and [PPO](https://blog.openai.com/openai-baselines-ppo/) for more information.

This implementation is inspired by the OpenAI baselines for [A2C](https://github.com/openai/baselines/tree/master/baselines/a2c), [ACKTR](https://github.com/openai/baselines/tree/master/baselines/acktr) and [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1). It uses the same hyper parameters and the model since they were well tuned for Atari games.

## Supported (and tested) environments (via [OpenAI Gym](https://gym.openai.com))
* [Atari Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
* [MuJoCo](http://mujoco.org)
* [PyBullet](http://pybullet.org) (including Racecar, Minitaur and Kuka)

I highly recommend PyBullet as a free open source alternative to MuJoCo for continuous control tasks.

All environments are operated using exactly the same Gym interface. See their documentations for a comprehensive list.

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [Vrep](http://www.coppeliarobotics.com/)
* [PyTorch](http://pytorch.org/)
* [Visdom](https://github.com/facebookresearch/visdom)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

## Installing Walkthrough

Download vrep
```
	http://www.coppeliarobotics.com/downloads.html
```
Download and Install Conda
```
	https://www.anaconda.com/download/#linux
	i - bash Anaconda3-5.0.1-Linux-x86_64.sh
	ii - Add Anaconda to Enviroment PATH
	iii - source ~/.bashrc
```
Install pytorch
```
	conda install pytorch-cpu torchvision -c pytorch
```
Install Baselines
```
	i - git clone https://github.com/openai/baselines.git
	ii - cd baselines
	iii - pip install -e .
		problems here:
		a) Atari problems:   
			sudo apt-get install zlib1g-dev
			sudo apt install cmake
		b) Mujoco problems:
			mpi4py - sudo apt install libopenmpi-dev  
		    Delete mujoco if you are not using it:
			    edit ./baselines/setup.py
					locate the line containing 'gym[mujoco,atari,classic_control,robotics]'
					delete mujoco and robotics, resulting in 'gym[atari,classic_control]'
					save setup.py
			run pip install -e .
```
Install Work branch
```
	i  - git clone https://github.com/R3NI3/pytorch-a2c-ppo-acktr.git
	ii - pip install -r requirements.txt
	iii- git fetch --all
	iv - switch to proper devel branch
```
Install Extra packages
```
	i - pip install opencv-python
	ii- pip install visdom
```

Obs: if you need to run mujoco go back to Mujoco problems and install mujoco as explained in https://github.com/openai/mujoco-py

## Contributions

Contributions are very welcome. If you know how to make this code better, please open an issue. If you want to submit a pull request, please open an issue first. Also see a todo list below.

Also I'm searching for volunteers to run all experiments on Atari and MuJoCo (with multiple random seeds).

## Disclaimer

It's extremely difficult to reproduce results for Reinforcement Learning methods. See ["Deep Reinforcement Learning that Matters"](https://arxiv.org/abs/1709.06560) for more information. I tried to reproduce OpenAI results as closely as possible. However, majors differences in performance can be caused even by minor differences in TensorFlow and PyTorch libraries.

### TODO
* Improve this README file. Rearrange images.
* Improve performance of KFAC, see kfac.py for more information
* Run evaluation for all games and algorithms

## Training

Start a `Visdom` server with `python -m visdom.server`, it will serve `http://localhost:8097/` by default.
Start Vrep if you're going to train with it

### Vrep
```
Open Vrep server
Switch to branch devel_vrepEnv_Renie_v0 
```

Run
```bash
python main.py --env-name "vrep_soccer-v0" --num-processes 1
```

### Atari
#### A2C

```bash
python main.py --env-name "PongNoFrameskip-v4"
```

#### PPO

```bash
python main.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --num-processes 8 --num-steps 128 --num-mini-batch 4 --vis-interval 1 --log-interval 1
```

#### ACKTR

```bash
python main.py --env-name "PongNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20
```

### MuJoCo
#### A2C

```bash
python main.py --env-name "Reacher-v1" --num-stack 1 --num-frames 1000000
```

#### PPO

```bash
python main.py --env-name "Reacher-v1" --algo ppo --use-gae --vis-interval 1  --log-interval 1 --num-stack 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-frames 1000000
```

#### ACKTR

ACKTR requires some modifications to be made specifically for MuJoCo. But at the moment, I want to keep this code as unified as possible. Thus, I'm going for better ways to integrate it into the codebase.

## Enjoy

Load a pretrained model from [my Google Drive](https://drive.google.com/open?id=0Bw49qC_cgohKS3k2OWpyMWdzYkk).

Also pretrained models for other games are available on request. Send me an email or create an issue, and I will upload it.

Disclaimer: I might have used different hyper-parameters to train these models.

### Atari

```bash
python enjoy.py --load-dir trained_models/a2c --env-name "PongNoFrameskip-v4" --num-stack 4
```

### MuJoCo

```bash
python enjoy.py --load-dir trained_models/ppo --env-name "Reacher-v1" --num-stack 1
```
