import argparse
import os
import types

import numpy as np
import torch
from torch.autograd import Variable
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from model import CNNPolicy, MLPPolicy

from envs import make_env

import vrep_env


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=4,
                    help='number of frames to stack (default: 4)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-file', default=None,
                    help='directory to save agent logs (default: None)')
args = parser.parse_args()


env = make_env(args.env_name, args.seed, 0, None)
env = DummyVecEnv([env])

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

if len(env.observation_space.shape) == 3:
        actor_critic = CNNPolicy(obs_shape[0], env.action_space, args.recurrent_policy)
else:
    actor_critic = MLPPolicy(obs_shape[0], env.action_space)

if (args.load_file):
    if os.path.isfile(args.load_file):
        print("=> loading checkpoint '{}'".format(args.load_file))
        checkpoint = torch.load(args.load_file)
        updates = checkpoint['epoch']
        actor_critic.load_state_dict(checkpoint['state_dict'])
        ob_rms = checkpoint['ob_rms']
else:
    exit()

if len(env.observation_space.shape) == 1:
    env = VecNormalize(env, ret=False)
    env.ob_rms = ob_rms

    # An ugly hack to remove updates
    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs
    env._obfilt = types.MethodType(_obfilt, env)
    render_func = env.venv.envs[0].render
else:
    render_func = env.envs[0].render

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
current_obs = torch.zeros(1, *obs_shape)
states = torch.zeros(1, actor_critic.state_size)
masks = torch.zeros(1, 1)


def update_current_obs(obs):
    shape_dim0 = env.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs

obs = env.reset()
update_current_obs(obs)

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

while True:
    value, action, _, states = actor_critic.act(Variable(current_obs, volatile=True),
                                                Variable(states, volatile=True),
                                                Variable(masks, volatile=True),
                                                deterministic=True)
    states = states.data
    cpu_actions = action.data.squeeze(1).cpu().numpy()
    # Obser reward and next obs
    obs, reward, done, _ = env.step(cpu_actions)

    masks.fill_(0.0 if done else 1.0)

    if current_obs.dim() == 4:
        current_obs *= masks.unsqueeze(2).unsqueeze(2)
    else:
        current_obs *= masks
    update_current_obs(obs)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)
