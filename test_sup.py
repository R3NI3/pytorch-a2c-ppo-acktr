import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from vss_dataset import vss_dataset
from torch.utils.data import DataLoader
from gym import spaces
import argparse

from model import CNNPolicy, MLPPolicy

parser = argparse.ArgumentParser(description='SUP')
parser.add_argument('--train-dir', default='./Datasets/train',
                    help='directory with logs to train (default: ./Datasets/train)')
parser.add_argument('--test-dir', default='./Datasets/test',
                    help='directory with logs to test (default: ./Datasets/test)')
parser.add_argument('--batch-size', type=int, default=10,
                    help='batch size (default: 10)')
args = parser.parse_args()

has_cuda = torch.cuda.is_available()

train_dataset = vss_dataset(args.train_dir)
train_data = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

test_dataset = vss_dataset(args.test_dir)
test_data = DataLoader(test_dataset, batch_size=1,
                        shuffle=False, num_workers=4)

action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(6,))
observation_space = spaces.Box(low=-200, high=200, dtype=np.float32, shape=(20,))
obs_shape = observation_space.shape

actor_critic = MLPPolicy(obs_shape[0], action_space)
if has_cuda:
    actor_critic.cuda()
optimizer = optim.RMSprop(actor_critic.parameters(), lr=7e-4, eps=1e-5, alpha=0.99, weight_decay = 1e-1)

loss_func = nn.MSELoss()

def train(num_epochs = 100000):
    for epc in range(num_epochs):
        train_loss = 0
        for idx,(obs,act) in enumerate(train_data):
            obs = Variable(obs)
            if has_cuda:
                obs.cuda()
            act = Variable(act, requires_grad=False)
            # Sample actions
            optimizer.zero_grad()
            value, pred_action, action_log_prob, states = actor_critic.act(obs,
                                                                      0,
                                                                      0,
                                                                      deterministic = True)
            loss = loss_func(pred_action, act)
            train_loss += loss.data[0]
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epc, train_loss / len(train_dataset)))
        epc +=1


def test():
    actor_critic.eval()
    for idx,(obs,act) in enumerate(test_data):
        obs = Variable(obs)
        act = Variable(act, requires_grad=False)
        value, pred_action, action_log_prob, states = actor_critic.act(obs,
                                                                      0,
                                                                      0,
                                                                      deterministic = True)
        loss = loss_func(pred_action, act)
        print('Test sample: {}\tAction {}\tPredicted {}\tLoss: {:.6f}'.format(
                    idx,act,pred_action, loss.data[0]))

def main():
    train(100000)
    test()
    #torch.save({
    #    'epoch': epochs + 1,
    #    'state_dict': model.state_dict(),
    #    'optimizer' : optimizer.state_dict()}, path_resume)


if __name__ == "__main__":
    main()

