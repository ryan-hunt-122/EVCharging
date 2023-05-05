import numpy as np

from .critic import Critic
from .actor import Actor

import random
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


class DDPG():

    def __init__(self, shape, max_buffer=100, minibatch_size=16, gamma=0.99, lr=0.0001, mult=5.0):
        self.shape = shape
        self.max_buffer = max_buffer
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.tau = 0.01
        # Randomly initialize critic network Q with weights q
        self.critic = Critic(self.shape)
        # and target critic network Q' with weights q'
        self.target_critic = Critic(self.shape)
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=lr*mult)

        # Randomly initialize actor network M with weights m
        self.actor = Actor(self.shape)
        # and target actor network M' with weights m'
        self.target_actor = Actor(self.shape)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=lr)

        # Initialize replay buffer R
        self.replay_buffer = []

        # Set of action descriptors
        self.action_descriptors = [(i, j) for j in range(0, 10) for i in range(0, j+1)]


    def select_action(self, state, sig):
        # Select action according to current actor policy (and exploration noise?)
        # tens = torch.tensor([state])
        # self.target_actor.forward(tens.type(torch.DoubleTensor))
        u = []
        for loc in self.action_descriptors:
            a = self.actor.forward(to_tensor(np.array([state])).unsqueeze(0), to_tensor(np.array([sig])).unsqueeze(0),
                               to_tensor(np.array([loc[0]])).unsqueeze(0), to_tensor(np.array([loc[1]])).unsqueeze(0))
            a = a.detach().tolist()[0][0]
            u.append(a)
        return u

    def store_transition(self, transition):
        a, b, c, d, e = transition

        transition = (a.tolist(), b, c, d, e)

        if len(self.replay_buffer) > self.max_buffer:
            self.replay_buffer = self.replay_buffer[1:]
        self.replay_buffer.append(transition)

    def sample_minibatch(self):
        if len(self.replay_buffer) >= self.minibatch_size:
            return random.sample(self.replay_buffer, self.minibatch_size)
        return self.replay_buffer

    def yi(self, transition):
        yi = transition[2]
        u = to_tensor(np.array([transition[3]])).unsqueeze(0)
        v = []
        tensors = []
        for loc in self.action_descriptors:
            a = self.target_actor.forward(u, to_tensor(np.array([transition[4]])).unsqueeze(0),
                               to_tensor(np.array([loc[0]])).unsqueeze(0), to_tensor(np.array([loc[1]])).unsqueeze(0))
            tensors.append(a)
            # a = a.detach().tolist()[0][0]
            # v.append(a)

        v = torch.cat(tensors, 1)

        # v = to_tensor(np.array(v)).unsqueeze(0)
        w = to_tensor(np.array([transition[3]])).unsqueeze(0)

        x = self.target_critic.forward(w, to_tensor(np.array([transition[4]])).unsqueeze(0), v).detach().numpy()[0][0]

        yi -= self.gamma * x

        return yi

    def update_critic(self, minibatch, target_batch):

        state_batch, action_batch, _, _, sig_batch = [[i[j] for i in minibatch] for j in range(5)]
        sig_batch = np.array(sig_batch)

        self.critic.zero_grad()

        # a = list(zip(state_batch, action_batch))

        # x = to_tensor(np.array(a)).unsqueeze(0)

        x = to_tensor(np.array(state_batch)).unsqueeze(1)
        y = to_tensor(np.array(action_batch))

        output_batch = self.critic(x, to_tensor(sig_batch).unsqueeze(1), y)

        lossfn = nn.MSELoss()

        target_batch = to_tensor(np.array(target_batch))

        loss = lossfn(output_batch, target_batch)
        loss.backward()

        # print(self.critic.conv1.weight.grad)
        self.critic_optim.step()
        return loss

    def update_actor(self, minibatch):
        state_batch, _, _, _, sig_batch = [[i[j] for i in minibatch] for j in range(5)]
        # print(state_batch)
        state_batch = np.array(state_batch)
        sig_batch = np.array(sig_batch)
        self.actor_optim.zero_grad(set_to_none=True)
        # print(self.actor.conv1.weight.grad)

        # u = to_tensor(np.array(state_batch)).unsqueeze(1)
        # v = self.actor(u, to_tensor(np.array(sig_batch)).unsqueeze(1))
        # v = v.detach().tolist()
        # v = np.reshape(v, (-1, self.shape[0], self.shape[1]))
        #
        # input = list(zip(state_batch, v))
        #
        # x = to_tensor(np.array(input), requires_grad=True)
        #
        # policy_grad = -self.critic(x)
        # print(to_tensor(state_batch).squeeze(0))
        # print(self.actor(to_tensor(state_batch), to_tensor(sig_batch)).reshape(-1, 30, 30))

        # print(f'1st shape: {to_tensor(state_batch).shape}')
        # print(f'2nd shape: {self.actor(to_tensor(state_batch).unsqueeze(1), to_tensor(sig_batch).unsqueeze(1)).reshape(-1, 30, 30)}')

        batch_len = len(state_batch)
        tensors = []
        for i, loc in enumerate(self.action_descriptors):
            a = self.actor(to_tensor(state_batch).unsqueeze(1), to_tensor(sig_batch).unsqueeze(1),
                                          to_tensor(np.array([loc[0]]*batch_len)).unsqueeze(1),
                                          to_tensor(np.array([loc[1]]*batch_len)).unsqueeze(1))
            tensors.append(a)
            # print(a.tolist())

            # for ind, elem in enumerate(a.tolist()):
            #     v[ind][i] = elem[0]
            # print(a.detach().tolist())
            # a = a.detach().tolist()[0][0]
            # v.append(a)
        # print(v)
        # print(sig_batch)
        # print(torch.from_numpy(v).shape)
        v = torch.cat(tensors, 1)

        policy_grad = -self.critic(to_tensor(state_batch).unsqueeze(1), to_tensor(sig_batch).unsqueeze(1), v)
        # print(policy_grad)

        policy_grad = policy_grad.mean()
        policy_grad.backward()
        # print(self.actor.conv1.weight.grad)
        self.actor_optim.step()

    def update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
