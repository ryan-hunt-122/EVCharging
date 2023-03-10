import numpy as np

from critic import Critic
from actor import Actor

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

    def __init__(self, shape, max_buffer=100, minibatch_size=16, gamma=0.01):
        self.shape = shape
        self.max_buffer = max_buffer
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.tau = 0.5
        # Randomly initialize critic network Q with weights q
        self.critic = Critic(self.shape)
        # and target critic network Q' with weights q'
        self.target_critic = Critic(self.shape)
        self.critic_optim = optim.RMSprop(self.critic.parameters())

        # Randomly initialize actor network M with weights m
        self.actor = Actor(self.shape)
        # and target actor network M' with weights m'
        self.target_actor = Actor(self.shape)
        self.actor_optim = optim.RMSprop(self.actor.parameters())

        # Initialize replay buffer R
        self.replay_buffer = []

    def select_action(self, state):
        # Select action according to current actor policy (and exploration noise?)
        # tens = torch.tensor([state])
        # self.target_actor.forward(tens.type(torch.DoubleTensor))
        return self.actor.forward(to_tensor(np.array([state])).unsqueeze(0))


    def store_transition(self, transition):
        a, b, c, d = transition

        transition = (a.tolist(), b, c, d)

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
        v = self.actor.forward(u)
        v = v.detach().numpy()
        v = np.reshape(v, self.shape)
        w = to_tensor(np.array([transition[3], v])).unsqueeze(0)

        x = self.critic.forward(w).detach().numpy()[0][0]

        yi += self.gamma * x

        return yi

    def update_critic(self, minibatch, target_batch):

        state_batch, action_batch, _, _ = [[i[j] for i in minibatch] for j in range(4)]

        self.critic.zero_grad()

        a = list(zip(state_batch, action_batch))

        x = to_tensor(np.array(a)).unsqueeze(0)

        output_batch = self.critic(x[0])

        lossfn = nn.MSELoss()

        target_batch = to_tensor(np.array(target_batch))

        loss = lossfn(output_batch, target_batch)
        loss.backward()
        self.critic_optim.step()

    def update_actor(self, minibatch):
        state_batch, _, _, _ = [[i[j] for i in minibatch] for j in range(4)]

        self.actor.zero_grad()

        u = to_tensor(np.array(state_batch)).unsqueeze(1)
        v = self.actor(u)
        v = v.detach().tolist()
        v = np.reshape(v, (-1, self.shape[0], self.shape[1]))

        input = list(zip(state_batch, v))

        x = to_tensor(np.array(input))

        policy_grad = -self.critic(x)

        policy_grad = policy_grad.mean()
        policy_grad.backward()
        self.actor_optim.step()

    def update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.tau) + param.data*self.tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.tau) + param.data*self.tau)














