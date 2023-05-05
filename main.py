import os
import pickle

import numpy as np
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from evcharging import EVCharging
from ddpg.ddpg import DDPG
import matplotlib.pyplot as plt
from pathlib import Path




def train():
    env = EVCharging()

    model = PPO(MlpPolicy, env, verbose=2)
    model.learn(total_timesteps=10000)
    env.close()
    return model

def train_ddpg(lr, mult):
    shape = (10, 10)

    # Randomly initialize networks and replay buffer
    # Shape of grid is shape = (30, 30)
    algo = DDPG(shape, minibatch_size=16, lr=0.0005, mult=10.0)

    # Initialize random process for action exploration
    env = EVCharging(10, 100, shape=shape)
    ep_rewards = []
    losses = []
    # Receive initial observation state
    for episode in range(20000):
        state_list = []
        # print(f'Episode {episode}:')
        s_1 = env.reset()
        s_t = s_1
        reward = 0
        done = False
        # For t=1,T do
        while not done:
            state_list.append((env.vehicles, env.get_signal()))
            # if env.t % 50 == 0:
            #     print(f'Episode {episode} at step {env.t} with accumulated reward {reward}')
            sig = env.get_signal()
            # Select action according to current policy
            a_t = algo.select_action(s_t, sig)

            # Execute action and observe reward and new state
            s_t1, r_t, done, info = env.step(a_t)
            reward += r_t
            # print(f'Signal {sig}, Reward {r_t}')
            # Store transition in R
            algo.store_transition((s_t, a_t, r_t, s_t1, sig))
            # Sample a random minibatch from R
            minibatch = algo.sample_minibatch()
            # Set yi for all i
            ys = [[algo.yi(tr)] for tr in minibatch]
            # Update critic by minimizing loss
            loss = algo.update_critic(minibatch, ys)
            # losses.append(loss.item())
            # fig1 = plt.figure()
            # plt.plot(range(len(losses)), losses)
            # fig1.savefig('train_loss.png')
            # plt.close()
            algo.update_actor(minibatch)

            algo.update_target_networks()
            s_t = s_t1
            # print(loss)
        print(f'Episode {episode} finished in {env.t} steps, with episode reward {reward}')
        ep_rewards.append(reward)
        fig = plt.figure()
        plt.plot(range(len(ep_rewards)), ep_rewards)
        fig.savefig(f'results/rewards_{lr}_{mult}.png')
        plt.close()

        with open(f'pickles/{lr}_{mult}/ep{episode}.pkl', 'wb') as fp:
            pickle.dump(state_list, fp)


if __name__ == '__main__':
    plt.style.use('dark_background')
    # lrs = [0.000001, 0.000005, 0.00001, 0.0001]
    # mults = [1.0, 5.0, 10.0, 20.0]
    # for lr in lrs:
    #     for mult in mults:
    train_ddpg(0.0001, 10.0)

    # env = EVCharging(300, 30)
    # obs = env.reset()
    # env.render()
    # treward = 0
    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    #     treward += reward
    # print(env.vehicles)
    # print(treward)

    # model = train()
    # env = EVCharging()
    # needed = []
    # signal = []
    # rew = 0
    # obs = env.reset()
    # for i in range(1):
    #     action = env.action_space.sample()
    #     # action = model.predict(obs)[0]
    #     obs, reward, done, info = env.step(action)
    #     needed.append(info['needed'])
    #     signal.append(info['signal'])
    #     rew += reward
    #     env.render()
    # print("Episode Reward: ", rew)
    # env.close()
    # fig = go.Figure(
    #     data=[go.Line(x=list(range(0,200)),y=needed), go.Line(x=list(range(0,200)),y=signal)],
    #     layout_title_text="Signal"
    # )
    # fig.show()

