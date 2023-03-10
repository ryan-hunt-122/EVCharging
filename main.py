import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from evcharging import EVCharging

import plotly.graph_objects as go
from ddpg import DDPG




def train():
    env = EVCharging()

    model = PPO(MlpPolicy, env, verbose=2)
    model.learn(total_timesteps=10000)
    env.close()
    return model

def train_ddpg():
    shape = (30, 30)

    # Randomly initialize networks and replay buffer
    # Shape of grid is shape = (20, 20)
    algo = DDPG(shape)

    # Initialize random process for action exploration
    env = EVCharging(shape)
    # Receive initial observation state
    s_1 = env.reset()
    env.render()
    print(s_1)
    s_t = s_1
    reward = 0

    # For t=1,T do
    for i in range(100):
        # Select action according to current policy
        a_t = algo.select_action(s_t)
        a_t = a_t.detach().tolist()
        a_t = np.reshape(a_t, shape).tolist()

        # Execute action and observe reward and new state
        s_t1, r_t, done, info = env.step(a_t)
        reward += r_t
        # Store transition in R
        algo.store_transition((s_t, a_t, r_t, s_t1))
        # Sample a random minibatch from R
        minibatch = algo.sample_minibatch()
        # Set yi for all i
        # for transition in minibatch:
        ys = [[algo.yi(tr)] for tr in minibatch]
        # Update critic by minimizing loss
        algo.update_critic(minibatch, ys)
        algo.update_actor(minibatch)

        algo.update_target_networks()
        s_t = s_t1
    print(reward)

if __name__ == '__main__':
    train_ddpg()


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

