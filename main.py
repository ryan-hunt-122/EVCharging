import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from evcharging import EVCharging
from reference import get_state_reference
import plotly.graph_objects as go
from ddpg import DDPG
import matplotlib.pyplot as plt




def train():
    env = EVCharging()

    model = PPO(MlpPolicy, env, verbose=2)
    model.learn(total_timesteps=10000)
    env.close()
    return model

def train_ddpg():
    shape = (30, 30)

    # Randomly initialize networks and replay buffer
    # Shape of grid is shape = (30, 30)
    algo = DDPG(shape, minibatch_size=16)

    # Initialize random process for action exploration
    env = EVCharging(300, 30)
    ep_rewards = []
    # Receive initial observation state
    for episode in range(100):
        # print(f'Episode {episode}:')
        s_1 = env.reset()
        s_t = s_1
        reward = 0
        done = 0
        # For t=1,T do
        # while not done:
        for i in range(100):
            if env.t % 50 == 0:
                print(f'Episode {episode} at step {env.t} with accumulated reward {reward}')
            sig = env.get_signal()
            # Select action according to current policy
            a_t = algo.select_action(s_t, sig)
            a_t = a_t.detach().tolist()
            a_t = np.reshape(a_t, shape).tolist()
            # print(a_t)

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
            algo.update_critic(minibatch, ys)
            algo.update_actor(minibatch)

            algo.update_target_networks()
            s_t = s_t1
        print(f'Episode {episode} finished in {env.t} steps, with episode reward {reward}')
        ep_rewards.append(reward)
        plt.plot(range(len(ep_rewards)), ep_rewards)
        plt.savefig('train_loss.png')


if __name__ == '__main__':
    plt.style.use('dark_background')
    train_ddpg()

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

