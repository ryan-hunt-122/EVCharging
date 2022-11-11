
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from evcharging import EVCharging

import plotly.graph_objects as go





def train():
    env = EVCharging()

    model = PPO(MlpPolicy, env, verbose=2)
    model.learn(total_timesteps=10000)
    env.close()
    return model

if __name__ == '__main__':
    # model = train()
    env = EVCharging()
    needed = []
    signal = []
    rew = 0
    obs = env.reset()
    for i in range(500):
        action = env.action_space.sample()
        # action = model.predict(obs)[0]
        obs, reward, done, info = env.step(action)
        needed.append(info['needed'])
        signal.append(info['signal'])
        rew += reward
        env.render()
    print("Episode Reward: ", rew)
    env.close()
    # fig = go.Figure(
    #     data=[go.Line(x=list(range(0,200)),y=needed), go.Line(x=list(range(0,200)),y=signal)],
    #     layout_title_text="Signal"
    # )
    # fig.show()

