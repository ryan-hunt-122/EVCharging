import gym
from gym import spaces
import numpy as np
from numpy import random
from PIL import Image
from reference import get_state_reference
import random

class EVCharging(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, timesteps, n_vehicles, shape=(30, 30)):
        super(EVCharging, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.h, self.w = shape[0], shape[1]
        self.timesteps = timesteps
        self.N_VEHICLES = n_vehicles

        self.box_size = (300 // self.h)

        self.N_BOXES = int(((self.timesteps / self.box_size) ** 2) / 2 + (self.timesteps / self.box_size) / 2)

        self.init_vehicles, self.reference_signal = get_state_reference(self.timesteps, self.N_VEHICLES)

        self.vehicles = self.init_vehicles.copy()
        self.t = 0

        self.columns = int(self.timesteps / self.box_size)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.h, self.w))
        self.observation_space = spaces.Box(low=0, high=255, shape=(2, self.N_VEHICLES))
        print(self.vehicles)

    def get_box_id(self, td, ts):
        td = int(td / self.box_size)
        ts = int(ts / self.box_size)
        # Find location of box containing (td, ts)
        # ID calculated with (x-2)(x-1)/2 + (y+1)
        return int(((td-2)*(td-1)/2) + (ts+1))

    def get_signal(self):
        return self.reference_signal[self.t]

    def get_histogram(self):
        histogram = np.zeros((self.timesteps // self.box_size, self.timesteps // self.box_size))

        for ind, (td, ts) in self.vehicles.iterrows():
            histogram[int(td / self.box_size), int(ts / self.box_size)] += 1

        histogram = histogram / self.N_VEHICLES

        return histogram



    def step(self, action):

        def get_dts(ts, td):
            return 0 if (ts / self.box_size) <= 0 else (
                1 if int((ts / self.box_size) + 1) >= int(td / self.box_size) else action[int(td / self.box_size)][
                    int(ts / self.box_size)])

        def get_dtd(td):
            return 1 if td > 0 else 0

        total_charge = 0
        # Update td and ts according to time and action
        self.vehicles['dt_d'] = self.vehicles.apply(lambda x: get_dtd(x['t_d']), axis=1)
        self.vehicles['dt_s'] = self.vehicles.apply(lambda x: get_dts(x['t_s'], x['t_d']), axis=1)

        self.vehicles['t_d'] -= self.vehicles['dt_d']
        self.vehicles['t_s'] -= self.vehicles['dt_s']

        total_charge = self.vehicles['dt_s'].sum()

        self.vehicles = self.vehicles.drop(['dt_d', 'dt_s'], axis=1)

        # observation = ([i[0] for i in self.vehicles], [i[1] for i in self.vehicles])
        observation = self.get_histogram()
        reward = - abs(self.get_signal() - total_charge)
        done = (self.vehicles['t_s'] <= 0).all()
        info = {'signal':self.get_signal()}
        self.t += 1
        return observation, reward, done, info

    def reset(self):
        self.vehicles = self.init_vehicles.copy()
        self.t = 0

        # observation = ([i[0] for i in self.vehicles], [i[1] for i in self.vehicles])
        observation = self.get_histogram()
        return observation

    def render(self, mode='human'):
        histogram = self.get_histogram()

        histogram1 = 256 - (histogram * 256)
        im = Image.fromarray(histogram1)
        im = im.convert("L")
        im.save('render.png')

        histogram2 = (histogram * self.N_VEHICLES)
        histogram2 = histogram2 / np.amax(histogram2)
        histogram2 = 256 - (histogram2*256)
        im2 = Image.fromarray(histogram2)
        im2 = im2.convert("L")
        im2.save('render_human.png')



    def close(self):
        ...