import gym
from gym import spaces
import numpy as np
from numpy import random
from PIL import Image

class EVCharging(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, shape):
        super(EVCharging, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.h, self.w = shape[0], shape[1]
        self.timesteps = 300
        self.box_size = 10
        self.N_BOXES = int(((self.timesteps / self.box_size) ** 2) / 2 + (self.timesteps / self.box_size) / 2)
        self.N_VEHICLES = 100
        self.vehicles = self.gen_vehicles()
        self.signal = 0.0
        self.t = 0

        self.columns = int(self.timesteps / self.box_size)
        self.action_space = spaces.Box(low=0, high=1, shape=(int((self.columns-2)*(self.columns-1)/2),))
        self.observation_space = spaces.Box(low=0, high=255, shape=(2, self.N_VEHICLES))

    def get_box_id(self, td, ts):
        td = int(td / self.box_size)
        ts = int(ts / self.box_size)
        # Find location of box containing (td, ts)
        # ID calculated with (x-2)(x-1)/2 + (y+1)
        return int(((td-2)*(td-1)/2) + (ts+1))

    def gen_vehicles(self):
        # Generate vehicles for use in this episode
        vehicles = []
        for i in range(self.N_VEHICLES):
            # Gaussian for y
            y = random.normal(loc=0.5, scale=0.125)
            # Uniform for x
            x = random.randint(self.timesteps)
            vehicles.append([x, x*y])
        # Return list of tuples [(td, ts)] for all vehicles
        # eg. vehicles = [[100, 80], [30, 13], [20, 15], [10, 3]]
        return vehicles

    def get_needed(self):
        needed = 0.0
        for veh in self.vehicles:
            needed += veh[1] / veh[0] if veh[0] != 0 else 0

        return needed

    def get_signal(self):
        # Arbitrary signal randomiser function
        needed = self.get_needed()
        if needed == 0:
            return 0

        if self.signal > needed:
            diff = max(50.0, ((self.signal/needed)-1.2)*100)
            downprob = 50+diff
        else:
            diff = max(50.0, ((needed/self.signal)-1.2)*100)
            downprob = 50-diff

        y = random.randint(100)
        if y > downprob:
            self.signal += 0.005*needed
        else:
            self.signal -= 0.005*needed
        return self.signal

    def get_histogram(self):
        histogram = np.zeros((self.timesteps // self.box_size, self.timesteps // self.box_size))

        for td, ts in self.vehicles:
            histogram[int(td / self.box_size), int(ts / self.box_size)] += 1

        histogram = histogram / self.N_VEHICLES

        return histogram


        # histogram = [0 for i in range(self.N_BOXES)]
        # for td, ts in self.vehicles:
        #     histogram[self.get_box_id(td, ts)] += 1
        # histogram = [i // self.N_VEHICLES for i in histogram]
        # return histogram


    def step(self, action):
        total_charge = 0
        # Update td and ts according to time and action
        for i in range(len(self.vehicles)):
            # Update td and ts for each vehicle
            veh = self.vehicles[i]
            # box_id = self.get_box_id(veh[0], veh[1]) # Get the id of the box it is in
            dtd = 1 if veh[0] > 0 else 0
            # If ts <= 0 then no charge. If ts in box touching y=x then 1 charge. Else action[box_id]
            try:
                dts = 0 if (veh[1]/self.box_size) <= 0 else (1 if int((veh[1]/self.box_size)+1) >= int(veh[0]/self.box_size) else action[int(veh[0]/self.box_size)][int(veh[1]/self.box_size)])
            except:
                print(veh)
            self.vehicles[i][0] -= dtd
            self.vehicles[i][1] -= dts
            total_charge += dts

        # observation = ([i[0] for i in self.vehicles], [i[1] for i in self.vehicles])
        observation = self.get_histogram()
        reward = - abs(self.get_signal() - total_charge)
        done = True if all(i[0]<1 and i[1]<0 for i in self.vehicles) else False
        info = {'needed': self.get_needed(), 'signal':self.get_signal()}
        return observation, reward, done, info

    def reset(self):
        self.vehicles = self.gen_vehicles()
        ts = [i[0] for i in self.vehicles]
        td = [i[1] for i in self.vehicles]
        self.signal = self.get_needed()

        # observation = ([i[0] for i in self.vehicles], [i[1] for i in self.vehicles])
        observation = self.get_histogram()
        return observation  # reward, done, info can't be included

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