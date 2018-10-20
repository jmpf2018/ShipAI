#!/usr/bin/python
#-*- coding: utf-8 -*-

from gym import Env, spaces
import numpy as np
from shapely.geometry import LineString, Point
from viewer import Viewer
from ship_data import ShipExperiment
from simulator import Simulator


class ShipEnv:
    def __init__(self):
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]))
        self.obs_space = spaces.Box(low=np.array([0, -np.pi/2, 0, -4, -1.0]), high=np.array([150, np.pi/2, 4.0, 4.0, 1.0]))
        self.init_space = spaces.Box(low=np.array([0, -np.pi/15, 2.0, 0.2,  -0.1]), high=np.array([30, np.pi/15, 3.0, 0.3, 0.1]))
        self.ship_data = ShipExperiment()
        self.last_pos = np.zeros(3) # last_pos = [xg yg thg]
        self.last_action = np.zeros(1) #only one action
        self.simulator = Simulator()
        self.point_a = (0.0, 0.0)
        self.point_b = (2000, 0.0)
        self.guideline = LineString([self.point_a, self.point_b])
        self.start_pos = np.zeros(1)
        self.borders = np.array([[0, 150], [2000, 150]])
        self.viewer = None

    def step(self, action):
        action = action * np.sign(self.last_pos[1])
        rot_action = 0.1
        state_prime = self.simulator.step(angle_level=action[0], rot_level=rot_action)
        # transforma variáveis do simulador em variáveis observáveis
        obs = self.convert_state(state_prime)
        # print('Observed state: ', obs)
        dn = self.end(state_prime=state_prime, obs=obs)
        rew = self.calculate_reward(obs=obs)
        self.last_pos = [state_prime[0], state_prime[1], state_prime[2]]
        self.last_action = action
        self.ship_data.new_transition(state_prime, obs, action, rew)
        info = dict()
        return obs, rew, dn, info

    def convert_state(self, state):
        """
        This method generated the features used to build the reward function
        :param state: Global state of the ship
        """
        ship_point = Point((state[0], state[1]))
        side = np.sign(state[1] - self.point_a[1])
        d = ship_point.distance(self.guideline)  # meters
        theta = side*state[2]  # radians
        vx = state[3]  # m/s
        vy = side*state[4]  # m/s
        thetadot = side * state[5]  # graus/min
        obs = np.array([d, theta, vx, vy, thetadot])
        return obs

    def calculate_reward(self, obs):
        d, theta, vx, vy, thetadot = obs[0], obs[1]*180/np.pi, obs[2], obs[3], obs[4]*180/np.pi
        print("\n Action: %f,  State[%f %f %f], Velocidade [%f , %f] , Theta: %f, Distance: %f, thetadot: %f \n" % (self.last_action, self.last_pos[0], self.last_pos[1], self.last_pos[2], vx, vy, theta, d, thetadot))
        if self.last_pos[0] > 2000:
            print("\n Got there")
            return 1000
        elif d > 75:
            return vx * 1 - d / 150 - vy * np.sqrt(d) * 0.05 - 100 * thetadot ** 2 - 1000 / np.abs(150.001 - d) - (
                        theta / 50) ** 2
        else:
            return vx * 1 - d / 150 - vy * np.sqrt(d) * 0.05 - 100 * thetadot ** 2 - 10 / np.abs(150.001 - d) - (
                        theta / 50) ** 2

    def end(self, state_prime, obs):
        if not self.obs_space.contains(obs) or -1 > state_prime[0] or state_prime[0] > 2000 or 160 < state_prime[1] or state_prime[1]< -160:
            self.viewer.end_episode()
            return True
        else:
            return False

    def set_init_space(self, low, high):
        self.init_space = spaces.Box(low=np.array(low), high=np.array(high))

    def reset(self):
        init = list(map(float, self.init_space.sample()))
        self.simulator.reset_start_pos(np.array([self.start_pos[0], init[0],  init[1], init[2]*np.cos(init[1]), init[2]*np.sin(init[1]), 0]))
        self.last_pos = np.array([self.start_pos[0], init[0],  init[1]])
        print('Reseting position')
        state = self.simulator.get_state()
        if self.ship_data.iterations > 0:
            self.ship_data.save_experiment('_experiment_rnd')
        self.ship_data.new_iter(state, self.convert_state(state), np.array([0]), np.array([0]))
        return self.convert_state(state)

    def render(self, ):
        if self.viewer is None:
            self.viewer = Viewer()
            self.viewer.plot_guidance_line(self.point_a, self.point_b)
        self.viewer.plot_position(self.last_pos[0], self.last_pos[1],  self.last_pos[2],  self.last_action[0])

    def close(self, ):
        self.viewer.freeze_scream()


if __name__ == '__main__':
    mode = 'normal'
    if mode == 'normal':
        env = ShipEnv()
        shipExp = ShipExperiment()
        for i_episode in range(2):
            observation = env.reset()
            for t in range(10000):
                env.render()
                action = np.array([-0.1])
                observation, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        env.close()

