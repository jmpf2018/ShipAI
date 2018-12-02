#!/usr/bin/python
#-*- coding: utf-8 -*-

from gym import Env, spaces
import numpy as np
from shapely.geometry import LineString, Point
from viewer import Viewer
from ship_data import ShipExperiment
from simulator import Simulator


class ShipEnv(Env):
    def __init__(self, type='continuous', action_dim = 2):
        self.type = type
        self.action_dim = action_dim
        assert type == 'continuous' or type == 'discrete', 'type must be continuous or discrete'
        assert action_dim > 0 and action_dim <=2, 'action_dim must be 1 or 2'
        if type == 'continuous':
            self.action_space = spaces.Box(low=np.array([-1.0, 0]), high=np.array([1.0, 0.2]))
            self.observation_space = spaces.Box(low=np.array([0, -np.pi / 2, 0, -4, -0.2]), high=np.array([150, np.pi / 2, 4.0, 4.0, 0.2]))
            self.init_space = spaces.Box(low=np.array([0, -np.pi / 15, 1.0, 0.2, -0.01]), high=np.array([30, np.pi / 15, 1.5, 0.3, 0.01]))
        elif type == 'discrete':
            if action_dim == 2:
                self.action_space = spaces.Discrete(63)
            else:
                self.action_space = spaces.Discrete(21)
            self.observation_space = spaces.Box(low=np.array([0, -np.pi / 2, 0, -4, -0.4]), high=np.array([150, np.pi / 2, 4.0, 4.0, 0.4]))
            self.init_space = spaces.Box(low=np.array([0, -np.pi / 15, 1.0, 0.2, -0.01]), high=np.array([30, np.pi / 15, 2.0, 0.3, 0.01]))
        self.ship_data = None
        self.name_experiment = None
        self.last_pos = np.zeros(3)
        self.last_action = np.zeros(self.action_dim)
        self.simulator = Simulator()
        self.point_a = (0.0, 0.0)
        self.point_b = (2000, 0.0)
        self.max_x_episode = (5000, 0)
        self.guideline = LineString([self.point_a, self.max_x_episode])
        self.start_pos = np.zeros(1)
        self.number_loop = 0  # loops in the screen -> used to plot
        self.borders = [[0, 150], [2000, 150], [2000, -150], [0, -150]]
        self.viewer = None
        self.test_performance = False
        self.init_test_performance = np.linspace(0, np.pi / 15, 10)
        self.counter = 0

    def step(self, action):
        # According to the action stace a different kind of action is selected
        if self.type == 'continuous' and self.action_dim == 2:
            side = np.sign(self.last_pos[1])
            angle_action = action[0]*side
            rot_action = (action[1]+1)/10
        elif self.type == 'continuous' and self.action_dim == 1:
            side = np.sign(self.last_pos[1])
            angle_action = action[0]*side
            rot_action = 0.2
        elif self.type == 'discrete' and self.action_dim == 2:
            side = np.sign(self.last_pos[1])
            angle_action = (action // 3 - 10) / 10
            angle_action = angle_action * side
            rot_action = 0.1 * (action % 3)
        elif self.type == 'discrete' and self.action_dim == 1:
            side = np.sign(self.last_pos[1])
            angle_action = (action - 10) / 10
            angle_action = angle_action * side
            rot_action = 0.2
        state_prime = self.simulator.step(angle_level=angle_action, rot_level=rot_action)
        # convert simulator states into obervable states
        obs = self.convert_state(state_prime)
        # print('Observed state: ', obs)
        dn = self.end(state_prime=state_prime, obs=obs)
        rew = self.calculate_reward(obs=obs)
        self.last_pos = [state_prime[0], state_prime[1], state_prime[2]]
        self.last_action = np.array([angle_action, rot_action])
        if self.ship_data is not None:
            self.ship_data.new_transition(state_prime, obs, self.last_action, rew)
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
        if self.last_pos[0] > 5000:
            print("\n Got there")
        if not self.observation_space.contains(obs):
            return -1000
        else:
            return (4*(vx-1.5) + 5*(1-d/20) + 2*(1-vy**2/10) + 5*(1-np.abs(theta/30)) + 3*(1 - np.abs(thetadot)/12)) / 24

    def end(self, state_prime, obs):
        if not self.observation_space.contains(obs) or -1 > state_prime[0] or state_prime[0] > self.max_x_episode[0] or 160 < state_prime[1] or state_prime[1]< -160:
            if not self.observation_space.contains(obs):
                print("\n Smashed")
            if self.viewer is not None:
                self.viewer.end_episode()
            if self.ship_data is not None:
                if self.ship_data.iterations > 0:
                    self.ship_data.save_experiment(self.name_experiment)
            return True
        else:
            return False

    def set_init_space(self, low, high):
        self.init_space = spaces.Box(low=np.array(low), high=np.array(high))

    def reset(self):
        init = list(map(float, self.init_space.sample()))
        if self.test_performance:
            angle = self.init_test_performance[self.counter]
            v_lon = 1.5
            init_states = np.array([self.start_pos[0], 30, angle, v_lon * np.cos(angle), v_lon * np.sin(angle), 0])
            self.counter += 1
            init[0] = 30
            init[1] = angle
        else:
            init_states = np.array([self.start_pos[0], init[0], init[1], init[2] * np.cos(init[1]), init[2] * np.sin(init[1]), 0])
        self.simulator.reset_start_pos(init_states)
        self.last_pos = np.array([self.start_pos[0], init[0],  init[1]])
        print('Reseting position')
        state = self.simulator.get_state()
        if self.ship_data is not None:
            if self.ship_data.iterations > 0:
                self.ship_data.save_experiment(self.name_experiment)
            self.ship_data.new_iter(state, self.convert_state(state), np.zeros(len(self.last_action)), np.array([0]))
        if self.viewer is not None:
            self.viewer.end_episode()
        return self.convert_state(state)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = Viewer()
            self.viewer.plot_boundary(self.borders)
            self.viewer.plot_guidance_line(self.point_a, self.point_b)

        img_x_pos = self.last_pos[0] - self.point_b[0] * (self.last_pos[0] // self.point_b[0])
        if self.last_pos[0]//self.point_b[0] > self.number_loop:
            self.viewer.end_episode()
            self.viewer.plot_position(img_x_pos, self.last_pos[1], self.last_pos[2], 20 * self.last_action[0])
            self.viewer.restart_plot()
            self.number_loop += 1
        else:
            self.viewer.plot_position(img_x_pos, self.last_pos[1], self.last_pos[2], 20 * self.last_action[0])

    def close(self, ):
        self.viewer.freeze_scream()

    def set_save_experice(self, name='experiment_ssn_ddpg_10iter'):
        assert type(name) == type(""), 'name must be a string'
        self.ship_data = ShipExperiment()
        self.name_experiment = name

    def set_test_performace(self):
        self.test_performance = True

if __name__ == '__main__':
    mode = 'normal'
    if mode == 'normal':
        env = ShipEnv()
        shipExp = ShipExperiment()
        for i_episode in range(10):
            observation = env.reset()
            for t in range(10000):
                env.render()
                action = np.array([1, 2])
                observation, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        env.close()