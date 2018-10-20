import numpy as np
import matplotlib.pyplot as plt
import pickle
import time as t
import datetime
import scipy.io as io
from numpy.core.records import fromarrays


class ShipExperiment:
    def __init__(self, info=None):
        """
        Initialize function to save data
        :param info: Char - Information about the iteration tets
        """
        self.iterations = -1
        self.states = {}
        self.observations = {}
        self.actions = {}
        self.rewards = {}
        self.steps = {}
        self.info = info
        self.viewer = None
        self.scream = None

    def new_iter(self, s0, obs0, a0, r0):
        """
        A new interaction create a new sublist of states, actions and rewards, it increase the interaction count and
        initialize the steps count
        :param s0: numpy array of state 0 ie: [Xabs Yabs Thetaabs Vxabs Vyabs Thetadotabs]_0
        :param obs0: numpy array of the observation states ie: [d Vlon Theta Thetadot]_0
        :param a0: numpy array of actions ie: [Angle Propulsion]_0
        :param r0:  numpy array of reaward ie : [R]_0
        """
        self.iterations += 1
        it = self.iterations
        self.steps[it] = 0
        self.states[it] = s0
        self.observations[it] = obs0
        self.actions[it] = a0
        self.rewards[it] = r0

    def new_transition(self, s, obs, a, r):
        """
        Each transition pass a set of numpy  arrays to be saved
        :param s: numpy array of state 0 ie: [Xabs Yabs Thetaabs Vxabs Vyabs Thetadotabs]
        :param obs: numpy array of state 0 ie: [Xabs Yabs Thetaabs Vxabs Vyabs Thetadotabs]
        :param a: numpy array of actions ie: [Angle Propulsion]_0
        :param r: numpy array of reaward ie : [R]_0
        """
        it = self.iterations
        self.steps[it] += 1
        self.states[it] = np.vstack([self.states[it], s])
        self.observations[it] = np.vstack([self.observations[it], obs])
        self.actions[it] = np.vstack([self.actions[it], a])
        self.rewards[it] = np.vstack([self.rewards[it], r])

    def save_2text(self):
        """
        Use this method to save the vector of iteration in a Matlab format
        """
        st = datetime.datetime.fromtimestamp(t.time()).strftime('%Y%m%d%H')
        name = st+'matlab.mat'
        io.savemat(name, {'states': list(self.states.values()), 'actions': list(self.actions.values()), 'obs': list(self.observations.values())})

    def save_experiment(self, descr='_experiment'):
        """
        Use this method save an experiment in .pickle format
        """
        st = datetime.datetime.fromtimestamp(t.time()).strftime('%Y-%m-%d-%H')
        name = st+descr
        with open('../_experiments/'+name, 'wb') as f:
            pickle.dump(self.__dict__, f, 2)
        f.close()

    def load_from_experiment(self, name):
        """
        Use this method save an experiment in .pickle format
        :param name:
        """
        with open('_experiments/' + name, 'rb') as f:
            tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)

    def plot_actions(self, iter=0):
        """
        Plot actions of an iteration
        :param iter: iteration index
        """
        if iter == -1:
            f, axarr = plt.subplots(len(self.actions[0][0, :]), sharex=True)
            for j in range(self.iterations):
                for i in range(len(self.actions[0][0, :])):
                    axarr[i].plot(np.arange(0, self.steps[j] + 1, 1), self.actions[j][:, i])
                    axarr[i].set_title("Action:" + str(i))
                    axarr[i].set_ylabel('Actions')
                    axarr[i].set_xlabel('Steps')
            plt.show()
        else:
            f, axarr = plt.subplots(len(self.actions[iter][0, :]), sharex=True)
            for i in range(len(self.actions[iter][0, :])):
                axarr[i].plot(np.arange(0, self.steps[iter]+1, 1), self.actions[iter][:, i])
                axarr[i].set_title("Action:"+ str(i))
                axarr[i].ylabel('Actions')
                axarr[i].xlabel('Steps')
            plt.show()

    def plot_reward(self, iter=0):
        """
        Plot reward of an iteration
        :param iter: iteration index
        """
        if iter == -1:
            for i in range(self.iterations):
                plt.plot(np.arange(0, self.steps[i] + 1, 1), self.rewards[i])
                plt.ylabel('Reward')
                plt.xlabel('steps')
                plt.show()
        else:
            plt.plot(np.arange(0, self.steps[iter]+1, 1), self.rewards[iter])
            plt.ylabel('Reward')
            plt.xlabel('steps')
            plt.show()

    def plot_ship(self):
        if self.viewer is None:
            plt.ion()
            self.scream, self.viewer = plt.subplots(3, sharex=True)
            for i in range(3):
                self.viewer[i].set_title("Observed states")
                self.viewer[i].set_ylabel("Obs"+str(i))
                self.viewer[i].set_xlabel('Steps')
            plt.show()
        else:
            j = self.iterations
            if self.steps[j]//10 > 1:
                for i in range(3):
                    self.viewer[i].cla()
                    self.viewer[i].plot(np.arange(0, self.steps[j] + 1, 1), self.observations[j][:, i])
                    plt.pause(0.0001)

    def plot_obs(self, iter=0):
        img, ax = plt.subplots(3, sharex=True)
        for i in range(3):
            ax[i].set_title("Observed states")
            ax[i].set_ylabel("Obs" + str(i))
            ax[i].set_xlabel('Steps')
            ax[i].plot(np.arange(0, self.steps[iter] + 1, 1), self.observations[iter][:, i])
        plt.show()