import numpy as np
import pickle
import matplotlib.pyplot as plt
from ship_data import ShipExperiment

filename = 'ddpg_600kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2'
with open('_experiments/history_' + filename+'.pickle', 'rb') as f:
    hist = pickle.load(f)
f.close()

def _moving_average(a, n=20) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plt.figure()
plt.title('Reward Evolution')
plt.xlabel('Episodes')
plt.ylabel('Reward')
rw = _moving_average(hist['episode_reward'])
plt.plot(rw)

plt.figure()
plt.title('Survival Evolution')
plt.xlabel('Steps in the episode')
plt.ylabel('Episode')
nsteps = _moving_average(hist['nb_episode_steps'])
plt.plot(nsteps)
plt.show()

# Here you can load and plot you performance test
shipExp = ShipExperiment()
experiment_name = '2018-12-02-20experiment_ssn_ddpg_10iter'
shipExp.load_from_experiment(experiment_name)
shipExp.plot_obs(iter=-1) # seleciona os episodios manualmente ou coloque -1 para plotar todos
shipExp.plot_settling_time()
shipExp.plot_actions(iter=9)