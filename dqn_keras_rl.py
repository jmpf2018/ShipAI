import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from ship_env import ShipEnv
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #comment this line if you want to use cuda

# Get the environment and extract the number of actions.
env = ShipEnv(type='discrete', action_dim=2)
np.random.seed(551)
env.seed(551)


nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=2000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10000,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Define 'test' for testing an existing network weights or 'train' to train a new one!
mode = 'test'

if mode == 'train':
    filename = '400kit_rn4_maior2_mem20k_20acleme_target1000_epsgr1'
    hist = dqn.fit(env, nb_steps=300000, visualize=False, verbose=2)
    with open('C:/Users/JMPF/PycharmProjects/ShipAI/ShipAI/_experiments/history_dqn_test_'+ filename + '.pickle', 'wb') as handle:
        pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # After training is done, we save the final weights.
    dqn.save_weights('h5f_files/dqn_{}_weights.h5f'.format(filename), overwrite=True)
    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=10, visualize=True)

if mode == 'test':
    env.set_test_performace()  # Define the initialization as performance test
    env.set_save_experice()  # Save the test to plot the results after
    filename = '400kit_rn4_maior2_mem20k_20acleme_target1000_epsgr1'
    dqn.load_weights('h5f_files/dqn_{}_weights.h5f'.format(filename))
    dqn.test(env, nb_episodes=10, visualize=True)