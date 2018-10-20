import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from ship_env import ShipEnv


# Get the environment and extract the number of actions.
env = ShipEnv()
np.random.seed(666)
#env.seed(666)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(300))
actor.add(Activation('relu'))
actor.add(Dense(600))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(600)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.6, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=0.001,  clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
#agent.load_weights('ddpg_{}_weights.h5f'.format('ship_env_v9'))
agent.fit(env, nb_steps=30000, visualize=True, verbose=1, nb_max_episode_steps=1000)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format('ship_env_v9'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
#agent.test(env, nb_episodes=1, visualize=True, nb_max_episode_steps=1000)