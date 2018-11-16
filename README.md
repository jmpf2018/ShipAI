# SHIP AI
Ship AI is a reinforcement learning module based on the OpenAI Gym and Keras-RL projects.
This module presents a simulation environment for navigating a ship in a channel (Restricted Waters). The purpose of the navigation task is to keep the ship in the midline of the channel. With ShipAI you can:

  - Design Reinforced Learning (RL) algorithms to control ship maneuver movements.
  - Visualize the Ship movement Behavior over a channel.
  - Train, test and validate your RL agent. 

## States
The main purpose of the navigation task was defined as follows:
> Use the rudder and propulsion controls to perform a defined linear navigation path along a channel

The states chosen for the application of RL in the task of the ship were as follows:

![latex](https://latex.codecogs.com/gif.latex?s&space;=&space;(d,\theta,v_x,v_y,\dot{\theta}))

Where  ![latex](https://latex.codecogs.com/gif.latex?d) is the distance from the center of mass of the ship to the guideline; $\theta$ is the angle between the longitudinal axis of the ship and the guideline;  ![latex](https://latex.codecogs.com/gif.latex?v_x) is the horizontal speed of the ship in its center of mass (in the direction of the guideline; ![latex](https://latex.codecogs.com/gif.latex?v_y ) is the vertical speed of the ship in its center of mass (perpendicular to the guideline); ![latex](https://latex.codecogs.com/gif.latex?\dot{\theta}) is the angular velocity of the ship.

![Imgur](https://i.imgur.com/E4MtN4O.png)

 
## Reward
The reward function is defined as:

![Imgur](https://i.imgur.com/gikYyOm.gif)

Where:

![Imgur](https://i.imgur.com/lmf05VS.png)

## Actions
The action are the input parameters for controlling the ship maneuver movement. The forces that make the ship controllable are the rudder and propulsion forces. They have the vector form of    ![latex](https://latex.codecogs.com/gif.latex?A_V&space;=&space;[A_l,&space;A_p]) , where ![latex](https://latex.codecogs.com/gif.latex?A_l) is the dimensionless rudder command and ![latex](https://latex.codecogs.com/gif.latex?A_p)  the dimensionless propulsion command, such that ![latex](https://latex.codecogs.com/gif.latex?A_l&space;\in&space;[-1,&space;1&space;]) and ![latex](https://latex.codecogs.com/gif.latex?A_p&space;\in&space;[0,&space;1]). These parameters have a direct proportional relation with the rudder angle and the propulsion

## How to install ShipAI?
1. Make sure you have all necessary modules installed:
- `pip install TensorFlow`
- `pip install Keras`
- `pip install gym`
- `pip install keras-rl`
- `pip install Shapely`
- `pip install numpy`
- `pip install scipy`
2. Clone or download the repository:
- `git clone https://github.com/jmpf2018/ShipAI`
- `cd ShipAI`
3. Run DQN or DDPG RL algorithms:
- `python dqn_keras_rl.py`
or
- `python ddpg_keras_rl.py`

4.After that RL will start to run:
![Imgur](https://i.imgur.com/2RMsLvn.png)

5.Finally in the end of the process the vizualization screen you start:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/a7V2EouMkcE/0.jpg)](https://www.youtube.com/watch?v=a7V2EouMkcE)

## Final Considerations

- You can change the ship dynamics and adapt for your needs in: simulator.py

- You can change the reward and state design in: ship_env.py (DDPG) and ship_env_discrete (DQN).

- You can change the network hyperparameter in ddpg.py or dqn.py

## How to cite us?

@misc{shipAI,

    author = {Jonathas Pereira and Rodrigo Pereira Abou Rejaili},
    
    title = {ShipAI},
    
    year = {2018},
    
    publisher = {GitHub},
    
    journal = {GitHub repository},
    
    howpublished = {\url{https://github.com/jmpf2018/ShipAI}},
    
}
