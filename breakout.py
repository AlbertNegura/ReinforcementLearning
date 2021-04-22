import numpy as np
import gym
from models import build_atari_model
from agents import build_rl_agent
from tensorflow.keras.optimizers import Adam

env = gym.make('Breakout-v0')
episodes = 10
verbosity = 1
visualize = False

height, width, channels = env.observation_space.shape
actions = env.action_space.n

model = build_atari_model(height, width, channels, actions)
dqn = build_rl_agent(model, actions)

dqn.compile(Adam(lr=0.001))
dqn.fit(env, nb_steps=40000, visualize=visualize, verbose=verbosity)

for episode in range(1, episodes):
    state = env.reset()
    finished = False
    score = 0

    while not finished:
        env.render()
        state, reward, finished, info = env.step(env.action_space.sample())
        score+=reward

    print('Episode: {}\nScore: {}'.format(episode, score))

env.close()
