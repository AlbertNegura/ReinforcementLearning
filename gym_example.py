import numpy as np
import gym
from models import build_atari_model
from agents import build_rl_agent
from tensorflow.keras.optimizers import Adam

game = 'SI' # SI, BO, AS, FW
episodes = 10
verbosity = 1
visualize = True



game_name = 'SpaceInvaders-v0' if game=='SI' else 'Breakout-v0' if game == 'BO' else 'Assault-v0' if game == 'AS' else 'Freeway-v0'
env = gym.make(game_name)
height, width, channels = env.observation_space.shape
actions = env.action_space.n

model = build_atari_model(height, width, channels, actions)
dqn = build_rl_agent(model, actions)
dqn.compile(Adam(lr=0.01))
try:
    dqn.load_weights('models/{}.h5f'.format(game))
except:
    dqn.fit(env, nb_steps=40000, visualize=False, verbose=verbosity)
    dqn.save_weights('models/{}.h5f'.format(game))

scores = dqn.test(env, nb_episodes=episodes, visualize=visualize)
print(np.mean(scores.history['episode_reward']))

env.close()
