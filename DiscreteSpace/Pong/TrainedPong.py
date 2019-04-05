import numpy as np
import pickle
import gym
from time import sleep

with open(r"save.p", "rb") as file:
    model = pickle.load(file)

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p # return probability of taking action 2, and hidden state

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
while True:
  env.render()
  sleep(.035)

  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(6400)
  prev_x = cur_x


  aprob = policy_forward(x)
  action = 2 if 0.5 < aprob else 3
  observation, reward, done, info = env.step(action)

  if done:
      prev_x = None
      observation = env.reset()
