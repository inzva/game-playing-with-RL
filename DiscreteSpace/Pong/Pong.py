from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import gym
from PIL import Image

def prepro(I):


    I = I[35:195] # crop

    I = I[::2,::2,0] # downsample by factor of 2

    I[I == 144] = 0 # erase background (background type 1)

    I[I == 109] = 0 # erase background (background type 2)

    I[I != 0] = 1 # everything else (paddles, ball) just set to 1

    return I.astype(np.float).ravel()

def discount_rewards(r, gamma):
  """ take 1D float array of rewards and compute discounted reward """
  r = np.array(r)
  discounted_r = np.zeros_like(r)
  running_add = 0
  # we go from last reward to first one so we don't have to do exponentiations
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # if the game ended (in Pong), reset the reward sum
    running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
    discounted_r[t] = running_add
  discounted_r -= np.mean(discounted_r) #normalizing the result
  discounted_r /= np.std(discounted_r) #idem
  return discounted_r

model = Sequential()

model.add(Dense(units=200, input_dim=80*80, activation = 'relu'))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

model.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

env = gym.make('Pong-v0')
observation = env.reset()
prev_input = None

gamma = 0.99

x_train, y_train, rewards = [],[],[]
reward_sum = 0
episode_nb = 0

while True:
    current_input = prepro(observation)


    if prev_input is not None:
        difference = current_input - prev_input
    else:
        difference = np.zeros(80*80)
    prev_input = current_input

    probability = model.predict(np.expand_dims(difference, axis=1).T)
    action = 2 if np.random.uniform()<probability else 3
    y = 1 if action ==2 else 0

    x_train.append(difference)
    y_train.append(y)
    env.render()
    observation, reward, done, info = env.step(action)
    rewards.append(reward)
    reward_sum += reward

    if done:

        episode_nb +=1
        print('Total Reward is {}'.format(reward_sum))
        model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=0, sample_weight = discount_rewards(rewards, gamma))

        x_train, y_train, rewards = [],[],[]
        observation = env.reset()
        reward_sum = 0
        prev_input = None
