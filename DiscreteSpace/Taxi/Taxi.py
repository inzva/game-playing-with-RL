import gym
from time import sleep
import numpy as np


n_times = 100000
alfa = 0.1
epsilon = 0.2
gamma = 0.6

env = gym.make('Taxi-v2')
q_table = np.zeros((env.observation_space.n, env.action_space.n))

for _ in range(n_times+1):

    state = env.reset()

    done = False

    while not done:

        rand_number = np.random.uniform(0,1)

        if rand_number<epsilon:
            action = env.action_space.sample()

        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_q_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        q_table[state, action] = (1-alfa)*old_q_value + alfa*(reward + gamma*next_max)

        state = next_state

    if _%100==0:
        
        print('Training: {} / {}'.format(_, n_times))



print('Training finished..')
sleep(0.8)


for _ in range(20):

    state = env.reset()

    done = False

    while not done:
        action = np.argmax(q_table[state])
        env.render()
        state, reward, done, info = env.step(action)

        sleep(0.4)
