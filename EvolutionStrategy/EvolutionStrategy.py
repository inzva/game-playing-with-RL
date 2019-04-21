import gym
import numpy as np
import pickle
import sys

env = gym.make('BipedalWalker-v2')
np.random.seed(10)

hl_size = 100
version = 1
npop = 50
sigma = 0.1
alpha = 0.03
iter_num = 300
aver_reward = None
allow_writing = True
reload = True



if reload:
    with open(r"TrainedWeights.pickle", "rb") as input_file:
       model = pickle.load(input_file)

else:
    model = {}
    model['W1'] = np.random.randn(24, hl_size) / np.sqrt(24)
    model['W2'] = np.random.randn(hl_size, 4) / np.sqrt(hl_size)

def get_action(state, model):
    hl = np.matmul(state, model['W1'])
    hl = np.tanh(hl)
    action = np.matmul(hl, model['W2'])
    action = np.tanh(action)

    return action

def f(model, render=True):
    state = env.reset()
    total_reward = 0
    for t in range(iter_num):
        if render: env.render()

        action = get_action(state, model)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break
    return total_reward

if reload:
    iter_num = 10000
    for i_episode in range(10):
        print(f(model, True))
    sys.exit('demo finished')

for i in range(10001):
    N = {}
    for k, v in model.items():
        N[k] = np.random.randn(npop, v.shape[0], v.shape[1])
    R = np.zeros(npop)

    for j in range(npop):
        model_try = {}
        for k, v in model.items():
            model_try[k] = v + sigma*N[k][j]
        R[j] = f(model_try)

    A = (R - np.mean(R)) / np.std(R)
    for k in model:
        model[k] = model[k] + alpha/(npop*sigma) * np.dot(N[k].transpose(1, 2, 0), A)

    cur_reward = f(model)
    aver_reward = aver_reward * 0.9 + cur_reward * 0.1 if aver_reward != None else cur_reward
    print('iter %d, cur_reward %.2f, aver_reward %.2f' % (i, cur_reward, aver_reward))
