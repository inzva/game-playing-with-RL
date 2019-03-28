import gym
import random
from collections import deque
from time import sleep
from datetime import datetime

import json

import numpy as np

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import keras

import tensorflow as tf

from critic_network import CriticNetwork
from ornstein_uhlenbeck_process import OU

from policy_network import PolicyNetwork

############
from replay_buffer import ReplayBuffer

EPISODES = 50000

OU = OU()


def train_the_bipedal(train_indicator=1):
    env = gym.make('BipedalWalker-v2')

    observation_space = env.observation_space
    allowed_actions = env.action_space

    action_dim = np.shape(allowed_actions)[0]
    state_dim = np.shape(observation_space)[0]

    BUFFER_SIZE = 100000

    GAMMA = 0.99

    BATCH_SIZE = 32  # Batch size for policy
    # BATCH_SIZE_C = 32  # Batch size for critic

    TAU_P = 0.001  # Tau for policy
    TAU_C = 0.001  # Tau for critic

    LR_P = 0.0001  # Learning rate for policy
    LR_C = 0.001  # Learning rate for critic

    HIDDEN_UNITS_POLICY = [32, 256]
    HIDDEN_UNITS_CRITIC = [32, 256]

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1

    indicator = 0

    config = tf.ConfigProto()

    sess = tf.Session(config=config)

    K.set_session(sess)

    policy = PolicyNetwork(sess, HIDDEN_UNITS_POLICY, state_dim, action_dim, BATCH_SIZE, TAU_P, LR_P)

    critic = CriticNetwork(sess, HIDDEN_UNITS_CRITIC, state_dim, action_dim, BATCH_SIZE, TAU_C, LR_C)

    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    for e in range(EPISODES):
        print("Episode : " + str(e) + " Replay Buffer " + str(buff.count()))

        s_t = env.reset()

        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = policy._model.predict(s_t.reshape(1, s_t.shape[0]))

            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], -0.1, 1.00, 0.05)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], -0.1, 1.00, 0.05)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)
            noise_t[0][3] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            a_t[0][3] = a_t_original[0][3] + noise_t[0][3]

            ob, r_t, done, info = env.step(a_t[0])

            env.render()

            s_t1 = ob

            buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])

            y_t = np.asarray([e[2] for e in batch])

            target_q_values = critic._target_model.predict([new_states, policy._target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if (train_indicator):
                loss += critic._model.train_on_batch([states, actions], y_t)
                a_for_grad = policy._model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                policy.train(states, grads)
                policy.update_target_network()
                critic.update_target_network()

            total_reward += r_t
            s_t = s_t1

            print("Episode", e, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break


        print("TOTAL REWARD @ " + str(e) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.close()
    # saving the models
    policy._model.save_weights("actormodel.h5", overwrite=True)
    with open("policymodel.json", "w") as outfile:
        json.dump(policy._model.to_json(), outfile)

    critic._model.save_weights("criticmodel.h5", overwrite=True)
    with open("criticmodel.json", "w") as outfile:
        json.dump(critic._model.to_json(), outfile)


def main():

    trained_bird = train_the_bipedal()

    # let_trained_bipedal_play_the_game(trained_bird)


if __name__ == "__main__":
    main()
