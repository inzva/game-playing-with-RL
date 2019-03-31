import gym
import random
from collections import deque
from time import sleep
from datetime import datetime

import matplotlib.pyplot as plt

import json

import numpy as np

from keras import Sequential
from keras.engine.saving import model_from_json
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

EPISODES = 10000

OU = OU()


def train_the_bipedal(train_indicator=1):
    env = gym.make('BipedalWalker-v2')

    observation_space = env.observation_space
    allowed_actions = env.action_space

    action_dim = np.shape(allowed_actions)[0]
    state_dim = np.shape(observation_space)[0]

    BUFFER_SIZE = 5000

    GAMMA = 0.99

    BATCH_SIZE = 32  # Batch size for policy

    TAU_P = 0.001  # Tau for policy
    TAU_C = 0.001  # Tau for critic

    LR_P = 0.0001  # Learning rate for policy
    LR_C = 0.001  # Learning rate for critic

    HIDDEN_UNITS_POLICY = [32, 128, 256]
    HIDDEN_UNITS_CRITIC = [32, 128, 256]

    vision = False

    max_steps = 1000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    epsilon_decay = 0.9999

    config = tf.ConfigProto()

    sess = tf.Session(config=config)

    K.set_session(sess)

    policy = PolicyNetwork(sess, HIDDEN_UNITS_POLICY, state_dim, action_dim, BATCH_SIZE, TAU_P, LR_P)

    critic = CriticNetwork(sess, HIDDEN_UNITS_CRITIC, state_dim, action_dim, BATCH_SIZE, TAU_C, LR_C)

    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    total_rewards = []

    for e in range(EPISODES):
        print("Episode : " + str(e) + " Replay Buffer " + str(buff.count()))

        s_t = env.reset()

        total_reward = 0.

        step = 0

        for step in range(max_steps):
            loss = 0
            epsilon *= epsilon_decay
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            # a_t_original = policy.model.predict(s_t.reshape(1, s_t.shape[0]))

            """
            # Exploration is done by adding random noice into action decided by policy network
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0, 1.00, 0.1)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0, 1.00, 0.1)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0, 1.00, 0.1)
            noise_t[0][3] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][3], 0, 1.00, 0.1)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            a_t[0][3] = a_t_original[0][3] + noise_t[0][3]
            """
            if np.random.rand() <= epsilon:
                a_t = sess.run(tf.random.uniform([1, 4], minval=-1, maxval=1, dtype=tf.dtypes.float32))
            else:
                a_t = policy.model.predict(s_t.reshape(1, s_t.shape[0]))

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

            target_q_values = critic.target_model.predict([new_states, policy.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if train_indicator:
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = policy.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                policy.train(states, grads)

                policy.update_target_network()
                critic.update_target_network()

            total_reward += r_t
            s_t = s_t1

            print("Episode", e, "Epsilon", epsilon, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        total_rewards.append(total_reward)

        print("TOTAL REWARD @ " + str(e) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.close()
    # saving the models
    policy.model.save_weights("models/policymodel.h5", overwrite=True)
    with open("models/policymodel.json", "w") as outfile:
        json.dump(policy.model.to_json(), outfile)

    critic.model.save_weights("models/criticmodel.h5", overwrite=True)
    with open("models/criticmodel.json", "w") as outfile:
        json.dump(critic.model.to_json(), outfile)

    x_s = np.arange(1, EPISODES + 1)

    plt.plot(x_s, total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('plots/bipedal_rewards.png')

    sess.close()

    return policy.model, critic.model


def test_the_bipedal(policy_model, critic_model):
    env = gym.make('BipedalWalker-v2')

    observation_space = env.observation_space
    allowed_actions = env.action_space

    action_dim = np.shape(allowed_actions)[0]
    state_dim = np.shape(observation_space)[0]

    BUFFER_SIZE = 5000

    GAMMA = 0.99

    BATCH_SIZE = 32  # Batch size for policy

    TAU_P = 0.001  # Tau for policy
    TAU_C = 0.001  # Tau for critic

    LR_P = 0.0001  # Learning rate for policy
    LR_C = 0.001  # Learning rate for critic

    HIDDEN_UNITS_POLICY = [32, 128, 256]
    HIDDEN_UNITS_CRITIC = [32, 128, 256]

    vision = False

    max_steps = 1000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    epsilon_decay = 0.9999

    config = tf.ConfigProto()

    sess = tf.Session(config=config)

    K.set_session(sess)

    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    total_rewards = []

    for e in range(EPISODES):
        print("Episode : " + str(e) + " Replay Buffer " + str(buff.count()))

        s_t = env.reset()

        total_reward = 0.

        step = 0

        for step in range(max_steps):
            loss = 0
            epsilon *= epsilon_decay
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = policy_model.predict(s_t.reshape(1, s_t.shape[0]))

            a_t[0][0] = a_t_original[0][0]
            a_t[0][1] = a_t_original[0][1]
            a_t[0][2] = a_t_original[0][2]
            a_t[0][3] = a_t_original[0][3]

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

            target_q_values = critic_model.predict([new_states, policy_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            total_reward += r_t
            s_t = s_t1

            print("Episode", e, "Epsilon", epsilon, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        total_rewards.append(total_reward)

        print("TOTAL REWARD @ " + str(e) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.close()


def load_models():
    json_file = open('models/policymodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    policy_model = model_from_json(loaded_model_json)
    policy_model.load_weights("models/policymodel.h5")

    json_file = open('models/criticmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    critic_model = model_from_json(loaded_model_json)
    critic_model.load_weights("models/criticmodel.h5")

    return policy_model, critic_model


def main():

    policy_model, critic_model = train_the_bipedal()

    # policy_model, critic_model = load_models()

    test_the_bipedal(policy_model, critic_model)


if __name__ == "__main__":
    main()
