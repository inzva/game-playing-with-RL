import gym

import matplotlib.pyplot as plt

import json

import numpy as np

from keras import backend as K

import tensorflow as tf

from bipedal_walker.critic_network import CriticNetwork
from bipedal_walker.ornstein_uhlenbeck_process import OU

from bipedal_walker.policy_network import PolicyNetwork

from bipedal_walker.replay_buffer import ReplayBuffer


##############

EPISODES = 20000

FRAME_SIZE = 3  # include previous two frame as well into prediction and training

OU = OU()

BUFFER_SIZE = 5000

GAMMA = 0.99

BATCH_SIZE = 32  # Batch size for policy

TAU_P = 0.001  # Tau for policy
TAU_C = 0.001  # Tau for critic

LR_P = 0.0001  # Learning rate for policy
LR_C = 0.001  # Learning rate for critic

HIDDEN_UNITS_POLICY = [256, 64, 16]
HIDDEN_UNITS_CRITIC = [256, 64, 16]


def train_the_bipedal(train_indicator=1):
    env = gym.make('BipedalWalker-v2')

    observation_space = env.observation_space
    allowed_actions = env.action_space

    action_dim = np.shape(allowed_actions)[0]
    state_dim = np.shape(observation_space)[0]

    vision = False

    max_steps = 1000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    epsilon_decay = 0.9999
    exploration_noise = 0.1

    config = tf.ConfigProto()

    sess = tf.Session(config=config)

    K.set_session(sess)

    policy = PolicyNetwork(sess, HIDDEN_UNITS_POLICY, FRAME_SIZE * state_dim, action_dim, BATCH_SIZE, TAU_P, LR_P)

    critic = CriticNetwork(sess, HIDDEN_UNITS_CRITIC, FRAME_SIZE * state_dim, action_dim, BATCH_SIZE, TAU_C, LR_C)

    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    total_rewards = []

    for e in range(EPISODES):
        print("Episode : " + str(e) + " Replay Buffer " + str(buff.count()))

        s_t = np.zeros(FRAME_SIZE * state_dim)

        observation = env.reset()

        for i in range(FRAME_SIZE):
            s_t = np.append(s_t, observation)
            s_t = np.delete(s_t, np.s_[0:len(observation)])

        total_reward = 0.

        step = 0

        epsilon *= epsilon_decay

        for step in range(max_steps):
            loss = 0
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = policy.model.predict(s_t.reshape(1, s_t.shape[0]))

            """
            # Exploration is done by adding random noice into action decided by policy network
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0, 0, 0.1)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0, 0, 0.1)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0, 0, 0.1)
            noise_t[0][3] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][3], 0, 0, 0.1)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            a_t[0][3] = a_t_original[0][3] + noise_t[0][3]
            """

            """

            if np.random.rand() <= epsilon:
                a_t = sess.run(tf.random.uniform([1, 4], minval=-1, maxval=1, dtype=tf.dtypes.float32))
            else:
                a_t = policy.model.predict(s_t.reshape(1, s_t.shape[0]))
            """

            a_t = 0.2 * a_t_original[0] + np.random.normal(0, exploration_noise, size=action_dim)

            observation, r_t, done, info = env.step(a_t)

            env.render()

            s_t1 = np.append(s_t, observation)
            s_t1 = np.delete(s_t1, np.s_[0:len(observation)])

            buff.add(s_t, a_t, r_t, s_t1, done)  # Add replay buffer

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
    policy.model.save_weights("bipedal_walker/models/policymodel.h5", overwrite=True)
    with open("bipedal_walker/models/policymodel.json", "w") as outfile:
        json.dump(policy.model.to_json(), outfile)

    critic.model.save_weights("bipedal_walker/models/criticmodel.h5", overwrite=True)
    with open("bipedal_walker/models/criticmodel.json", "w") as outfile:
        json.dump(critic.model.to_json(), outfile)

    x_s = np.arange(1, EPISODES + 1)

    plt.plot(x_s, total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('plots/bipedal_rewards.png')

    sess.close()

    return policy.model, critic.model


def test_the_bipedal():
    env = gym.make('BipedalWalker-v2')

    observation_space = env.observation_space
    allowed_actions = env.action_space

    action_dim = np.shape(allowed_actions)[0]
    state_dim = np.shape(observation_space)[0]

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

    policy_model, critic_model = load_models(sess, state_dim, action_dim)

    total_rewards = []

    for e in range(1):
        observation = env.reset()
        s_t = np.zeros(FRAME_SIZE * state_dim)

        total_reward = 0.

        step = 0

        for i in range(FRAME_SIZE):
            s_t = np.append(s_t, observation)
            s_t = np.delete(s_t, np.s_[0:len(observation)])

        for step in range(max_steps):
            loss = 0
            epsilon *= epsilon_decay
            a_t = np.zeros([1, action_dim])

            a_t_original = policy_model.model.predict(s_t.reshape(1, s_t.shape[0]))

            a_t[0][0] = a_t_original[0][0]
            a_t[0][1] = a_t_original[0][1]
            a_t[0][2] = a_t_original[0][2]
            a_t[0][3] = a_t_original[0][3]

            ob, r_t, done, info = env.step(a_t[0])

            env.render()

            s_t1 = np.append(s_t, observation)
            s_t1 = np.delete(s_t1, np.s_[0:len(observation)])

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


def load_models(sess, state_dim, action_dim):
    policy_model = PolicyNetwork(sess, HIDDEN_UNITS_POLICY, FRAME_SIZE * state_dim, action_dim, BATCH_SIZE, TAU_P, LR_P)
    policy_model.model.load_weights("models/policymodel.h5")

    critic_model = CriticNetwork(sess, HIDDEN_UNITS_CRITIC, FRAME_SIZE * state_dim, action_dim, BATCH_SIZE, TAU_C, LR_C)
    critic_model.model.load_weights("models/criticmodel.h5")

    return policy_model, critic_model


def main():

    policy_model, critic_model = train_the_bipedal()

    test_the_bipedal()


if __name__ == "__main__":
    main()
