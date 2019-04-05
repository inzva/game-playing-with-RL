import numpy as np
import tensorflow as tf
import gym
from time import sleep

env = gym.make('CartPole-v0')
env.unwrapped
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

epoch = 3000
learning_rate = 0.01
gamma = 0.96

def reward_normalization(reward):
    dis_reward = np.zeros_like(reward)
    toplam = 0

    for i in reversed(range(len(reward))):
        toplam = toplam*gamma + reward[i]
        dis_reward[i] = toplam

    mean = np.mean(dis_reward)
    std = np.std(dis_reward)
    norm_dis_reward = (dis_reward - mean) / std

    return norm_dis_reward

x = tf.placeholder(tf.float32, [None, state_size])
actions = tf.placeholder(tf.int32, [None, action_size])
rewards = tf.placeholder(tf.float32, [None,])

fc1 = tf.contrib.layers.fully_connected(inputs = x, num_outputs =8,
                                        activation_fn = tf.nn.relu,
                                        weights_initializer = tf.contrib.layers.xavier_initializer())

fc2 = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs =4,
                                        activation_fn = tf.nn.relu,
                                        weights_initializer = tf.contrib.layers.xavier_initializer())

fc3 = tf.contrib.layers.fully_connected(inputs = fc2, num_outputs =2,
                                        activation_fn = None,
                                        weights_initializer = tf.contrib.layers.xavier_initializer())

action_taken = tf.nn.softmax(fc3)

log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions )

loss = tf.reduce_mean(log_prob * rewards)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

episode_states, episode_actions, episode_rewards = [],[],[]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epoch+1):
        episode_rew = 0

        state = env.reset()

        env.render()

        while True:
            pick_action = sess.run(action_taken, feed_dict = { x: state.reshape(1,4)})

            action = np.random.choice(2, p=pick_action.ravel())

            episode_states.append(state)

            state, reward, done, info = env.step(action)
            sleep(0.01)
            episode_rew += reward

            action_ = np.zeros(action_size)
            action_[action] = 1 #seçilen aksiyonu 1'e eşitle
            episode_actions.append(action_)

            episode_rewards.append(reward)

            if done:
                print('{} episode reward = {}'.format(i,episode_rew))
                
                if i%1==0:
                    discounted_rewards = reward_normalization(episode_rewards)


                    loss_, _ = sess.run([loss, optimizer], feed_dict= { x: np.vstack(episode_states),
                                                                   actions: np.vstack(episode_actions),
                                                                   rewards: discounted_rewards})

                    episode_states, episode_actions, episode_rewards = [],[],[]

                break
    env.close()
