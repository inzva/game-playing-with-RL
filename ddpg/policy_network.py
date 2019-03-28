import numpy as np
import math
import keras.backend as K
import tensorflow as tf
from keras import Sequential, Input, Model
from keras.layers import Dense


class PolicyNetwork:
    def __init__(self, sess, hidden_units, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess

        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(self.sess)

        self._model, self._weights, self._state = self.initialize_model(hidden_units, state_size, action_size)
        self._target_model, self._target_weights, self._target_state = self.initialize_model(hidden_units, state_size, action_size)

        self._action_gradient = tf.placeholder(tf.float32, [None, action_size])

        self._params_grad = tf.gradients(self._model.output, self._weights, -self._action_gradient)

        grads = zip(self._params_grad, self._weights)

        self._optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)

        self.sess.run(tf.initialize_all_variables())

        self._target_model.set_weights(self._model.get_weights())

    def train(self, states, action_grads):
        self.sess.run(self._optimize, feed_dict={self._state: states, self._action_gradient: action_grads})

    def initialize_model(self, hidden_units, state_size, action_size):
        S = Input(shape=[state_size])
        x = Dense(hidden_units[0], activation='relu')(S)

        for i in range(1, len(hidden_units)-1):
            x = Dense(hidden_units[i], activation='relu')(x)

        if len(hidden_units) > 1:
            x = Dense(hidden_units[-1], activation='relu')(x)

        # actions can must be in between -1 and 1
        V = Dense(action_size, activation='tanh')(x)

        model = Model(input=S, output=V)
        return model, model.trainable_weights, S

    def update_target_network(self):
        critic_network_weights = self._model.get_weights()
        critic_target_network_weights = self._target_model.get_weights()

        for i in range(len(critic_network_weights)):
            critic_target_network_weights[i] = self.TAU * critic_target_network_weights[i] + (1 - self.TAU)* critic_target_network_weights[i]
        self._target_model.set_weights(critic_target_network_weights)
