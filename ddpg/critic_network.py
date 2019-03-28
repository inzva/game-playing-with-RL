import numpy as np
import keras.backend as K
import tensorflow as tf
from keras import Sequential, Input, Model
from keras.layers import Dense, concatenate
from keras.optimizers import Adam


class CriticNetwork:
    def __init__(self, sess, hidden_units, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        print('Actor network, tısss')
        self.sess = sess

        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(self.sess)

        self._model, self._action, self._state = self.initialize_model(hidden_units, state_size, action_size)
        self._target_model, self.target_action, self._target_state = self.initialize_model(hidden_units, state_size, action_size)

        self._action_grads = tf.gradients(self._model.output, self._action)

        self.sess.run(tf.initialize_all_variables())

        self._target_model.set_weights(self._model.get_weights())

    def gradients(self, states, actions):
        return self.sess.run(self._action_grads, feed_dict={self._state: states, self._action: actions})[0]

    def initialize_model(self, hidden_units, state_size, action_size):
        S = Input(shape=[state_size])
        A = Input(shape=[action_size])

        w1 = Dense(hidden_units[0], activation='relu')(S)
        a1 = Dense(hidden_units[0], activation='relu')(A)

        x = concatenate([w1, a1])

        for i in range(1, len(hidden_units) - 1):
            x = Dense(hidden_units[i], activation='relu')(x)

        if len(hidden_units) > 1:
            x = Dense(hidden_units[-1], activation='relu')(x)

        V = Dense(1, activation='linear')(x)

        model = Model(input=[S, A], output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S

    def update_target_network(self):
        critic_network_weights = self._model.get_weights()
        critic_target_network_weights = self._target_model.get_weights()

        for i in range(len(critic_network_weights)):
            critic_target_network_weights[i] = self.TAU * critic_target_network_weights[i] + (1 - self.TAU)* critic_target_network_weights[i]
        self._target_model.set_weights(critic_target_network_weights)
