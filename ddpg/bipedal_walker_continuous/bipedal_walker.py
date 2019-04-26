import gym

import matplotlib.pyplot as plt

import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

from bipedal_walker_continuous.agent import Agent


class BipedalWalker:

    def __init__(self, env, state_size, action_size):
        self.env = self.new_env()
        self.state_size = state_size
        self.action_size = action_size
        self.agent = Agent(self.state_size, self.action_size)

    """
    def new_env(self):
        gym.logger.set_level(40)  # to surpress warnings
        return gym.make('MountainCarContinuous-v0').unwrapped
    """
    def new_env(self):
        gym.logger.set_level(40)  # to surpress warnings
        return gym.make('BipedalWalker-v2').unwrapped

    def preprocess_state(self, state):
        # mapping the state values to [-1,1]
        s = np.array(state)
        s[0] = ((state[0] + 1.2) / 1.8) * 2 - 1
        s[1] = ((state[1] + 0.07) / 0.14) * 2 - 1
        return s

    """
    def plot_Q(self):
        
        Plots 4 heatmaps that shows the behavior of the
        local critic and target when dealing with the state
        and actions space
        
        state_step = 0.1
        action_step = 0.1
        plot_range = np.arange(-1, 1 + state_step, state_step)
        action_range = np.arange(-1, 1 + action_step, action_step)
        shape = plot_range.shape[0]

        matrix_Q = np.ones((shape, shape))
        matrix_mQ = np.ones((shape, shape))
        matrix_sQ = np.ones((shape, shape))
        matrix_A = np.ones((shape, shape))

        for i in range(shape):
            for j in range(shape):
                pos = plot_range[j]
                vel = plot_range[i]
                state = np.array([pos, vel]).reshape(-1, 2)
                Q_list = []
                for a in action_range:
                    action = np.array(a).reshape(-1, 1)
                    Q_list.append(self.agent.critic_local.model.predict(
                                  [state, action]))
                matrix_Q[i][j] = np.max(Q_list)
                matrix_sQ[i][j] = np.std(Q_list)
                matrix_mQ[i][j] = action_range[np.argmax(Q_list)]
                matrix_A[i][j] = self.agent.actor_local.model.predict(state)
        extent = [plot_range[0], plot_range[-1], plot_range[0], plot_range[-1]]

        fig, ax = plt.subplots(2, 2, sharex=True)
        ax[0, 0].set_title('Q value max')
        ax[0, 0].set_ylabel('Velocity')
        ax[0, 0].set_xlabel('Position')
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[0, 0].imshow(matrix_Q, extent=extent, origin='lower')
        plt.colorbar(im, cax=cax)

        ax[0, 1].set_title('Q value std')
        ax[0, 1].set_ylabel('Velocity')
        ax[0, 1].set_xlabel('Position')
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[0, 1].imshow(matrix_sQ, extent=extent, origin='lower')
        plt.colorbar(im, cax=cax)

        ax[1, 0].set_title('Action with Q max')
        ax[1, 0].set_ylabel('Velocity')
        ax[1, 0].set_xlabel('Position')
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[1, 0].imshow(matrix_mQ, extent=extent, origin='lower')
        plt.colorbar(im, cax=cax)

        ax[1, 1].set_title('Predicted Action')
        ax[1, 1].set_ylabel('Velocity')
        ax[1, 1].set_xlabel('Position')
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[1, 1].imshow(matrix_A, extent=extent, origin='lower')
        plt.colorbar(im, cax=cax)

        plt.subplots_adjust(top=0.92, right=0.95, hspace=0.25, wspace=0.4)

        plt.savefig('plots/plots.png')
    """

    def run_epoch(self, max_steps, render=True, training=True):
        state = self.env.reset()
        self.agent.reset_episode(state)
        actions_list = []
        total_reward = 0
        steps = 0
        while steps < max_steps:
            steps += 1
            noisy_action, pure_action = self.agent.decide_actions(state)

            # use action with OUNoise if training
            action = noisy_action if training else pure_action

            # step into the environment and update values
            next_state, reward, done, info = self.env.step(action)
            #next_state = self.preprocess_state(next_state)
            state = next_state
            total_reward += reward
            actions_list.append(pure_action)

            # only train agent if in training
            if training:
                self.agent.step(action, reward, next_state, done)

            if render:
                self.env.render()

            if done:
                if render:  # workaround render errors
                    self.env.close()
                    self.env = self.new_env()
                break

        action_mean = np.mean(actions_list)
        action_std = np.std(actions_list)

        return total_reward, done, action_mean, action_std, steps

    def run_model(self, max_epochs=100, n_solved=1, r_solved=90, max_steps=1000, plot_Q=False, verbose=1):
        """
        Train the learner

        Params
            ======
                max_epochs (int): Maximum number of training episodes
                max_steps (int): Maximum steps in each episode
                r_solved (int): Minimum reward value to consider episode solved
                n_solved (int): Targed number of solved episodes before break
                plot_Q (bool): If true will plot state action values heatmaps
                verbose (int): How much information each epoch will print,
                  possible values are 1,0 and-1 in order of verbosity
        """

        solved = False
        train_hist = []
        test_hist = []

        for epoch in range(1, max_epochs+1):
            train_reward, train_done, train_action_mean, train_action_std, train_steps = self.run_epoch(max_steps=max_steps)
            test_reward, test_done, test_action_mean, test_action_std, test_steps = self.run_epoch(max_steps=1000, training=False)

            train_hist.append([train_reward, train_steps])
            test_hist.append([test_reward, test_steps])

            # check if solved
            # if the mean of last n_solved teste episodes are
            # greater than r_solved, it is solved!

            if epoch > n_solved:
                train_running = np.mean([r for r, s in train_hist][-n_solved:])
                test_running = np.mean([r for r, s in test_hist][-n_solved:])
            else:
                train_running = np.mean([r for r, s in train_hist])
                test_running = np.mean([r for r, s in test_hist])

            print_vals = np.array([epoch, train_reward, train_steps,
                                   train_running, train_action_mean,
                                   train_action_std, test_reward, test_steps,
                                   test_running, test_action_mean,
                                   test_action_std])

            self.print_epoch(print_vals, verbose)
            if test_running > r_solved and epoch > n_solved:
                    print('\nSolved! Average of {:4.1f} from episode {:3d}'' to {:3d}'.format(test_running, epoch - n_solved + 1, epoch))
                    solved = epoch
                    break

        if plot_Q:
            pass
            # TODO fix plot_Q function
            #self.plot_Q()
        return train_hist, test_hist, solved

    def print_epoch(self, vals, verbose):
        if verbose == 1:
            pstr = ('Epoch:{:4}\nTrain: reward:{: 6.1f} steps:{:6.0f} hist:'
                    '{: 6.1f} action/std:{: .3f}/{: .3f} \nTest:  reward:'
                    '{: 6.1f} steps:{:6.0f} hist:{: 6.1f} action/std:{: .3f}'
                    '/{: .3f}\n'.format(*vals))
        elif verbose == 0:
            pstr = ('Epoch {:4} train reward:{: 6.1f} test reward:{: 6.1f}'
                    '\r'.format(*vals[[0, 1, 6]]))
        elif verbose == -1:
            return
        print(pstr)


if __name__ == '__main__':
    print('Running learner directly')

    env = gym.make('BipedalWalker-v2')

    observation_space = env.observation_space
    allowed_actions = env.action_space

    action_dim = np.shape(allowed_actions)[0]
    state_dim = np.shape(observation_space)[0]

    Learner = BipedalWalker(env, state_dim, action_dim)
    _, _, solved = Learner.run_model(max_epochs=10000, max_steps=2500, n_solved=1, plot_Q=True)

    env.close()

    if solved:
        Learner.agent.save_models()
