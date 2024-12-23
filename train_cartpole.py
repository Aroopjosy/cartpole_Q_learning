import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt



def plot_performance(rewards_per_episode):
        plt.plot(rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Performance: CartPole-v1') 
        plt.grid()
        plt.savefig('cartpole_episode_total_reward.png')  # Save the second plot
        # plt.show()


def plot_learning_curve(rewards_per_episode, window_size=100):
    """Plot the learning curve by averaging rewards over a sliding window."""
    smoothed_rewards = [np.mean(rewards_per_episode[max(0, i - window_size):i + 1]) for i in range(len(rewards_per_episode))]
    plt.plot(smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Learning Curve: CartPole-v1')
    plt.grid()
    plt.savefig('learning_curve.png')  # Save the learning curve plot
    # plt.show()  # Display the plot  

def plot_epsilon_decay(epsilon_history):
    """Plot the decay of epsilon over episodes."""
    plt.plot(epsilon_history)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay over Episodes')
    plt.grid()
    plt.savefig('epsilon_decay.png')  # Save the epsilon decay plot


def train(render=False):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Define state spaces (position, velocity, pole angle, pole angular velocity)
    pos_space = np.linspace(-2.4, 2.4, 20)
    vel_space = np.linspace(-4, 4, 20)
    ang_space = np.linspace(-.2095, .2095, 20)
    ang_vel_space = np.linspace(-4, 4, 20)

    Q_table = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n)) # 15x15x15x15x2 array
    print("Q_table initialized")
    print("Q_table Shape: ", Q_table.shape)

    learning_rate_a = 0.1  # alpha or learning rate
    discount_factor_g = 0.99  # gamma or discount factor
    epsilon = 1  # Start with 100% random actions
    epsilon_decay_rate = 0.0001  # Epsilon decay rate
    rng = np.random.default_rng()  # Random number generator

    rewards_per_episode = []

    i = 0

    while True:
        state = env.reset()[0]  # Starting position, velocity, angle, angular velocity
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False
        rewards = 0

        while not terminated and rewards < 10000:
            if rng.random() < epsilon:
                action = env.action_space.sample()  # Choose random action exploring the environment
            else:
                action = np.argmax(Q_table[state_p, state_v, state_a, state_av, :])  # Choose best action from Q-table

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            Q_table[state_p, state_v, state_a, state_av, action] += learning_rate_a * (
                reward + discount_factor_g * np.max(Q_table[new_state_p, new_state_v, new_state_a, new_state_av, :]) - Q_table[state_p, state_v, state_a, state_av, action]
            )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward

        rewards_per_episode.append(rewards)

        if i % 100 == 0:
            print(f'Episode: {i} Rewards: {rewards} Epsilon: {epsilon:.2f} Mean Reward: {np.mean(rewards_per_episode[-100:]):.2f}')

        if np.mean(rewards_per_episode[-100:]) > 100:
            print("Training completed. Mean reward over 100 episodes: ", np.mean(rewards_per_episode[-100:]))
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)  # Decay epsilon

        i += 1

    env.close()

    # Save Q_table to file
    # with open('cartpole_Q_table111.pkl', 'wb') as f:
    #     pickle.dump(Q_table, f)
    #     f.close()
    np.save('Q_table.npy', Q_table)
    print("Q_table saved")

    # Plot the performance
    plot_performance(rewards_per_episode)





    # # Plot the rewards
    # mean_rewards = [np.mean(rewards_per_episode[max(0, t-100):(t+1)]) for t in range(i)]
    # plt.plot(mean_rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Mean Reward")
    # plt.title("Training Performance - CartPole")
    # plt.grid()
    # plt.savefig('cartpole_training.png')
    # print("Training completed. Q_table saved.")

    
    





if __name__ == '__main__':
    train()
