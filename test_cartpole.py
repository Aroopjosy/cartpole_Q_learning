import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def test():
    env = gym.make('CartPole-v1', render_mode='human')  # Set render_mode to 'human' to visualize

    # Define state spaces (position, velocity, pole angle, pole angular velocity)
    pos_space = np.linspace(-2.4, 2.4, 20)
    vel_space = np.linspace(-4, 4, 20)
    ang_space = np.linspace(-.2095, .2095, 20)
    ang_vel_space = np.linspace(-4, 4, 20)

    # Load pre-trained Q-table from file and print its shape
    Q_table = np.load('Q_table.npy', allow_pickle=True)
    print("Q_table loaded")
    print("Q_table Shape: ", Q_table.shape)

    rewards_per_episode = [] # Store rewards for each episode
    i = 0 # Episode counter

    while i < 10:  # Test for 10 episodes
        state = env.reset()[0]  # Starting position, velocity, angle, angular velocity
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False # Episode termination flag
        rewards = 0 # Reward counter

        while not terminated: # Loop until episode is terminated
            action = np.argmax(Q_table[state_p, state_v, state_a, state_av, :])  # Choose best action from Q-table
            new_state, reward, terminated, truncated, info = env.step(action) # Take action and observe new state, reward, and termination

            new_state_p = np.digitize(new_state[0], pos_space) 
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward

        rewards_per_episode.append(rewards) # Store rewards for this episode
        print(f'Episode: {i} Rewards: {rewards}') # Print episode number and rewards

        i += 1 # Increment episode counter

    env.close() # Close the environment

if __name__ == '__main__':
    test()
