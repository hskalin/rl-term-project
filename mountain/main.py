import numpy as np
import matplotlib.pyplot as plt
import mountainCarCont
from rlalgos import QLearning, DQNAgent, A2CAgent



def main():
    # Import and initialize Mountain Car Environment
    #env = gym.make('MountainCar-v0', render_mode="rgb_array")
    env = mountainCarCont.Continuous_MountainCarEnv()
    env.reset()

    num_frames = 1000
    gamma = 0.9
    entropy_weight = 1e-2

    agent = A2CAgent(env, gamma, entropy_weight)

    agent.train(num_frames)

    frames = agent.test()

    exit()

    num_frames = 1000
    memory_size = 1000
    batch_size = 32
    target_update = 100
    epsilon_decay = 1 / 2000

    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
    agent.train(num_frames, plotting_interval=5)

    video_folder="dqn"
    agent.test(video_folder=video_folder)

    exit()

    # Run Q-learning algorithm
    rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 1000)

    # Plot Rewards
    plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('rewards.jpg')     
    plt.close()  

if __name__ == '__main__':
    main()