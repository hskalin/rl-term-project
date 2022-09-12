import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import cliffworld
import rlalgos as rl

w = cliffworld.CliffWorld()

# print optimal policy
def print_optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, w.WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, w.WORLD_WIDTH):
            if [i, j] == w.GOAL:
                optimal_policy[-1].append('G')
                continue
            elif [i, j] in w.cliff:
                optimal_policy[-1].append('O')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == w.ACTION_UP:
                optimal_policy[-1].append('↑')
            elif bestAction == w.ACTION_DOWN:
                optimal_policy[-1].append('↓')
            elif bestAction == w.ACTION_LEFT:
                optimal_policy[-1].append('←')
            elif bestAction == w.ACTION_RIGHT:
                optimal_policy[-1].append('→')
    for row in optimal_policy:
        print(row)


def sim_run(q_value):
    canvas = [[' ' for x in range(w.WORLD_WIDTH)] for y in range(w.WORLD_HEIGHT)]

    state = w.START
    s = 0
    while state != w.GOAL:
        action = rl.choose_action(state, q_value, w, True)
        next_state, _ = w.step(state, action)
        
        canvas[next_state[0]][next_state[1]] = 'X'

        state = next_state
        #print(s)
        s += 1

    for i in range(w.WORLD_HEIGHT):
        for j in range(w.WORLD_WIDTH):
            if [i, j] == w.GOAL:
                canvas[i][j] = 'G'
            elif [i, j] == w.START:
                canvas[i][j] = 'S'
            elif [i, j] in w.cliff:
                canvas[i][j] = 'O'

    
    for row in canvas:
        print(row)
        
    return s

def run():

    q_value = np.zeros((w.WORLD_HEIGHT, w.WORLD_WIDTH, 4))
    episode_limit = 1000

    steps = []
    
    for ep in tqdm(range(episode_limit)):
        steps.append(rl.q_learning(q_value, w)[1])
        #steps.append(sarsa(q_value, True, ALPHA)[1])
        

    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig('figure_6_3.png')
    plt.close()

    # display optimal policy
    #print('Sarsa Optimal Policy:')
    #print_optimal_policy(q_sarsa)
    print('Q-Learning Optimal Policy:')
    print_optimal_policy(q_value)

    print('a simulation run:')
    print(sim_run(q_value))


def plot_rewards():
    # episodes of each run
    episodes = 500

    # perform 40 independent runs
    runs = 50

    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)
    for r in tqdm(range(runs)):
        q_sarsa = np.zeros((w.WORLD_HEIGHT, w.WORLD_WIDTH, 4))
        q_q_learning = np.copy(q_sarsa)
        for i in range(0, episodes):
            # cut off the value by -100 to draw the figure more elegantly
            # rewards_sarsa[i] += max(sarsa(q_sarsa), -100)
            # rewards_q_learning[i] += max(q_learning(q_q_learning), -100)
            rewards_sarsa[i] += rl.sarsa(q_sarsa, w)[0]
            rewards_q_learning[i] += rl.q_learning(q_q_learning, w)[0]

    # averaging over independt runs
    rewards_sarsa /= runs
    rewards_q_learning /= runs

    # draw reward curves
    plt.plot(rewards_sarsa, label='Sarsa')
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('rewards-vs-episodes.png')
    plt.close()

    # display optimal policy
    print('Sarsa Optimal Policy:')
    print_optimal_policy(q_sarsa)
    print('Q-Learning Optimal Policy:')
    print_optimal_policy(q_q_learning)


def plot_performance():
    step_sizes = np.arange(0.1, 1.1, 0.1)
    episodes = 1000
    runs = 10

    ASY_SARSA = 0
    ASY_EXPECTED_SARSA = 1
    ASY_QLEARNING = 2
    INT_SARSA = 3
    INT_EXPECTED_SARSA = 4
    INT_QLEARNING = 5
    methods = range(0, 6)

    performace = np.zeros((6, len(step_sizes)))
    for run in range(runs):
        for ind, step_size in tqdm(list(zip(range(0, len(step_sizes)), step_sizes))):
            q_sarsa = np.zeros((w.WORLD_HEIGHT, w.WORLD_WIDTH, 4))
            q_expected_sarsa = np.copy(q_sarsa)
            q_q_learning = np.copy(q_sarsa)
            for ep in range(episodes):
                sarsa_reward = rl.sarsa(q_sarsa, w, expected=False, step_size=step_size)[0]
                expected_sarsa_reward = rl.sarsa(q_expected_sarsa, w, expected=True, step_size=step_size)[0]
                q_learning_reward = rl.q_learning(q_q_learning, w, step_size=step_size)[0]
                performace[ASY_SARSA, ind] += sarsa_reward
                performace[ASY_EXPECTED_SARSA, ind] += expected_sarsa_reward
                performace[ASY_QLEARNING, ind] += q_learning_reward

                if ep < 100:
                    performace[INT_SARSA, ind] += sarsa_reward
                    performace[INT_EXPECTED_SARSA, ind] += expected_sarsa_reward
                    performace[INT_QLEARNING, ind] += q_learning_reward

    performace[:3, :] /= episodes * runs
    performace[3:, :] /= 100 * runs
    labels = ['Asymptotic Sarsa', 'Asymptotic Expected Sarsa', 'Asymptotic Q-Learning',
              'Interim Sarsa', 'Interim Expected Sarsa', 'Interim Q-Learning']

    for method, label in zip(methods, labels):
        plt.plot(step_sizes, performace[method, :], label=label)
    plt.xlabel('alpha')
    plt.ylabel('reward per episode')
    plt.legend()

    plt.savefig('performance.png')
    plt.close()

if __name__ == '__main__':
    run()

    plot_rewards()
    plot_performance()