import numpy as np

# the grid world class
class CliffWorld:
    def __init__(self) -> None:
        # world width
        self.WORLD_WIDTH = 12

        # world height
        self.WORLD_HEIGHT = 4

        # reward for each step
        self.REWARD = -1

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.ACTIONS = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state (y, x)
        self.START = [3, 0]

        # goal states (y, x)
        self.GOAL = [3, 11]

        self.cliff = [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10]]
        
        # step count
        # self.STEP_CNT = 0

        # max steps
        # self.max_steps = float('inf')

    # the step function
    def step(self, state, action):
        """
        Takes a step in the grid world while following the constraints 
        Arguments:
            state: a tuple specifying the current state, i.e. coordiantes in [y,x] format
            action: an integer denoting either of ther four possible actions
        Returns:
            a tuple of the following -
            next_state: a tuple specifying the next state of the agent
            reward: the reward incured for taking the step
        """
        i, j = state
        if action == self.ACTION_UP:
            next_state = [max(i - 1, 0), j]
        elif action == self.ACTION_LEFT:
            next_state = [i, max(j - 1, 0)]
        elif action == self.ACTION_RIGHT:
            next_state = [i, min(j + 1, self.WORLD_WIDTH - 1)]
        elif action == self.ACTION_DOWN:
            next_state = [min(i + 1, self.WORLD_HEIGHT - 1), j]
        else:
            assert False

        reward = -1

        if (action == self.ACTION_DOWN and i == 2 and 1 <= j <= 10) or (
            action == self.ACTION_RIGHT and state == self.START):
            reward = -100
            next_state = self.START

        return next_state, reward
            
        # # simulates wind effect 80% of the times
        # wind = np.zeros(len(self.WIND))
        # if np.random.binomial(1, self.WIND_PROB) == 1:
        #     wind = self.WIND

        # y, x = state
        # if action == self.ACTION_UP:
        #     y = max(y - 1, 0)
        #     x = int(max(x - wind[y], 0))
        # elif action == self.ACTION_DOWN:
        #     y = min(y + 1, self.WORLD_HEIGHT - 1)
        #     x = int(max(x - wind[y], 0))
        # elif action == self.ACTION_LEFT:
        #     x = int(max(x - 1 - wind[y], 0))
        # elif action == self.ACTION_RIGHT:
        #     x = int(max(min(x + 1 - wind[y], self.WORLD_WIDTH - 1),0))
        # else:
        #     raise ValueError(f'action passed is {action}, but only 0, 1, 2, 3 are accepted')

        # if [y, x] in self.obstacles:
        #     y, x = state
        # if [y, x] == self.GOAL:
        #     reward = 0.0
        # else:
        #     reward = self.REWARD

        # return [y, x], reward