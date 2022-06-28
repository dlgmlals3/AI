import numpy as np
import random
from collections import defaultdict
import gym

class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def learn(self, state, action, reward, next_state):
        state, next_state = str(state), str(next_state)
        current_q = self.q_table[state][action]
        max_next_state_q = max(self.q_table[next_state])
        td = reward + self.discount_factor * max_next_state_q - current_q
        new_q = current_q + self.step_size * td
        self.q_table[state][action] = new_q

    # 큐함수에 의거하여 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state = str(state)
            q_list = self.q_table[state]
            action = arg_max(q_list)
        return action

# 큐함수의 값에 따라 최적의 행동을 반환
def arg_max(q_list):
    max_idx_list = np.argwhere(q_list == np.amax(q_list))
    max_idx_list = max_idx_list.flatten().tolist()
    return random.choice(max_idx_list)

if __name__ == "__main__":
    env = gym.make('Taxi-v3')
    render = True
    action_size = 6 # 0 south, 1 north, 2 east, 3 west, 4 pickup, 5 drop
    agent = QLearningAgent(actions=list(range(action_size)))

    for episode in range(5000):
        state = env.reset()
        #print('episode {} start'.format(episode))
        while True:
            #env.render()
            action = agent.get_action(state)
            next_state, reward, done, what = env.step(action)
            next_action = agent.get_action(next_state)
            agent.learn(state, action, reward, next_state)
            state = next_state
            if done:
                print("{} reward:{} {}".format(episode, reward, list(agent.q_table.items())))
                break