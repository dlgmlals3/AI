import numpy as np
from environment import GraphicDisplay, Env


class ValueIteration:
    def __init__(self , env):
        self.env = env
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        self.discount_factor = 0.9

    def value_iteration(self):
        next_value_table = [[0, 0] * self.env.width for _ in range(self.env.height)]

        for state in self.env.get_all_states():
            if state == [2,2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue
            value_list = []
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((reward + self.discount_factor * next_value))

            next_value_table[state[0]][state[1]] = max(value_list)
        self.value_table = next_value_table

    def get_action(self, state):
        if state == [2, 2]:
            return []
        # 모든 행동에 대해 큐함수(보삼 + (감가율 * 다음사태 가치함수)) 계산
        value_list = []
        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = (reward + self.discount_factor * next_value)
            value_list.append(value)

        # 최대 큐 함수를 가진 행동(복수일 경우 여러개) 반환
        max_idx_list = np.argwhere(value_list == np.amax(value_list))
        action_list = max_idx_list.flatten().tolist()
        return action_list
