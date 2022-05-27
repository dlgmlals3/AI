import copy
import pylab
import random
import numpy as np
from environment import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# 딥살사 인공신경망
class DeepSARSA(tf.keras.Model):
    def __init__(self, action_size):
        super(DeepSARSA, self).__init__()
        self.fc1 = Dense(30, activation='relu')
        self.fc2 = Dense(30, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q

# 그리드월드 예제에서의 딥살사 에이전트
class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # 딥살사 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = DeepSARSA(self.action_size)
        self.optimizer = Adam(lr=self.learning_rate)

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model(state)
            return np.argmax(q_values[0])

    # <s, a, r, s', a'>의 샘플로부터 모델 업데이트
    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 학습 파라메터
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predict = self.model(state)[0][action]

            next_q = self.model(next_state)[0][next_action]
            target = reward + (1 - done) * self.discount_factor * next_q

            # MSE 오류 함수 계산
            loss = tf.reduce_mean(tf.square(target - predict))

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


if __name__ == "__main__":
    env = Env(render_speed=0.01)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)
    scores, episodes = [], []
    EPISODES = 1000
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)
            agent.train_model(state, action, reward, next_state, next_action, done)
            score += reward
            state = next_state
            if done:
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(e, score, agent.epsilon))
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.ylabel("score")
                pylab.savefig("./save_graph/graph.png")
        if e % 100 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')
