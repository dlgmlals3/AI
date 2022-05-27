import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque

from skimage.color import rgb2gray
from skimage.transform import resize

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Dense, Flatten



# 상태가 입력, 큐함수가 출력인 인공신경망 생성
class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                            input_shape=state_size)
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.fc = Dense(512, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        # 1 * 84 * 84 * 4
        x = self.conv1(x) # 1 * 20 * 20 * 32
        x = self.conv2(x) # 1 * 9 * 9 * 64
        x = self.conv3(x) # 1 * 7 * 7 * 64
        x = self.flatten(x) # 1 * 3136
        x = self.fc(x)
        q = self.fc_out(x)
        return q


# 브레이크아웃 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, action_size, state_size=(84, 84, 4)):
        self.render = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.02
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = self.epsilon_start - self.epsilon_end
        self.epsilon_decay_step /= self.exploration_steps
        self.batch_size = 32
        self.train_start = 5000 # start training 50000 기존값 실행
        self.update_target_rate = 10000

        # 리플레이 메모리, 최대 크기 100,000
        self.memory = deque(maxlen=100000)
        # 게임 시작 후 랜덤하게 움직이지 않는 것에 대한 옵션
        self.no_op_steps = 30

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)
        self.optimizer = Adam(self.learning_rate, clipnorm=10.)
        # 타깃 모델 초기화
        self.update_target_model()
        self.avg_q_max, self.avg_loss = 0, 0
        self.model_path = os.path.join(os.getcwd(), 'save_model', 'model')

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(history)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))



    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        batch = random.sample(self.memory, self.batch_size)
        history = np.array([sample[0][0] / 255. for sample in batch],dtype=np.float32)
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_history = np.array([sample[3][0] / 255. for sample in batch],dtype=np.float32)
        dones = np.array([sample[4] for sample in batch])

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predicts = self.model(history)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)
            target_predicts = self.target_model(next_history)
            max_q = np.amax(target_predicts, axis=1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q

            error = tf.abs(targets - predicts)
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
            self.avg_loss += loss.numpy()

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    # 환경과 DQN 에이전트 생성
    env = gym.make('Breakout-v4')
    agent = DQNAgent(action_size=3)
    global_step = 0
    score_avg = 0
    score_max = 0

    action_dict = {0:1, 1:2, 2:3}
    num_episode = 50000
    for e in range(num_episode):
        done = False
        dead = False
        step, score, start_life = 0, 0, 5
        observe = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            global_step += 1
            step += 1
            action = agent.get_action(history)
            real_action = action_dict[action]
            if dead:
                action, real_action, dead = 0, 1, False
            observe, reward, done, info = env.step(real_action)
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)
            agent.avg_q_max += np.amax(agent.model(np.float32(history / 255.))[0])
            if start_life > info['lives']:
                dead = True
                start_life = info['lives']

            score += reward
            reward = np.clip(reward, -1., 1.)
            agent.append_sample(history, action, reward, next_history, dead)
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()
            if dead:
                history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                history = np.reshape([history], (1, 84, 84, 4))
            else:
                history = next_history
            if done:
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                score_max = score if score > score_max else score_max

                log = "episode: {:5d} | ".format(e)
                log += "score: {:4.1f} | ".format(score)
                log += "score max : {:4.1f} | ".format(score_max)
                log += "score avg: {:4.1f} | ".format(score_avg)
                log += "memory length: {:5d} | ".format(len(agent.memory))
                log += "epsilon: {:.3f} | ".format(agent.epsilon)
                log += "q avg : {:3.2f} | ".format(agent.avg_q_max / float(step))
                log += "avg loss : {:3.2f}".format(agent.avg_loss / float(step))
                print(log)

                agent.avg_q_max, agent.avg_loss = 0, 0

        if e % 1000 == 0:
            agent.model.save_weights("./save_model/model", save_format="tf")
