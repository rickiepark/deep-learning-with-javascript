/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

import {createDeepQNetwork} from './dqn';
import {getRandomAction, SnakeGame, NUM_ACTIONS, ALL_ACTIONS, getStateTensor} from './snake_game';
import {ReplayMemory} from './replay_memory';
import { assertPositiveInteger } from './utils';

export class SnakeGameAgent {
  /**
   * SnakeGameAgent의 생성자
   *
   * @param {SnakeGame} game 게임 객체
   * @param {object} config 다음과 같은 키를 가진 설정 객체
   *   - `replayBufferSize` {number} 재생 메모리 크기. 양수여야 합니다.
   *   - `epsilonInit` {number} (입실론 그리디 알고리즘을 위한) 초기 입실론 값.
   *     0보다 크거나 같고 1보다 작거나 같아야 합니다.
   *   - `epsilonFinal` {number} 최종 입실론 값. 0보다 크거나 같고 1보다 작거나 같아야 합니다.
   *   - `epsilonDecayFrames` {number} `episloInit`에서 `epsilonFinal` 까지
   *     `epsilon` 값을 선형적으로 감소시킬 프레임 수
   *   - `learningRate` {number} 훈련에서 사용할 학습률
   */
  constructor(game, config) {
    assertPositiveInteger(config.epsilonDecayFrames);

    this.game = game;

    this.epsilonInit = config.epsilonInit;
    this.epsilonFinal = config.epsilonFinal;
    this.epsilonDecayFrames = config.epsilonDecayFrames;
    this.epsilonIncrement_ = (this.epsilonFinal - this.epsilonInit) /
        this.epsilonDecayFrames;

    this.onlineNetwork =
        createDeepQNetwork(game.height,  game.width, NUM_ACTIONS);
    this.targetNetwork =
        createDeepQNetwork(game.height,  game.width, NUM_ACTIONS);
    // 타깃 네트워크를 동결합니다: 온라인 네트워크에서 가중치를 업데이트합니다.
    this.targetNetwork.trainable = false;

    this.optimizer = tf.train.adam(config.learningRate);

    this.replayBufferSize = config.replayBufferSize;
    this.replayMemory = new ReplayMemory(config.replayBufferSize);
    this.frameCount = 0;
    this.reset();
  }

  reset() {
    this.cumulativeReward_ = 0;
    this.fruitsEaten_ = 0;
    this.game.reset();
  }

  /**
   * 게임의 한 스텝을 플레이합니다.
   *
   * @returns {number | null} 이 스텝이 게임을 종료시키면 게임의 총 보상이 반환되고
   *   그렇지 않으면 `null`이 반환됩니다.
   */
  playStep() {
    this.epsilon = this.frameCount >= this.epsilonDecayFrames ?
        this.epsilonFinal :
        this.epsilonInit + this.epsilonIncrement_  * this.frameCount;
    this.frameCount++;

    // 입실론-그리디 알고리즘
    let action;
    const state = this.game.getState();
    if (Math.random() < this.epsilon) {
      // 랜덤하게 행동을 선택합니다.
      action = getRandomAction();
    } else {
      // 온라인 DQN의 출력을 기반으로 행동을 선택합니다.
      tf.tidy(() => {
        const stateTensor =
            getStateTensor(state, this.game.height, this.game.width)
        action = ALL_ACTIONS[
            this.onlineNetwork.predict(stateTensor).argMax(-1).dataSync()[0]];
      });
    }

    const {state: nextState, reward, done, fruitEaten} = this.game.step(action);

    this.replayMemory.append([state, action, reward, done, nextState]);

    this.cumulativeReward_ += reward;
    if (fruitEaten) {
      this.fruitsEaten_++;
    }
    const output = {
      action,
      cumulativeReward: this.cumulativeReward_,
      done,
      fruitsEaten: this.fruitsEaten_
    };
    if (done) {
      this.reset();
    }
    return output;
  }

  /**
   * 재생 버퍼에서 랜덤하게 선택한 배치에서 훈련을 수행합니다.
   *
   * @param {number} batchSize 배치 크기
   * @param {number} gamma 보상 할인 계수. 0보다 크거나 같고 1보다 작거나 같아야 합니다.
   * @param {tf.train.Optimizer} optimizer 온라인 네트워크의 가중치를 업데이트하기 위해
   *   사용할 옵티마이저 객체
   */
  trainOnReplayBatch(batchSize, gamma, optimizer) {
    // 재생 버퍼에서 샘플의 배치를 가져옵니다.
    const batch = this.replayMemory.sample(batchSize);
    const lossFunction = () => tf.tidy(() => {
      const stateTensor = getStateTensor(
          batch.map(example => example[0]), this.game.height, this.game.width);
      const actionTensor = tf.tensor1d(
          batch.map(example => example[1]), 'int32');
      const qs = this.onlineNetwork.apply(stateTensor, {training: true})
          .mul(tf.oneHot(actionTensor, NUM_ACTIONS)).sum(-1);

      const rewardTensor = tf.tensor1d(batch.map(example => example[2]));
      const nextStateTensor = getStateTensor(
          batch.map(example => example[4]), this.game.height, this.game.width);
      const nextMaxQTensor =
          this.targetNetwork.predict(nextStateTensor).max(-1);
      const doneMask = tf.scalar(1).sub(
          tf.tensor1d(batch.map(example => example[3])).asType('float32'));
      const targetQs =
          rewardTensor.add(nextMaxQTensor.mul(doneMask).mul(gamma));
      return tf.losses.meanSquaredError(targetQs, qs);
    });

    // 온라인 DQN의 가중치에 대한 손실 함수의 그레이디언트를 계산합니다.
    const grads = tf.variableGrads(lossFunction);
    // 이 그레이디언트를 사용해 온라인 DQN의 가중치를 업데이트합니다.
    optimizer.applyGradients(grads.grads);
    tf.dispose(grads);
  }
}
