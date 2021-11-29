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

if (typeof tf === 'undefined') {
  global.tf = require('@tensorflow/tfjs');
}

import {assertPositiveInteger, getRandomInteger} from './utils.js';

const DEFAULT_HEIGHT = 16;
const DEFAULT_WIDTH = 16;
const DEFAULT_NUM_FRUITS = 1;
const DEFAULT_INIT_LEN = 4;

export const NO_FRUIT_REWARD = -0.2;
export const FRUIT_REWARD = 10;
export const DEATH_REWARD = -10;

export const ACTION_GO_STRAIGHT = 0;
export const ACTION_TURN_LEFT = 1;
export const ACTION_TURN_RIGHT = 2;

export const ALL_ACTIONS = [ACTION_GO_STRAIGHT, ACTION_TURN_LEFT, ACTION_TURN_RIGHT];
export const NUM_ACTIONS = ALL_ACTIONS.length;

/**
 * 가능한 모든 행동에서 랜덤한 행동 선택하기
 *
 * @return {0 | 1 | 2} 숫자로 표현된 행동
 */
export function getRandomAction() {
  return getRandomInteger(0, NUM_ACTIONS);
}

export class SnakeGame {
  /**
   * SnakeGame의 생성자
   *
   * @param {object} args 게임 설정. 다음 필드를 포함합니다:
   *   - height {number} 보드 높이 (양수)
   *   - width {number} 보드 너비 (양수)
   *   - numFruits {number} 어떤 스텝에서 화면에 나타날 과일 개수
   *   - initLen {number} 스네이크의 초기 길이
   */
  constructor(args) {
    if (args == null) {
      args = {};
    }
    if (args.height == null) {
      args.height = DEFAULT_HEIGHT;
    }
    if (args.width == null) {
      args.width = DEFAULT_WIDTH;
    }
    if (args.numFruits == null) {
      args.numFruits = DEFAULT_NUM_FRUITS;
    }
    if (args.initLen == null) {
      args.initLen = DEFAULT_INIT_LEN;
    }

    assertPositiveInteger(args.height, 'height');
    assertPositiveInteger(args.width, 'width');
    assertPositiveInteger(args.numFruits, 'numFruits');
    assertPositiveInteger(args.initLen, 'initLen');

    this.height_ = args.height;
    this.width_ = args.width;
    this.numFruits_ = args.numFruits;
    this.initLen_ = args.initLen;

    this.reset();
  }

  /**
   * 게임 상태를 리셋합니다.
   *
   * @return {object} 게임의 초기 상태
   *   자세한 내용은 `getState()` 메서드를 참고하세요.
   */
  reset() {
    this.initializeSnake_();
    this.fruitSquares_ = null;
    this.makeFruits_();
    return this.getState();
  }

  /**
   * 게임의 한 스텝을 플레이합니다.
   *
   * @param {0 | 1 | 2 | 3} action 현새 스텝에서 선택한 행동
   *   값의 의미:
   *     0 - 왼쪽
   *     1 - 위쪽
   *     2 - 오른쪽
   *     3 - 아래쪽
   * @return {object} 다음 키를 가진 객체:
   *   - `reward` {number} 보상 값
   *     - 0: 이 스텝에서 과일을 먹지 못했습니다.
   *     - 1: 이 스텝에서 과일을 먹었습니다.
   *   - `state`: 이 스텝 다음의 게임 상태
   *   - `fruitEaten` {boolean} 이 스텝에서 과일을 먹었는지 여부
   *   - `done` {boolean} 이 스텝 다음에 게임이 종료되는지 여부
   *     스네이크 머리가 보드 밖으로 나가거나 자신의 몸통과 부딪히면 게임이 끝납니다.
   */
  step(action) {
    const [headY, headX] = this.snakeSquares_[0];

    // 새로운 머리 좌표를 계산하고 보드 밖으로 나갔는지 확인합니다.
    let done;
    let newHeadY;
    let newHeadX;

    this.updateDirection_(action);
    if (this.snakeDirection_ === 'l') {
      newHeadY = headY;
      newHeadX = headX - 1;
      done = newHeadX < 0;
    } else if (this.snakeDirection_ === 'u') {
      newHeadY = headY - 1;
      newHeadX = headX;
      done = newHeadY < 0
    } else if (this.snakeDirection_ === 'r') {
      newHeadY = headY;
      newHeadX = headX + 1;
      done = newHeadX >= this.width_;
    } else if (this.snakeDirection_ === 'd') {
      newHeadY = headY + 1;
      newHeadX = headX;
      done = newHeadY >= this.height_;
    }

    // 머리가 몸통에 부딪혔는지 확인합니다.
    for (let i = 1; i < this.snakeSquares_.length; ++i) {
      if (this.snakeSquares_[i][0] === newHeadY &&
          this.snakeSquares_[i][1] === newHeadX) {
        done = true;
      }
    }

    let fruitEaten = false;
    if (done) {
      return {reward: DEATH_REWARD, done, fruitEaten};
    }

    // 스네이크의 위치를 업데이트합니다.
    this.snakeSquares_.unshift([newHeadY, newHeadX]);

    // 과일을 먹었는지 확인합니다.
    let reward = NO_FRUIT_REWARD;
    for (let i = 0; i < this.fruitSquares_.length; ++i) {
      const fruitYX = this.fruitSquares_[i];
      if (fruitYX[0] === newHeadY && fruitYX[1] === newHeadX) {
        reward = FRUIT_REWARD;
        fruitEaten = true;
        this.fruitSquares_.splice(i, 1);
        this.makeFruits_();
        break;
      }
    }
    if (!fruitEaten) {
      // 과일을 먹지 않았다면 꼬리를 하나 줄입니다.
      this.snakeSquares_.pop();
    }

    const state = this.getState();
    return {reward, state, done, fruitEaten};
  }

  updateDirection_(action) {
    if (this.snakeDirection_ === 'l') {
      if (action === ACTION_TURN_LEFT) {
        this.snakeDirection_ = 'd';
      } else if (action === ACTION_TURN_RIGHT) {
        this.snakeDirection_ = 'u';
      }
    } else if (this.snakeDirection_ === 'u') {
      if (action === ACTION_TURN_LEFT) {
        this.snakeDirection_ = 'l';
      } else if (action === ACTION_TURN_RIGHT) {
        this.snakeDirection_ = 'r';
      }
    } else if (this.snakeDirection_ === 'r') {
      if (action === ACTION_TURN_LEFT) {
        this.snakeDirection_ = 'u';
      } else if (action === ACTION_TURN_RIGHT) {
        this.snakeDirection_ = 'd';
      }
    } else if (this.snakeDirection_ === 'd') {
      if (action === ACTION_TURN_LEFT) {
        this.snakeDirection_ = 'r';
      } else if (action === ACTION_TURN_RIGHT) {
        this.snakeDirection_ = 'l';
      }
    }
  }

  /**
   * 스네이크의 현재 방향을 반환합니다.
   *
   * @returns {'l' | 'u' | 'r' | 'd'} 스네이크의 현재 방향
   */
  get snakeDirection() {
    return this.snakeDirection_;
  }

  initializeSnake_() {
    /**
     * @private {Array<[number, number]>} 스네이크가 현재 점유한 사각형
     *
     * 각 원소는 사각형의 [y, x] 좌표를 나타내는 길이가 2인 배열입니다.
     * 배열의 첫 번째 원소는 스네이크의 머리이고 마지막은 꼬리입니다.
     */
    this.snakeSquares_ = [];

    // 스네이크는 일직선이고 세로 방향으로 놓인 상태로 시작합니다.
    const y = getRandomInteger(0, this.height_);
    let x = getRandomInteger(this.initLen_ - 1, this.width_);
    for (let i = 0; i < this.initLen_; ++i) {
      this.snakeSquares_.push([y, x - i]);
    }

    /**
     * 현재 스테이크 방향 {'l' | 'u' | 'r' | 'd'}.
     *
     * 스네이크는 일직선이고 세로 방향으로 놓인 상태로 시작합니다.
     * 초기 방향은 항상 오른쪽입니다.
     */
    this.snakeDirection_ = 'r';
  }

  /**
   * 랜덤한 위치에서 새로운 과일을 생성합니다.
   *
   * 생성된 과일 개수는 생성자에서 지정한 numFruits와 같습니다.
   *
   * 과일은 비어있는 사각형에 생성됩니다.
   */
  makeFruits_() {
    if (this.fruitSquares_ == null) {
      this.fruitSquares_ = [];
    }
    const numFruits = this.numFruits_ - this.fruitSquares_.length;
    if (numFruits <= 0) {
      return;
    }

    const emptyIndices = [];
    for (let i = 0; i < this.height_; ++i) {
      for (let j = 0; j < this.width_; ++j) {
	      emptyIndices.push(i * this.width_ + j);
      }
    }

    // 비어있는 인덱스에서 스네이크가 점유한 사각형을 제거합니다.
    const occupiedIndices = [];
    this.snakeSquares_.forEach(yx => {
      occupiedIndices.push(yx[0] * this.width_ + yx[1]);
    });
    occupiedIndices.sort((a, b) => a - b);
    for (let i = occupiedIndices.length - 1; i >= 0; --i) {
      emptyIndices.splice(occupiedIndices[i], 1);
    }

    for (let i = 0; i < numFruits; ++i) {
      const fruitIndex = emptyIndices[getRandomInteger(0, emptyIndices.length)];
      const fruitY = Math.floor(fruitIndex / this.width_);
      const fruitX = fruitIndex % this.width_;
      this.fruitSquares_.push([fruitY, fruitX]);
      if (numFruits > 1) {
	      emptyIndices.splice(emptyIndices.indexOf(fruitIndex), 1);
      }
    }
  }

  get height() {
    return this.height_;
  }

  get width() {
    return this.width_;
  }

  /**
   * 게임 상태의 자바스크립트 표현
   *
   * @return 두 개의 키를 가진 객체:
   *   - s: {Array<[number, number]>} 스네이크가 점유한 사각형
   *        배열의 첫 번째 원소는 스네이크의 머리이고 마지막은 꼬리입니다.
   *   - f: {Array<[number, number]>} 과일이 점유한 사각형
   */
  getState() {
    return {
      "s": this.snakeSquares_.slice(),
      "f": this.fruitSquares_.slice()
    }
  }
}

/**
 * 게임의 현재 상태를 이미지 텐서로 반환합니다.
 *
 * @param {object | object[]} state `SnakeGame.getState()`가 반환한 상태 객체.
 *   두 개의 키를 가집니다: `s`는 스네이크이고 `f`는 과일입니다.
 *   또한 상태 객체의 배열일 수도 있습니다.
 * @param {number} h 높이
 * @param {number} w 너비
 * @return {tf.Tensor} [numExamples, height, width, 2] 크기의 `float32` 텐서
 *   - 첫 번째 채널은 스네이크를 표시하기 위해 0-1-2 값을 사용합니다.
 *     - 0: 빈 사각형
 *     - 1: 스네이크 몸통
 *     - 2: 스네이크 머리
 *   - 두 번째 채널은 과일을 표시하기 위해 0-1 값을 사용합니다.
 *   - `state` 매개변수가 하나의 객체이거나 객체의 배열이면 `numExamples`는 1입니다.
 *     그렇지 않으면 상태 객체 배열의 길이와 같습니다.
 */
export function getStateTensor(state, h, w) {
  if (!Array.isArray(state)) {
    state = [state];
  }
  const numExamples = state.length;
  const buffer = tf.buffer([numExamples, h, w, 2]);

  for (let n = 0; n < numExamples; ++n) {
    if (state[n] == null) {
      continue;
    }
    // 스네이크를 표시합니다.
    state[n].s.forEach((yx, i) => {
      buffer.set(i === 0 ? 2 : 1, n, yx[0], yx[1], 0);
    });

    // 과일을 표시합니다.
    state[n].f.forEach(yx => {
      buffer.set(1, n, yx[0], yx[1], 1);
    });
  }
  return buffer.toTensor();
}
