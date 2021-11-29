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

/** DQN 훈련을 위한 재생 버퍼 */
export class ReplayMemory {
  /**
   * ReplayMemory의 생성자
   *
   * @param {number} maxLen 최대 버퍼 길이
   */
  constructor(maxLen) {
    this.maxLen = maxLen;
    this.buffer = [];
    for (let i = 0; i < maxLen; ++i) {
      this.buffer.push(null);
    }
    this.index = 0;
    this.length = 0;

    this.bufferIndices_ = [];
    for (let i = 0; i < maxLen; ++i) {
      this.bufferIndices_.push(i);
    }
  }

  /**
   * 재생 버퍼에 항목 추가하기
   *
   * @param {any} item 추가할 항목
   */
  append(item) {
    this.buffer[this.index] = item;
    this.length = Math.min(this.length + 1, this.maxLen);
    this.index = (this.index + 1) % this.maxLen;
  }

  /**
   * 재생 버퍼에서 랜덤하게 배치를 샘플링합니다.
   *
   * 샘플링은 중복을 허용하지 않습니다.
   *
   * @param {number} batchSize 배치 크기
   * @return {Array<any>} 샘플링된 항목
   */
  sample(batchSize) {
    if (batchSize > this.maxLen) {
      throw new Error(
          `배치 크기(${batchSize})가 버퍼 길이(${this.maxLen})를 초과합니다.`);
    }
    tf.util.shuffle(this.bufferIndices_);

    const out = [];
    for (let i = 0; i < batchSize; ++i) {
      out.push(this.buffer[this.bufferIndices_[i]]);
    }
    return out;
  }
}
