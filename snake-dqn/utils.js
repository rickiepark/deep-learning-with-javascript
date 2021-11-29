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

/**
 * min보다 크거나 같고 max보다 작은 랜덤한 정수를 생성합니다.
 *
 * @param {number} min 하한값. 포함.
 * @param {number} max 상한값. 미포함.
 * @return {number} 랜덤한 정수
 */
export function getRandomInteger(min, max) {
  // 성능을 최적화하기 위해 getRandomIntegers()` 함수를 사용하지 않습니다.
  return Math.floor((max - min) * Math.random()) + min;
}

/**
 * min보다 크거나 같고 max보다 작은 랜덤한 정수를 생성합니다.
 *
 * @param {number} min 하한값. 포함.
 * @param {number} max 상한값. 미포함.
 * @param {number} numIntegers 생성할 정수 개수
 * @return {number[]} 랜덤한 정수 배열
 */
export function getRandomIntegers(min, max, numIntegers) {
  const output = [];
  for (let i = 0; i < numIntegers; ++i) {
    output.push(Math.floor((max - min) * Math.random()) + min);
  }
  return output;
}


export function assertPositiveInteger(x, name) {
  if (!Number.isInteger(x)) {
    throw new Error(
        `${name}는 정수여야 합니다. 현재 값: ${x}`);
  }
  if (!(x > 0)) {
    throw new Error(
        `${name}는 양수여야 합니다. 현재 값: ${x}`);
  }
}
