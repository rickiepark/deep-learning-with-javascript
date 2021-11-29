/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

/**
 * 숫자 배열의 평균을 계산합니다.
 *
 * @param {number[]} xs
 * @returns {number} `xs`의 평균
 */
export function mean(xs) {
  return sum(xs) / xs.length;
}

/**
 * 숫자 배열의 합을 계산합니다.
 *
 * @param {number[]} xs
 * @returns {number} `xs`의 합
 * @throws `xs`가 비어있으면 에러가 발생합니다.
 */
export function sum(xs) {
  if (xs.length === 0) {
    throw new Error('xs가 비어있습니다.');
  } else {
    return xs.reduce((x, prev) => prev + x);
  }
}
