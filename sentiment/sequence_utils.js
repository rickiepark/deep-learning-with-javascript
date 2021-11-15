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

/**
 * 순차 데이터를 위한 유틸리티
 */

export const PAD_INDEX = 0;  // 패딩 문자 인덱스
export const OOV_INDEX = 2;  // OOV 문자 인덱스

/**
 * 모든 시퀀스를 같은 길이로 패딩하거나 자릅니다.
 *
 * @param {number[][]} sequences 숫자 배열의 배열로 표현된 시퀀스
 * @param {number} maxLen 최대 길이. `maxLen`보다 긴 시퀀스는 잘리고 짧은 시퀀스는 패딩됩니다.
 * @param {'pre'|'post'} padding 패딩 타입
 * @param {'pre'|'post'} truncating 잘림 타입
 * @param {number} value 패딩 값
 */
export function padSequences(
    sequences, maxLen, padding = 'pre', truncating = 'pre', value = PAD_INDEX) {
  return sequences.map(seq => {
    // 시퀀스를 자릅니다.
    if (seq.length > maxLen) {
      if (truncating === 'pre') {
        seq.splice(0, seq.length - maxLen);
      } else {
        seq.splice(maxLen, seq.length - maxLen);
      }
    }

    // 패딩을 추가합니다.
    if (seq.length < maxLen) {
      const pad = [];
      for (let i = 0; i < maxLen - seq.length; ++i) {
        pad.push(value);
      }
      if (padding === 'pre') {
        seq = pad.concat(seq);
      } else {
        seq = seq.concat(pad);
      }
    }

    return seq;
  });
}
