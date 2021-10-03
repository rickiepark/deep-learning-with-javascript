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
 * 사용자가 특정 레이블의 샘플 텐서를 추가할 수 있는 웹캠 제어를 위한 데이터셋.
 * 이 객체는 샘플을 두 개의 큰 xs과 ys로 연결합니다.
 */
export class ControllerDataset {
  constructor(numClasses) {
    this.numClasses = numClasses;
  }

  /**
   * 제어 데이터셋에 샘플을 추가합니다.
   * @param {Tensor} example 샘플을 나타내는 텐서. 이미지, 활성화 출력
   *     또는 다른 종류의 텐서도 가능합니다.
   * @param {number} label 샘플의 레이블로 숫자여야 합니다.
   */
  addExample(example, label) {
    // 레이블을 원-핫 인코딩합니다.
    const y = tf.tidy(
        () => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));

    if (this.xs == null) {
      // ControllerDataset이 입력의 메모리를 유지하도록
      // 샘플이 처음 추가될 때 example과 y를 tf.keep()으로 감쌉니다.
      // tf.tidy() 안에서 addExample()이 호출될 때 텐서가 삭제되지 않도록 보장합니다.
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
  }
}
