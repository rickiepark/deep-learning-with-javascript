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

export function createDeepQNetwork(h, w, numActions) {
  if (!(Number.isInteger(h) && h > 0)) {
    throw new Error(`높이는 양수여야 합니다. 현재 입력된 값: ${h}`);
  }
  if (!(Number.isInteger(w) && w > 0)) {
    throw new Error(`너비이는 양수여야 합니다. 현재 입력된 값: ${w}`);
  }
  if (!(Number.isInteger(numActions) && numActions > 1)) {
    throw new Error(
        `numActions는 1보다 큰 정수여야 합니다. ` +
        `현재 입력된 값: ${numActions}`);
  }

  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    strides: 1,
    activation: 'relu',
    inputShape: [h, w, 2]
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.conv2d({
    filters: 256,
    kernelSize: 3,
    strides: 1,
    activation: 'relu'
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.conv2d({
    filters: 256,
    kernelSize: 3,
    strides: 1,
    activation: 'relu'
  }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units: 100, activation: 'relu'}));
  model.add(tf.layers.dropout({rate: 0.25}));
  model.add(tf.layers.dense({units: numActions}));

  return model;
}

/**
 * 심층 Q 네트워크의 가중치를 다른 네트워크로 복사합니다.
 *
 * @param {tf.LayersModel} destNetwork 가중치를 복사할 목표 네트워크
 * @param {tf.LayersModel} srcNetwork 가중치를 복사할 소스 네트워크
 */
export function copyWeights(destNetwork, srcNetwork) {
  // https://github.com/tensorflow/tfjs/issues/1807:
  // 두 `LayersModel` 객체의 trainable 속성이 동일하지 않으면 가중치 순서가 맞지 않습니다.
  let originalDestNetworkTrainable;
  if (destNetwork.trainable !== srcNetwork.trainable) {
    originalDestNetworkTrainable = destNetwork.trainable;
    destNetwork.trainable = srcNetwork.trainable;
  }

  destNetwork.setWeights(srcNetwork.getWeights());

  // 두 `LayersModel` 객체의 trainable 속성이 동일하지 않으면 가중치 순서가 맞지 않습니다.
  // 두 `LayersModel` 객체의 trainable 속성이 동일하면 `originalDestNetworkTrainable`이
  // null이고 아무런 작업을 할 필요가 없습니다.
  if (originalDestNetworkTrainable != null) {
    destNetwork.trainable = originalDestNetworkTrainable;
  }
}
