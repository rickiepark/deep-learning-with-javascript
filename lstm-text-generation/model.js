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
import {TextData} from './data.js';

/**
 * 다음 문자 예측을 위한 모델을 만듭니다.
 * @param {number} sampleLen 샘플 길이: 모델에 입력할 문자 개수
 * @param {number} charSetSize 문자 집합 크기: 고유한 문자 개수
 * @param {number|numbre[]} lstmLayerSizes LSTM 층 크기.
 * @return {tf.Model} 다음 문자 예측 모델. 입력 크기는 `[null, sampleLen, charSetSize]`이고
 *   출력 크기는 `[null, charSetSize]`입니다.
 */
export function createModel(sampleLen, charSetSize, lstmLayerSizes) {
  if (!Array.isArray(lstmLayerSizes)) {
    lstmLayerSizes = [lstmLayerSizes];
  }

  const model = tf.sequential();
  for (let i = 0; i < lstmLayerSizes.length; ++i) {
    const lstmLayerSize = lstmLayerSizes[i];
    model.add(tf.layers.lstm({
      units: lstmLayerSize,
      returnSequences: i < lstmLayerSizes.length - 1,
      inputShape: i === 0 ? [sampleLen, charSetSize] : undefined
    }));
  }
  model.add(
      tf.layers.dense({units: charSetSize, activation: 'softmax'}));

  return model;
}

export function compileModel(model, learningRate) {
  const optimizer = tf.train.rmsprop(learningRate);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  console.log(`학습률 ${learningRate}로 모델을 컴파일합니다.`);
  model.summary();
}

/**
 * 모델을 훈련합니다.
 *
 * @param {tf.Model} model 다음 문자 예측 모델. 입력 크기는
 *   `[null, sampleLen, charSetSize]`이고 출력 크기는 `[null, charSetSize]`로 가정합니다.
 * @param {TextData} textData 훈련에 사용할 TextData 객체
 * @param {number} numEpochs 훈련 에포크 횟수
 * @param {number} examplesPerEpoch 에포크 마다 `textData`에서 뽑을 샘플 개수
 * @param {number} batchSize 훈련 배치 크기
 * @param {number} validationSplit 훈련을 위한 검증 세트 비율
 * @param {tf.CustomCallbackArgs} callbacks `model.fit()`에 사용할 사용자 정의 콜백
 */
export async function fitModel(
    model, textData, numEpochs, examplesPerEpoch, batchSize, validationSplit,
    callbacks) {
  for (let i = 0; i < numEpochs; ++i) {
    const [xs, ys] = textData.nextDataEpoch(examplesPerEpoch);
    await model.fit(xs, ys, {
      epochs: 1,
      batchSize: batchSize,
      validationSplit,
      callbacks
    });
    xs.dispose();
    ys.dispose();
  }
}

/**
 * 다음 문자 예측 모델을 사용해 텍스트를 생성합니다.
 *
 * @param {tf.Model} model 텍스트 생성을 위해 사용할 모델 객체.
 *   입력의 크기는 `[null, sampleLen, charSetSize]`이고
 *   출력의 크기는 `[null, charSetSize]`이라고 가정합니다.
 * @param {number[]} sentenceIndices 시드 문장의 문자 인덱스
 * @param {number} length 생성할 문장 길이
 * @param {number} temperature 온도 값. 0보다 크거나 같고 1보다 작거나 같아야 합니다.
 * @param {(char: string) => Promise<void>} onTextGenerationChar 문자를 생성할 때마다 호출될 콜백
 * @returns {string} 생성한 문장
 */
export async function generateText(
    model, textData, sentenceIndices, length, temperature,
    onTextGenerationChar) {
  const sampleLen = model.inputs[0].shape[1];
  const charSetSize = model.inputs[0].shape[2];

  // 원본 입력을 덮어 쓰지 않습니다.
  sentenceIndices = sentenceIndices.slice();

  let generated = '';
  while (generated.length < length) {
    // 입력 시퀀스를 원-핫 텐서로 인코딩합니다.
    const inputBuffer =
        new tf.TensorBuffer([1, sampleLen, charSetSize]);

    // 시드 문장의 원-핫 인코딩을 만듭니다.
    for (let i = 0; i < sampleLen; ++i) {
      inputBuffer.set(1, 0, i, sentenceIndices[i]);
    }
    const input = inputBuffer.toTensor();

    // Call model.predict()를 호출하여 다음 문자의 확률 값을 얻습니다.
    const output = model.predict(input);

    // 확률 값을 기반으로 랜덤하게 샘플링합니다.
    const winnerIndex = sample(tf.squeeze(output), temperature);
    const winnerChar = textData.getFromCharSet(winnerIndex);
    if (onTextGenerationChar != null) {
      await onTextGenerationChar(winnerChar);
    }

    generated += winnerChar;
    sentenceIndices = sentenceIndices.slice(1);
    sentenceIndices.push(winnerIndex);

    // 메모리 정리
    input.dispose();
    output.dispose();
  }
  return generated;
}

/**
 * 확률을 기반으로 샘플을 뽑습니다.
 *
 * @param {tf.Tensor} probs 예측한 확률 점수. `[charSetSize]` 크기의 1D `tf.Tensor`.
 * @param {tf.Tensor} temperature 샘플링에 사용할 온도(즉, 무작위성 또는 다양성의 수준).
 *   0보다 큰 스칼라 `tf.Tensor`.
 * @returns {number} `[0, charSetSize - 1]` 범위에서 랜덤하게 뽑은 샘플의 인덱스.
 * 인덱스는 0부터 시작합니다.
 */
export function sample(probs, temperature) {
  return tf.tidy(() => {
    const logits = tf.div(tf.log(probs), Math.max(temperature, 1e-6));
    const isNormalized = false;
    // `logits`은 `temperature`로 스케일이 조정된 다항 분포입니다.
    // 이 분포에서 랜덤하게 샘플을 뽑습니다.
    return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
  });
}
