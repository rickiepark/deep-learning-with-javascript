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
 * 온도 예측 문제를 위해 `tf.LayersModel`을 만들고 훈련합니다.
 */

import {JenaWeatherData} from './data.js';

// 훈련과 검증 세트를 위한 행 범위
const TRAIN_MIN_ROW = 0;
const TRAIN_MAX_ROW = 200000;
const VAL_MIN_ROW = 200001;
const VAL_MAX_ROW = 300000;

/**
 * 온도 예측 정확도의 일반적인 기준을 계산합니다.
 *
 * 온도 특성의 마지막 값을 예측으로 사용합니다.
 *
 * @param {boolean} normalize 훈련에 정규화된 데이터를 사용할지 여부
 * @param {boolean} includeDateTime 훈련데 날짜와 시간 특성을 포함할지 여부
 * @param {number} lookBack 룩백 타임 스텝 횟수
 * @param {number} step 입력 특성을 생성하기 위해 사용할 스텝 크기
 * @param {number} delay 예측할 미래 타임 스텝 수
 * @returns {number} 기준 예측에 대한 평균 절댓값 오차
 */
export async function getBaselineMeanAbsoluteError(
    jenaWeatherData, normalize, includeDateTime, lookBack, step, delay) {
  const batchSize = 128;
  const dataset = tf.data.generator(
      () => jenaWeatherData.getNextBatchFunction(
          false, lookBack, delay, batchSize, step, VAL_MIN_ROW, VAL_MAX_ROW,
          normalize, includeDateTime));

  const batchMeanAbsoluteErrors = [];
  const batchSizes = [];
  await dataset.forEachAsync(dataItem => {
    const features = dataItem.xs;
    const targets = dataItem.ys;
    const timeSteps = features.shape[1];
    batchSizes.push(features.shape[0]);
    batchMeanAbsoluteErrors.push(tf.tidy(
        () => tf.losses.absoluteDifference(
            targets,
            features.gather([timeSteps - 1], 1).gather([1], 2).squeeze([2]))));
  });

  const meanAbsoluteError = tf.tidy(() => {
    const batchSizesTensor = tf.tensor1d(batchSizes);
    const batchMeanAbsoluteErrorsTensor = tf.stack(batchMeanAbsoluteErrors);
    return batchMeanAbsoluteErrorsTensor.mul(batchSizesTensor)
        .sum()
        .div(batchSizesTensor.sum());
  });
  tf.dispose(batchMeanAbsoluteErrors);
  return meanAbsoluteError.dataSync()[0];
}

/**
 * 온도 예측 문제를 위한 선형 회귀 모델을 만듭니다.
 *
 * @param {tf.Shape} inputShape (배치 차원이 없는) 입력 크기
 * @returns {tf.LayersModel} TensorFlow.js의 tf.LayersModel 객체
 */
function buildLinearRegressionModel(inputShape) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape}));
  model.add(tf.layers.dense({units: 1}));
  return model;
}

/**
 * 온도 예측 문제를 위한 MLP 모델을 만듭니다.
 *
 * @param {tf.Shape} inputShape (배치 차원이 없는) 입력 크기
 * @param {tf.regularizer.Regularizer} kernelRegularizer (옵션 매개변수)
 *   MLP의 첫 번째 (은닉) 밀집층의 커널에 적용할 규제.
 *   지정하지 않으면 MLP에 가중치 규제를 적용하지 않습니다.
 * @param {number} dropoutRate MLP의 두 개의 은닉층 사이에 적용할 드롭아웃 층의 드롭아웃 비율.
 *   지정하지 않으면 MLP에 드롭아웃 층이 추가되지 않습니다.
 * @returns {tf.LayersModel} TensorFlow.js의 tf.LayersModel 객체.
 */
export function buildMLPModel(inputShape, kernelRegularizer, dropoutRate) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape}));
  model.add(
      tf.layers.dense({units: 32, kernelRegularizer, activation: 'relu'}));
  if (dropoutRate > 0) {
    model.add(tf.layers.dropout({rate: dropoutRate}));
  }
  model.add(tf.layers.dense({units: 1}));
  return model;
}

/**
 * 온도 예측 문제를 위한 간단한 RNN 모델을 만듭니다.
 *
 * @param {tf.Shape} inputShape (배치 차원이 없는) 입력 크기
 * @returns {tf.LayersModel} simpleRNN 층으로 구성된 TensorFlow.js 모델
 */
export function buildSimpleRNNModel(inputShape) {
  const model = tf.sequential();
  const rnnUnits = 32;
  model.add(tf.layers.simpleRNN({units: rnnUnits, inputShape}));
  model.add(tf.layers.dense({units: 1}));
  return model;
}

/**
 * 온도 예측 문제를 위한 GRU 모델을 만듭니다.
 *
 * @param {tf.Shape} inputShape (배치 차원이 없는) 입력 크기
 * @param {number} dropout (옵션 매개변수) 입력 드롭아웃 비율
 * @param {number} recurrentDropout (옵션 매개변수) 순환 드롭아웃 비율
 * @returns {tf.LayersModel} TensorFlow.js GRU 모델
 */
export function buildGRUModel(inputShape, dropout, recurrentDropout) {
  const model = tf.sequential();
  const rnnUnits = 32;
  model.add(tf.layers.gru({
    units: rnnUnits,
    inputShape,
    dropout: dropout || 0,
    recurrentDropout: recurrentDropout || 0
  }));
  model.add(tf.layers.dense({units: 1}));
  return model;
}

/**
 * 온도 예측 문제를 위한 모델을 만듭니다.
 *
 * @param {string} modelType 모델 종류
 * @param {number} numTimeSteps 각 입력 샘플의 타임 스텝 횟수
 * @param {number} numFeatures 타임 스텝마다 특성 개수
 * @returns 컴파일된 `tf.LayersModel` 객체
 */
export function buildModel(modelType, numTimeSteps, numFeatures) {
  const inputShape = [numTimeSteps, numFeatures];

  console.log(`modelType = ${modelType}`);
  let model;
  if (modelType === 'mlp') {
    model = buildMLPModel(inputShape);
  } else if (modelType === 'mlp-l2') {
    model = buildMLPModel(inputShape, tf.regularizers.l2());
  } else if (modelType === 'linear-regression') {
    model = buildLinearRegressionModel(inputShape);
  } else if (modelType === 'mlp-dropout') {
    const regularizer = null;
    const dropoutRate = 0.25;
    model = buildMLPModel(inputShape, regularizer, dropoutRate);
  } else if (modelType === 'simpleRNN') {
    model = buildSimpleRNNModel(inputShape);
  } else if (modelType === 'gru') {
    model = buildGRUModel(inputShape);
  } else {
    throw new Error(`지원하지 않는 모델 타입입니다: ${modelType}`);
  }

  model.compile({loss: 'meanAbsoluteError', optimizer: 'rmsprop'});
  model.summary();
  return model;
}

/**
 * 예나 날씨 데이터에서 모델을 훈련합니다.
 *
 * @param {tf.LayersModel} model 컴파일된 tf.LayersModel 객체.
 *   `[numExamples, timeSteps, numFeatures]` 크기의 3차원 입력과
 *   온도 예측을 위해 `[numExamples, 1]` 크기의 출력을 기대합니다.
 * @param {JenaWeatherData} jenaWeatherData JenaWeatherData 객체.
 * @param {boolean} normalize 훈련에 정규화된 데이터를 사용할지 여부
 * @param {boolean} includeDateTime 훈련에 날짜와 시간 특성을 포함할지 여부
 * @param {number} lookBack 룩백 타임 스텝 횟수
 * @param {number} step 입력 특성을 생성하는데 사용할 스텝 크기
 * @param {number} delay 예측할 미래 타임 스텝 수
 * @param {number} batchSize 훈련 배치 크기
 * @param {number} epochs 훈련 에포크 수
 * @param {tf.Callback | tf.CustomCallbackArgs} customCallback 에포크 종료마다 호출할 콜백.
 *   `onBatchEnd`와 `onEpochEnd` 필드를 포함할 수 있습니다.
 */
export async function trainModel(
    model, jenaWeatherData, normalize, includeDateTime, lookBack, step, delay,
    batchSize, epochs, customCallback) {
  const trainShuffle = true;
  const trainDataset =
      tf.data
          .generator(
              () => jenaWeatherData.getNextBatchFunction(
                  trainShuffle, lookBack, delay, batchSize, step, TRAIN_MIN_ROW,
                  TRAIN_MAX_ROW, normalize, includeDateTime))
          .prefetch(8);
  const evalShuffle = false;
  const valDataset = tf.data.generator(
      () => jenaWeatherData.getNextBatchFunction(
          evalShuffle, lookBack, delay, batchSize, step, VAL_MIN_ROW,
          VAL_MAX_ROW, normalize, includeDateTime));

  await model.fitDataset(trainDataset, {
    batchesPerEpoch: 500,
    epochs,
    callbacks: customCallback,
    validationData: valDataset
  });
}
