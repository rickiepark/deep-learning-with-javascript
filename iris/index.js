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

import * as data from './data.js';
import * as loader from './loader.js';
import * as ui from './ui.js';

let model;

/**
 * 붓꽃의 품종을 분류하기 위한 `tf.Model`을 훈련합니다.
 *
 * @param xTrain [numTrainExamples, 4] 크기의 `tf.Tensor`인 훈련 특성 데이터
 *   두 번째 차원에 꽃잎 길이, 꽃잎 너비, 꽃받침 길이, 꽃받침 너비 특성이 담겨 있습니다.
 * @param yTrain [numTrainExamples, 3] 크기의 `tf.Tensor`인 원-핫 인코딩된 훈련 레이블
 * @param xTest [numTestExamples, 4] 크기의 `tf.Tensor`인 테스트 특성 데이터
 * @param yTest [numTestExamples, 3] 크기의 `tf.Tensor`인 원-핫 인코딩된 테스트 레이블
 * @returns 훈련된 `tf.Model` 객체
 */
async function trainModel(xTrain, yTrain, xTest, yTest) {
  ui.status('모델을 훈련합니다... 잠시 기다려 주세요.');

  const params = ui.loadTrainParametersFromUI();

  // 두 개의 밀집 층으로 구성된 모델을 정의합니다.
  const model = tf.sequential();
  model.add(tf.layers.dense(
      {units: 10, activation: 'sigmoid', inputShape: [xTrain.shape[1]]}));
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  model.summary();

  const optimizer = tf.train.adam(params.learningRate);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  const trainLogs = [];
  const lossContainer = document.getElementById('lossCanvas');
  const accContainer = document.getElementById('accuracyCanvas');
  const beginMs = performance.now();
  // 모델을 훈련하기 위해 `model.fit` 메서드를 호출합니다.
  const history = await model.fit(xTrain, yTrain, {
    epochs: params.epochs,
    validationData: [xTest, yTest],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // 훈련 에포크가 끝날 때마다 손실과 정확도 값을 출력합니다.
        const secPerEpoch =
            (performance.now() - beginMs) / (1000 * (epoch + 1));
        ui.status(`모델을 훈련합니다... 약 ${
            secPerEpoch.toFixed(4)} 초/에포크`)
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ['loss', 'val_loss'],
            {xLabel:'에포크', yLabel:'손실'})
        tfvis.show.history(accContainer, trainLogs, ['acc', 'val_acc'],
            {xLabel:'에포크', yLabel:'정확도'})
        calculateAndDrawConfusionMatrix(model, xTest, yTest);
      },
    }
  });
  const secPerEpoch = (performance.now() - beginMs) / (1000 * params.epochs);
  ui.status(
      `모델 훈련 완료:  ${secPerEpoch.toFixed(4)} 초/에포크`);
  return model;
}

/**
 * 직접 입력한 붓꽃 데이터로 추론을 실행합니다.
 *
 * @param model 추론을 실행할 `tf.Model` 객체
 */
async function predictOnManualInput(model) {
  if (model == null) {
    ui.setManualInputWinnerMessage('ERROR: 먼저 모델을 로드하거나 훈련하세요.');
    return;
  }

  // `tf.tidy`를 사용함녀 `predict` 메서드가 점유한 WebGL 메모리를 마지막에 해제시켜 줍니다.
  tf.tidy(() => {
    // 2D `tf.Tensor`로 입력 데이터를 준비합니다.
    const inputData = ui.getManualInputData();
    const input = tf.tensor2d([inputData], [1, 4]);

    // `model.predict`를 호출하여 붓꽃에 대한 확률을 출력으로 얻습니다.
    const predictOut = model.predict(input);
    const logits = Array.from(predictOut.dataSync());
    const winner = data.IRIS_CLASSES[predictOut.argMax(-1).dataSync()[0]];
    ui.setManualInputWinnerMessage(winner);
    ui.renderLogitsForManualInput(logits);
  });
}

/**
 * 오차 행렬을 그립니다.
 */
async function calculateAndDrawConfusionMatrix(model, xTest, yTest) {
  const [preds, labels] = tf.tidy(() => {
    const preds = model.predict(xTest).argMax(-1);
    const labels = yTest.argMax(-1);
    return [preds, labels];
  });

  const confMatrixData = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = document.getElementById('confusion-matrix');
  tfvis.render.confusionMatrix(
      container,
      {values: confMatrixData, labels: data.IRIS_CLASSES},
      {shadeDiagonal: true, xLabel:'예측', yLabel:'레이블'},
  );

  tf.dispose([preds, labels]);
}

/**
 * 테스트 붓꽃 데이터에서 추론을 실행합니다.
 *
 * @param model 추론을 실행할 `tf.Model` 객체
 * @param xTest [numTestExamples, 4] 크기의 `tf.Tensor`인 테스트 데이터 특성
 * @param yTest [numTestExamples, 3] 크기의 `tf.Tensor`인 테스트 레이블
 */
async function evaluateModelOnTestData(model, xTest, yTest) {
  ui.clearEvaluateTable();

  tf.tidy(() => {
    const xData = xTest.dataSync();
    const yTrue = yTest.argMax(-1).dataSync();
    const predictOut = model.predict(xTest);
    const yPred = predictOut.argMax(-1);
    ui.renderEvaluateTable(
        xData, yTrue, yPred.dataSync(), predictOut.dataSync());
    calculateAndDrawConfusionMatrix(model, xTest, yTest);
  });

  predictOnManualInput(model);
}

const HOSTED_MODEL_JSON_URL =
    'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json';

/**
 * 붓꽃 데모의 메인 함수
 */
async function iris() {
  const [xTrain, yTrain, xTest, yTest] = data.getIrisData(0.15);

  const localLoadButton = document.getElementById('load-local');
  const localSaveButton = document.getElementById('save-local');
  const localRemoveButton = document.getElementById('remove-local');

  document.getElementById('train-from-scratch')
      .addEventListener('click', async () => {
        model = await trainModel(xTrain, yTrain, xTest, yTest);
        await evaluateModelOnTestData(model, xTest, yTest);
        localSaveButton.disabled = false;
      });

  if (await loader.urlExists(HOSTED_MODEL_JSON_URL)) {
    ui.status('모델 위치: ' + HOSTED_MODEL_JSON_URL);
    const button = document.getElementById('load-pretrained-remote');
    button.addEventListener('click', async () => {
      ui.clearEvaluateTable();
      model = await loader.loadHostedPretrainedModel(HOSTED_MODEL_JSON_URL);
      await predictOnManualInput(model);
      localSaveButton.disabled = false;
    });
  }

  localLoadButton.addEventListener('click', async () => {
    model = await loader.loadModelLocally();
    await predictOnManualInput(model);
  });

  localSaveButton.addEventListener('click', async () => {
    await loader.saveModelLocally(model);
    await loader.updateLocalModelStatus();
  });

  localRemoveButton.addEventListener('click', async () => {
    await loader.removeModelLocally();
    await loader.updateLocalModelStatus();
  });

  await loader.updateLocalModelStatus();

  ui.status('대기중');
  ui.wireUpEvaluateTableCallbacks(() => predictOnManualInput(model));
}

iris();
