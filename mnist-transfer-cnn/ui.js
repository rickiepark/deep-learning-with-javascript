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

import * as util from './util.js';

export function status(statusText, statusColor) {
  console.log(statusText);
  document.getElementById('status').textContent = statusText;
  document.getElementById('status').style.color = statusColor;
}

export function prepUI(predict, retrain, testExamples, imageSize) {
  setPredictFunction(predict, testExamples, imageSize);
  const imageInput = document.getElementById('image-input');
  imageInput.value = util.imageVectorToText(testExamples['5_1'], imageSize);
  predict(imageInput.value);
  setRetrainFunction(retrain);
  document.getElementById('retrain').disabled = false;
  document.getElementById('test-image-select').disabled = false;
}

export function getImageInput() {
  return document.getElementById('image-input').value;
}

export function getEpochs() {
  return Number.parseInt(document.getElementById('epochs').value);
}

function setPredictFunction(predict, testExamples, imageSize) {
  const imageInput = document.getElementById('image-input');
  imageInput.addEventListener('keyup', () => {
    const result = predict(imageInput.value);
  });

  const testImageSelect = document.getElementById('test-image-select');
  testImageSelect.addEventListener('change', () => {
    imageInput.value =
        util.imageVectorToText(testExamples[testImageSelect.value], imageSize);
    predict(imageInput.value);
  });
}

function setRetrainFunction(retrain) {
  const retrainButton = document.getElementById('retrain');
  retrainButton.addEventListener('click', async () => {
    document.getElementById('retrain').disabled = true;
    await retrain();
  });
}

export function getProgressBarCallbackConfig(epochs) {
  // 에포크 끝에서 진행 막대를 업데이트하기 위한 사용자 정의 콜백
  const trainProg = document.getElementById('trainProg');
  let beginMillis;
  const progressBarCallbackConfig = {
    onTrainBegin: async (logs) => {
      beginMillis = tf.util.now();
      status(
          '모델이 재훈련되는 동안 다른 것을 클릭하지 말고 기다리세요...',
          'blue');
      trainProg.value = 0;
    },
    onTrainEnd: async (logs) => {
      // 다시 재훈련 버튼을 활성화시킵니다.
      document.getElementById('retrain').disabled = false;
      status(
          `${epochs} 에포크 재훈련 완료 (소요시간: ` +
              `${(tf.util.now() - beginMillis).toFixed(1)} ms` +
              `). 대기중.`,
          'black');
    },
    onEpochEnd: async (epoch, logs) => {
      status(
          `모델이 재훈련되는 동안 다른 것을 클릭하지 말고 ` +
          `기다리세요... (에포크 ${epoch + 1} / ${epochs})`);
      trainProg.value = (epoch + 1) / epochs * 100;
    },
  };
  return progressBarCallbackConfig;
}

export function setPredictError(text) {
  const predictHeader = document.getElementById('predict-header');
  const predictValues = document.getElementById('predict-values');
  predictHeader.innerHTML = '<td>Error:&nbsp;' + text + '</td>';
  predictValues.innerHTML = '';
}

export function setPredictResults(predictOut, winner) {
  const predictHeader = document.getElementById('predict-header');
  const predictValues = document.getElementById('predict-values');

  predictHeader.innerHTML =
      '<td>5</td><td>6</td><td>7</td><td>8</td><td>9</td>';
  let valTds = '';
  for (const predictVal of predictOut) {
    const valTd = '<td>' + predictVal.toFixed(6) + '</td>';
    valTds += valTd;
  }
  predictValues.innerHTML = valTds;
  document.getElementById('winner').textContent = winner;
}

export function disableLoadModelButtons() {
  document.getElementById('load-pretrained-remote').style.display = 'none';
  document.getElementById('load-pretrained-local').style.display = 'none';
}

export function getTrainingMode() {
  return document.getElementById('training-mode').value;
}
