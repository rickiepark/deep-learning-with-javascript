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

import {linearRegressionModel, multiLayerPerceptronRegressionModel1Hidden, multiLayerPerceptronRegressionModel2Hidden, multiLayerPerceptronRegressionModel1HiddenNoSigmoid, run} from './index.js';

const statusElement = document.getElementById('status');
export function updateStatus(message) {
  statusElement.innerText = message;
};

const baselineStatusElement = document.getElementById('baselineStatus');
export function updateBaselineStatus(message) {
  baselineStatusElement.innerText = message;
};

export function updateModelStatus(message, modelName) {
  const statElement = document.querySelector(`#${modelName} .status`);
  statElement.innerText = message;
};

const NUM_TOP_WEIGHTS_TO_DISPLAY = 5;
/**
 * 간단한 선형 모델에서 학습된 가중치에 관한 정보를 업데이트합니다.
 *
 * @param {List} weightsList 'value':number 와 'description':string 의 객체 리스트
 */
export function updateWeightDescription(weightsList) {
  const inspectionHeadlineElement =
      document.getElementById('inspectionHeadline');
  inspectionHeadlineElement.innerText =
      `가장 큰 가중치 상위 ${NUM_TOP_WEIGHTS_TO_DISPLAY}개`;
  // 절댓값 크기로 가중치를 정렬합니다.
  weightsList.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  var table = document.getElementById('myTable');
  // 테이블을 초기화합니다.
  table.innerHTML = '';
  // 테이블에 행을 추가합니다.
  weightsList.forEach((weight, i) => {
    if (i < NUM_TOP_WEIGHTS_TO_DISPLAY) {
      let row = table.insertRow(-1);
      let cell1 = row.insertCell(0);
      let cell2 = row.insertCell(1);
      if (weight.value < 0) {
        cell2.setAttribute('class', 'negativeWeight');
      } else {
        cell2.setAttribute('class', 'positiveWeight');
      }
      cell1.innerHTML = weight.description;
      cell2.innerHTML = weight.value.toFixed(4);
    }
  });
};

export async function setup() {
  const trainSimpleLinearRegression = document.getElementById('simple-mlr');
  const trainNeuralNetworkLinearRegression1Hidden =
      document.getElementById('nn-mlr-1hidden');
  const trainNeuralNetworkLinearRegression2Hidden =
      document.getElementById('nn-mlr-2hidden');
  const trainNeuralNetworkLinearRegression1HiddenNoSigmoid =
      document.getElementById('nn-mlr-1hidden-no-sigmoid');

  trainSimpleLinearRegression.addEventListener('click', async (e) => {
    const model = linearRegressionModel();
    await run(model, 'linear', true);
  }, false);

  trainNeuralNetworkLinearRegression1Hidden.addEventListener(
      'click', async () => {
        const model = multiLayerPerceptronRegressionModel1Hidden();
        await run(model, 'oneHidden', false);
      }, false);

  trainNeuralNetworkLinearRegression2Hidden.addEventListener(
      'click', async () => {
        const model = multiLayerPerceptronRegressionModel2Hidden();
        await run(model, 'twoHidden', false);
      }, false);

  trainNeuralNetworkLinearRegression1HiddenNoSigmoid.addEventListener(
      'click', async () => {
        const model = multiLayerPerceptronRegressionModel1HiddenNoSigmoid();
        await run(model, 'nosigHidden', false);
      }, false);
};
