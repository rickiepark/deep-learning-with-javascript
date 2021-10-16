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

import * as game from './game.js';

export const lossContainerElement =
    document.getElementById('training-loss-canvas');
export const accuracyContainerElement =
    document.getElementById('training-accuracy-canvas');

const toArrayContainerElement = document.getElementById('to-array-container');
const batchSizeElement = document.getElementById('generator-batch');
const takeElement = document.getElementById('generator-take');
const statusElement = document.getElementById('train-model-message');
const numSimulationsSoFarElement =
    document.getElementById('num-simulations-so-far');
const batchesPerEpochElement = document.getElementById('batches-per-epoch');
const epochsToTrainElement = document.getElementById('epochs-to-train');
const expectedSimulationsElement =
    document.getElementById('expected-simulations');


export const useOneHotElement = document.getElementById('use-one-hot');

export function getBatchSize() {
  return batchSizeElement.valueAsNumber;
}

export function getTake() {
  return takeElement.valueAsNumber;
}

export function getBatchesPerEpoch() {
  return batchesPerEpochElement.valueAsNumber;
}

export function getEpochsToTrain() {
  return epochsToTrainElement.valueAsNumber;
}

export function getInputCards() {
  const cards = [];
  for (let i = 0; i < game.GAME_STATE.num_cards_per_hand; i++) {
    cards.push(document.getElementById(`input-card-${i}`).valueAsNumber);
  }
  return cards;
}

/** 시뮬레이션 카운드를 업데이트합니다. */
export function displayNumSimulationsSoFar() {
  numSimulationsSoFarElement.innerText = game.GAME_STATE.num_simulations_so_far;
}

/** 훈련 결과 메시지를 업데이트 합니다. 훈련 속도를 위해 사용합니다. */
export function displayTrainLogMessage(message) {
  statusElement.innerText = message;
}

/** 모델 훈련을 위해 예상 시뮬레이션 횟수를 출력합니다. */
export function displayExpectedSimulations() {
  const expectedSimulations =
      getBatchSize() * getBatchesPerEpoch() * getEpochsToTrain();
  expectedSimulationsElement.innerText = expectedSimulations;
}

/** 모델의 예측을 출력합니다. */
export function displayPrediction(text) {
  document.getElementById('prediction').innerText = text;
}

/** 게임의 특성과 레이블을 출력하는 헬퍼 함수 */
function featuresAndLabelsToPrettyString(features) {
  const basicArray = [];
  for (const value of features) {
    basicArray.push(value);
  }
  return basicArray;
}

/**
 * 게임 시뮬레이션 테이블에 데이터를 채웁니다.
 * @param {player1, player2, win]} sample  게임 상태.  첫 번째 원소는 플레이어 1의 카드입니다.
 *     두 번째 원소는 플레이어 2의 카드입니다.
 *     세 번째 원소는 플레이어 1이 이기면 1, 그렇지 않으면 0입니다.
 * @param {features, label} featuresAndLabel 모델에 주입하기 적절하게 처리된 샘플 데이터
 */
export function displaySimulation(sample, featuresAndLabel) {
  const player1Row = document.getElementById('player1-row');
  player1Row.innerHTML = '';
  // 플레이어 1 시뮬레이션 데이터
  for (let i = 0; i < game.GAME_STATE.num_cards_per_hand; i++) {
    const newDiv = document.createElement('div');
    newDiv.className = 'divTableCell';
    newDiv.innerText = sample.player1Hand[i];
    player1Row.appendChild(newDiv);
  }

  const player2Row = document.getElementById('player2-row');
  player2Row.innerHTML = '';
  // 플레이어 2 시뮬레이션 데이터
  for (let i = 0; i < game.GAME_STATE.num_cards_per_hand; i++) {
    const newDiv = document.createElement('div');
    newDiv.className = 'divTableCell';
    newDiv.innerText = sample.player2Hand[i];
    player2Row.appendChild(newDiv);
  }

  const resultRow = document.getElementById('result-row');
  resultRow.innerHTML = '';
  // 결과
  const newDiv = document.createElement('div');
  newDiv.className = 'divTableCell';
  newDiv.innerText = sample.player1Win;
  resultRow.appendChild(newDiv);

  const features = featuresAndLabel.xs.dataSync();
  const label = featuresAndLabel.ys.dataSync();
  document.getElementById('sim-features').innerText =
      JSON.stringify(featuresAndLabelsToPrettyString(features));
  document.getElementById('sim-label').innerText = label;
};

/**
 * 생성된 샘플 데이터를 div 원소를 사용해 HTML 테이블을 만듭니다.
 */
export async function displayBatches(arr) {
  toArrayContainerElement.textContent = '';
  let i = 0;
  for (const batch of arr) {
    const oneKeyRow = document.createElement('div');
    oneKeyRow.className = 'divTableRow';
    oneKeyRow.align = 'left';
    const featuresDiv = document.createElement('div');
    const labelDiv = document.createElement('div');
    featuresDiv.className = 'divTableCell';
    labelDiv.className = 'divTableCell';
    featuresDiv.textContent = batch.xs;
    labelDiv.textContent = batch.ys;
    oneKeyRow.appendChild(featuresDiv);
    oneKeyRow.appendChild(labelDiv);
    // updateSampleRowOutput에 div를 추가합니다.
    toArrayContainerElement.appendChild(oneKeyRow);
  }
};

export function updatePredictionInputs() {
  const container = document.getElementById('prediction-input');
  container.innerHTML = '';
  for (let i = 0; i < game.GAME_STATE.num_cards_per_hand; i++) {
    const newH4 = document.createElement('h4');
    newH4.innerText = `카드 ${i} `;
    const newInput = document.createElement('input');
    newInput.type = 'number';
    newInput.id = `input-card-${i}`;
    newInput.value = 13;
    newH4.appendChild(newInput);
    container.appendChild(newH4);
  }
}

export function enableTrainButton() {
  document.getElementById('train-model-using-fit-dataset')
      .removeAttribute('disabled');
}

export function disableTrainButton() {
  document.getElementById('train-model-using-fit-dataset')
      .setAttribute('disabled', true);
}

export function enableStopButton() {
  document.getElementById('stop-training').removeAttribute('disabled');
}

export function disableStopButton() {
  document.getElementById('stop-training').setAttribute('disabled', true);
}

export function enablePredictButton() {
  document.getElementById('predict').removeAttribute('disabled');
}

export function disablePredictButton() {
  document.getElementById('predict').setAttribute('disabled', true);
}
