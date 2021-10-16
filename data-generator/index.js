
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
import * as ui from './ui.js';

/**
 * 무제한으로 게임 플레이를 만드는 데이터셋을 반환합니다.
 */
export const GAME_GENERATOR_DATASET = tf.data.generator(function* gen() {
  while (true) {
    yield game.generateOnePlay();
  }
});

/**
 * 모델이 현재 에포크를 마치고 훈련을 중지해야 하는지 나타내는 전역 불리언 변수.
 */
let STOP_REQUESTED = false;

/**
 * 특성 표현을 다시 계산하기 위해 가장 최근의 시뮬레이션 게임 상태를 가지고 있습니다.
 */
let SAMPLE_GAME_STATE;

/**
 * 훈련, 평가할 모델
 */
let GLOBAL_MODEL;

/**
 * 완료된 게임 상태를 받아 훈련에 적절한 특성을 반환합니다.
 * 반환된 객체가 담고 있는 특성은 원-핫 인코딩을 사용한 플레이어 1의 카드이고,
 * 레이블은 플레이어 1이 이길지 여부입니다.
 * @param {*} gameState
 */
function gameToFeaturesAndLabel(gameState) {
  return tf.tidy(() => {
    const player1Hand = tf.tensor1d(gameState.player1Hand, 'int32');
    const handOneHot = tf.oneHot(
        tf.sub(player1Hand, tf.scalar(1, 'int32')),
        game.GAME_STATE.max_card_value);
    const features = tf.sum(handOneHot, 0);
    const label = tf.tensor1d([gameState.player1Win]);
    return {xs: features, ys: label};
  });
}

/**
 * 랜덤한 한 번의 게임 플레이를 수행합니다.
 * 샘플에서 게임 플레이를 표현하는 특성과 레이블을 만듭니다.
 * UI 메서드를 호출하여 샘플과 처리된 샘플을 나타냅니다.
 * @param {bool} wantNewGame : true이면 새로운 게임이 생성됩니다.
 */
async function simulateGameHandler(wantNewGame) {
  if (wantNewGame) {
    SAMPLE_GAME_STATE = game.generateOnePlay();
  }
  const featuresAndLabel = gameToFeaturesAndLabel(SAMPLE_GAME_STATE);
  ui.displaySimulation(SAMPLE_GAME_STATE, featuresAndLabel);
  ui.displayNumSimulationsSoFar();
}

/**
 * 비동기 코드를 격리하기 위해 별도의 함수로 만듭니다.
 *  @see datasetToArrayHandler
 */
async function datasetToArray() {
  return GAME_GENERATOR_DATASET.map(gameToFeaturesAndLabel)
      .batch(ui.getBatchSize())
      .take(ui.getTake())
      .toArray();
}

/**
 * GAME_GENERATOR_DATASET에서 데이터셋 파이프라인을 만듭니다.
 * 1) gameToFeaturesAndlabel 함수를 적용합니다.
 * 2) 배치 크기 B의 배치를 만듭니다.
 * 3) 데이터셋에서 처음 N 개의 샘플을 가져옵니다.
 *
 * 그다음 데이터셋을 실해하여 배열을 채웁니다.
 * 마지막으로 이 배열을 UI에 전달하여 테이블을 그립니다.
 */
async function datasetToArrayHandler() {
  const arr = await datasetToArray();
  ui.displayBatches(arr);
  ui.displayNumSimulationsSoFar();
}

/**
 * 특성 표현에서 승리 여부를 예측하기에 적절한 세 개의 층을 가진 시퀀셜 모델을 반환합니다.
 * 입력 크기는 원-핫 표현을 사용하는지에 따라 달라집니다.
 */
function createDNNModel() {
  GLOBAL_MODEL = tf.sequential();
  GLOBAL_MODEL.add(tf.layers.dense({
    inputShape: [game.GAME_STATE.max_card_value],
    units: 20,
    activation: 'relu'
  }));
  GLOBAL_MODEL.add(tf.layers.dense({units: 20, activation: 'relu'}));
  GLOBAL_MODEL.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  return GLOBAL_MODEL;
}


/**
 * model.fitDataset을 사용해 주어진 데이터셋에서 모델을 훈련합니다.
 * 에포크가 끝날 때마다 콜백을 호출하여 손실과 정확도를 UI에 그립니다.
 * 또한 훈련 속도와 수동으로 입력한 카드에 대한 현재 예측을 출력합니다.
 * @param {tf.Model} model
 * @param {tf.data.Dataset} dataset
 */
async function trainModelUsingFitDataset(model, dataset) {
  const trainLogs = [];
  const beginMs = performance.now();
  const fitDatasetArgs = {
    batchesPerEpoch: ui.getBatchesPerEpoch(),
    epochs: ui.getEpochsToTrain(),
    validationData: dataset,
    validationBatches: 10,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // 훈련 에포크가 끝날 때마다 손실과 정확도 값을 그래프로 그립니다.
        const secPerEpoch =
            (performance.now() - beginMs) / (1000 * (epoch + 1));
        ui.displayTrainLogMessage(
            `모델 훈련 중... 약 ` +
            `${secPerEpoch.toFixed(4)} 초/에포크`);
        trainLogs.push(logs);
        tfvis.show.history(
            ui.lossContainerElement, trainLogs, ['loss', 'val_loss'])
        tfvis.show.history(
            ui.accuracyContainerElement, trainLogs, ['acc', 'val_acc'],
            {zoomToFitAccuracy: true})
        ui.displayNumSimulationsSoFar();
        // 예측을 업데이트합니다.
        predictHandler();
        // 종료 버튼을 눌렀다면 훈련을 멈춥니다.
        if (STOP_REQUESTED) {
          model.stopTraining = true;
        }
      },
    }
  };
  ui.disableTrainButton();
  ui.enableStopButton();
  ui.enablePredictButton();
  await model.fitDataset(dataset, fitDatasetArgs);
  ui.enableTrainButton();
  ui.disableStopButton();
}

/**
 * 새로운 모델을 만들고 GAME_GENERATOR_DATASET으로 구축한 데이터셋 파이프라인에서 훈련합니다.
 * 이 데이터셋 파이프라인은 특성을 계산하고 배치를 생성합니다.
 * 자세한 훈련 내용은 trainModelUsingFitDataset를 참고하세요.
 */
async function trainModelUsingFitDatasetHandler() {
  STOP_REQUESTED = false;
  const model = createDNNModel();
  model.compile({
    optimizer: 'rmsprop',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  const dataset = GAME_GENERATOR_DATASET.map(gameToFeaturesAndLabel)
                      .batch(ui.getBatchSize());
  trainModelUsingFitDataset(model, dataset);
}

/**
 * 수동으로 입력한 카드에 모델을 적용하고 예측을 UI에 업데이트합니다.
 */
function predictHandler() {
  const cards = ui.getInputCards();
  const features =
      gameToFeaturesAndLabel({player1Hand: cards, player1Win: 1}).xs;
  const output = GLOBAL_MODEL.predict(features.expandDims(0));
  ui.displayPrediction(`${output.dataSync()[0].toFixed(3)}`);
}

/**
 * 카드 개수를 업데이트하고 UI를 초기화합니다.
 */
function selectCardsPerHandHandler() {
  game.GAME_STATE.num_cards_per_hand =
      Number.parseInt(document.getElementById('select-cards-per-hand').value);
  simulateGameHandler(true);
  ui.updatePredictionInputs();
  ui.displayBatches([]);
  ui.disablePredictButton();
  ui.displayPrediction('새로운 모델을 훈련해야 합니다');
}

/** 사용자의 클릭을 휘한 핸들러를 등록합니다. */
document.addEventListener('DOMContentLoaded', async () => {
  console.log('콘텐츠가 로딩되었습니다... 버튼을 연결 중입니다.');
  document.getElementById('select-cards-per-hand')
      .addEventListener('change', selectCardsPerHandHandler, false);
  document.getElementById('simulate-game')
      .addEventListener('click', () => simulateGameHandler(true), false);
  document.getElementById('dataset-to-array')
      .addEventListener('click', datasetToArrayHandler, false);
  document.getElementById('dataset-to-array')
      .addEventListener('click', datasetToArrayHandler, false);
  document.getElementById('train-model-using-fit-dataset')
      .addEventListener('click', trainModelUsingFitDatasetHandler, false);
  document.getElementById('stop-training')
      .addEventListener('click', () => STOP_REQUESTED = true);
  document.getElementById('generator-batch').addEventListener('change', () => {
    ui.displayExpectedSimulations();
    ui.displayBatches([]);
  }, false);
  document.getElementById('generator-take').addEventListener('change', () => {
    ui.displayBatches([]);
  }, false);
  document.getElementById('batches-per-epoch')
      .addEventListener('change', ui.displayExpectedSimulations, false);
  document.getElementById('epochs-to-train')
      .addEventListener('change', ui.displayExpectedSimulations, false);
  document.getElementById('predict').addEventListener(
      'click', predictHandler, false);
  ui.displayNumSimulationsSoFar();
  ui.displayExpectedSimulations();
  ui.updatePredictionInputs();
});
