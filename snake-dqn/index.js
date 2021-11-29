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

import {ALL_ACTIONS, getStateTensor, SnakeGame} from './snake_game.js';
import {renderSnakeGame} from './snake_graphics.js';

const gameCanvas = document.getElementById('game-canvas');

const loadHostedModelButton = document.getElementById('load-hosted-model');

const stepButton = document.getElementById('step');
const resetButton = document.getElementById('reset');
const autoPlayStopButton = document.getElementById('auto-play-stop');
const gameStatusSpan = document.getElementById('game-status');
const showQValuesCheckbox = document.getElementById('show-q-values');

let game;
let qNet;

let cumulativeReward = 0;
let cumulativeFruits = 0;
let autoPlaying = false;
let autoPlayIntervalJob;

/** 게임 상태를 리셋합니다. */
async function reset() {
  if (game == null) {
    return;
  }
  game.reset();
  await calcQValuesAndBestAction();
  renderSnakeGame(gameCanvas, game,
      showQValuesCheckbox.checked ? currentQValues : null);
  gameStatusSpan.textContent = '게임 시작';
  stepButton.disabled = false;
  autoPlayStopButton.disabled = false;
}

/**
 * 게임을 한 스텝 플레이합니다.
 *
 * - 현재 최선의 행동을 사용하여 게임에서 한 스텝을 진행합니다.
 * - 보상을 누적합니다.
 * - 게임이 종료되었는지 판단하고 이에 따라 화면을 업데이트합니다.
 * - 게임이 종료되지 않았다면 현재 Q-가치와 최선의 행동을 계산합니다.
 * - 게임을 캔바스에 렌더링합니다.
 */
async function step() {
  const {reward, done, fruitEaten} = game.step(bestAction);
  invalidateQValuesAndBestAction();
  cumulativeReward += reward;
  if (fruitEaten) {
    cumulativeFruits++;
  }
  gameStatusSpan.textContent =
      `보상=${cumulativeReward.toFixed(1)}; 과일=${cumulativeFruits}`;
  if (done) {
    gameStatusSpan.textContent += '. 게임 종료!';
    cumulativeReward = 0;
    cumulativeFruits = 0;
    if (autoPlayIntervalJob) {
      clearInterval(autoPlayIntervalJob);
      autoPlayStopButton.click();
    }
    autoPlayStopButton.disabled = true;
    stepButton.disabled = true;
  }
  await calcQValuesAndBestAction();
  renderSnakeGame(gameCanvas, game,
      showQValuesCheckbox.checked ? currentQValues : null);
}

let currentQValues;
let bestAction;

/** 현재 Q-가치와 최선의 행동을 계산합니다. */
async function calcQValuesAndBestAction() {
  if (currentQValues != null) {
    return;
  }
  tf.tidy(() => {
    const stateTensor = getStateTensor(game.getState(), game.height, game.width);
    const predictOut = qNet.predict(stateTensor);
    currentQValues = predictOut.dataSync();
    bestAction = ALL_ACTIONS[predictOut.argMax(-1).dataSync()[0]];
  });
}

function invalidateQValuesAndBestAction() {
  currentQValues = null;
  bestAction = null;
}

const LOCAL_MODEL_URL = './dqn/model.json';
const REMOTE_MODEL_URL = 'https://storage.googleapis.com/tfjs-examples/snake-dqn/dqn/model.json';

function enableGameButtons() {
  autoPlayStopButton.disabled = false;
  stepButton.disabled = false;
  resetButton.disabled = false;
}

async function initGame() {
  game = new SnakeGame({
    height: 9,
    width: 9,
    numFruits: 1,
    initLen: 2
  });

  // qNet 워밍업
  for (let i = 0; i < 3; ++i) {
    qNet.predict(getStateTensor(game.getState(), game.height, game.width));
  }

  await reset();

  stepButton.addEventListener('click', step);

  autoPlayStopButton.addEventListener('click', () => {
    if (autoPlaying) {
      autoPlayStopButton.textContent = '자동 플레이';
      if (autoPlayIntervalJob) {
        clearInterval(autoPlayIntervalJob);
      }
    } else {
      autoPlayIntervalJob = setInterval(() => {
        step(game, qNet);
      }, 100);
      autoPlayStopButton.textContent = '중지';
    }
    autoPlaying = !autoPlaying;
    stepButton.disabled = autoPlaying;
  });

  resetButton.addEventListener('click',  () => reset(game));
}

(async function() {
  try {
    qNet = await tf.loadLayersModel(LOCAL_MODEL_URL);
    loadHostedModelButton.textContent = `${LOCAL_MODEL_URL}에서 모델을 로드했습니다.`;
    initGame();
    enableGameButtons();
  } catch (err) {
    console.log('로컬 모델 로딩 실패');
    loadHostedModelButton.disabled = false;
  }

  loadHostedModelButton.addEventListener('click', async () => {
    try {
      qNet = await tf.loadLayersModel(REMOTE_MODEL_URL);
      loadHostedModelButton.textContent = `원격 모델을 로드했습니다.`;
      loadHostedModelButton.disabled = true;
      initGame();
      enableGameButtons();
    } catch (err) {
      loadHostedModelButton.textContent = '모델을 로드하지 못했습니다.'
      loadHostedModelButton.disabled = true;
    }
  });
})();
