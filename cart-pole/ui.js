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

import {CartPole} from './cart_pole.js';
import {SaveablePolicyNetwork} from './index.js';
import {mean, sum} from './utils.js';

const appStatus = document.getElementById('app-status');
const storedModelStatusInput = document.getElementById('stored-model-status');
const hiddenLayerSizesInput = document.getElementById('hidden-layer-sizes');
const createModelButton = document.getElementById('create-model');
const deleteStoredModelButton = document.getElementById('delete-stored-model');
const cartPoleCanvas = document.getElementById('cart-pole-canvas');

const numIterationsInput = document.getElementById('num-iterations');
const gamesPerIterationInput = document.getElementById('games-per-iteration');
const discountRateInput = document.getElementById('discount-rate');
const maxStepsPerGameInput = document.getElementById('max-steps-per-game');
const learningRateInput = document.getElementById('learning-rate');
const renderDuringTrainingCheckbox =
    document.getElementById('render-during-training');

const trainButton = document.getElementById('train');
const testButton = document.getElementById('test');
const iterationStatus = document.getElementById('iteration-status');
const iterationProgress = document.getElementById('iteration-progress');
const trainStatus = document.getElementById('train-status');
const trainSpeed = document.getElementById('train-speed');
const trainProgress = document.getElementById('train-progress');

const stepsContainer = document.getElementById('steps-container');

// 정책 네트워크 객체
let policyNet;
let stopRequested = false;

/**
 * 정보 div에 메시지를 출력합니다.
 *
 * @param {string} message 출력할 메시지
 */
function logStatus(message) {
  appStatus.textContent = message;
}

// 훈련하는 동안 카트-막대 상태 출력을 위한 객체와 함수
let renderDuringTraining = true;
export async function maybeRenderDuringTraining(cartPole) {
  if (renderDuringTraining) {
    renderCartPole(cartPole, cartPoleCanvas);
    await tf.nextFrame();
  }
}

/**
 * 훈련하는 동안 게임 끝마다 호출될 함수
 *
 * @param {number} gameCount 현재 훈련 반복에서 지금까지 완료된 게임 횟수
 * @param {number} totalGames 현재 훈련 반복에서 완료해야할 전체 게임 횟수
 */
export function onGameEnd(gameCount, totalGames) {
  iterationStatus.textContent = `게임: ${gameCount} / ${totalGames}`;
  iterationProgress.value = gameCount / totalGames * 100;
  if (gameCount === totalGames) {
    iterationStatus.textContent = '가중치 업데이트 중...';
  }
}

/**
 * 훈련 반복 끝에 호출될 함수
 *
 * @param {number} iterationCount 지금까지 완료된 반복 횟수
 * @param {*} totalIterations 완료해야 할 전체 반복 횟수
 */
function onIterationEnd(iterationCount, totalIterations) {
  trainStatus.textContent = `반복: ${iterationCount} / ${totalIterations}`;
  trainProgress.value = iterationCount / totalIterations * 100;
}

// 훈련 도중 게임 스텝을 출력하기 위한 객체와 함수
let meanStepValues = [];
function plotSteps() {
  tfvis.render.linechart(stepsContainer, {values: meanStepValues}, {
    xLabel: '훈련 반복',
    yLabel: '게임 당 평균 스텝수',
    width: 400,
    height: 300,
  });
}

function disableModelControls() {
  trainButton.textContent = '중지';
  testButton.disabled = true;
  deleteStoredModelButton.disabled = true;
}

function enableModelControls() {
  trainButton.textContent = '훈련';
  testButton.disabled = false;
  deleteStoredModelButton.disabled = false;
}

/**
 * HTML 캔바스에 시스템의 현재 상태를 렌더링합니다.
 *
 * @param {CartPole} cartPole 렌더링할 cart-pole 시스템 객체
 * @param {HTMLCanvasElement} canvas 렌더링이 일어날 HTMLCanvasElement 객체
 */
function renderCartPole(cartPole, canvas) {
  if (!canvas.style.display) {
    canvas.style.display = 'block';
  }
  const X_MIN = -cartPole.xThreshold;
  const X_MAX = cartPole.xThreshold;
  const xRange = X_MAX - X_MIN;
  const scale = canvas.width / xRange;

  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  const halfW = canvas.width / 2;

  // 카트 그리기
  const railY = canvas.height * 0.8;
  const cartW = cartPole.cartWidth * scale;
  const cartH = cartPole.cartHeight * scale;

  const cartX = cartPole.x * scale + halfW;

  context.beginPath();
  context.strokeStyle = '#000000';
  context.lineWidth = 2;
  context.rect(cartX - cartW / 2, railY - cartH / 2, cartW, cartH);
  context.stroke();

  // 카트 아래 바퀴 그리기
  const wheelRadius = cartH / 4;
  for (const offsetX of [-1, 1]) {
    context.beginPath();
    context.lineWidth = 2;
    context.arc(
        cartX - cartW / 4 * offsetX, railY + cartH / 2 + wheelRadius,
        wheelRadius, 0, 2 * Math.PI);
    context.stroke();
  }

  // 막대 그리기
  const angle = cartPole.theta + Math.PI / 2;
  const poleTopX =
      halfW + scale * (cartPole.x + Math.cos(angle) * cartPole.length);
  const poleTopY = railY -
      scale * (cartPole.cartHeight / 2 + Math.sin(angle) * cartPole.length);
  context.beginPath();
  context.strokeStyle = '#ffa500';
  context.lineWidth = 6;
  context.moveTo(cartX, railY - cartH / 2);
  context.lineTo(poleTopX, poleTopY);
  context.stroke();

  // 트랙 그리기
  const groundY = railY + cartH / 2 + wheelRadius * 2;
  context.beginPath();
  context.strokeStyle = '#000000';
  context.lineWidth = 1;
  context.moveTo(0, groundY);
  context.lineTo(canvas.width, groundY);
  context.stroke();

  const nDivisions = 40;
  for (let i = 0; i < nDivisions; ++i) {
    const x0 = canvas.width / nDivisions * i;
    const x1 = x0 + canvas.width / nDivisions / 2;
    const y0 = groundY + canvas.width / nDivisions / 2;
    const y1 = groundY;
    context.beginPath();
    context.moveTo(x0, y0);
    context.lineTo(x1, y1);
    context.stroke();
  }

  // 왼쪽과 오른쪽 경계 그리기
  const limitTopY = groundY - canvas.height / 2;
  context.beginPath();
  context.strokeStyle = '#ff0000';
  context.lineWidth = 2;
  context.moveTo(1, groundY);
  context.lineTo(1, limitTopY);
  context.stroke();
  context.beginPath();
  context.moveTo(canvas.width - 1, groundY);
  context.lineTo(canvas.width - 1, limitTopY);
  context.stroke();
}

async function updateUIControlState() {
  const modelInfo = await SaveablePolicyNetwork.checkStoredModelStatus();
  if (modelInfo == null) {
    storedModelStatusInput.value = '저장된 모델이 없습니다.';
    deleteStoredModelButton.disabled = true;

  } else {
    storedModelStatusInput.value = `저장@${modelInfo.dateSaved.toISOString()}`;
    deleteStoredModelButton.disabled = false;
    createModelButton.disabled = true;
  }
  createModelButton.disabled = policyNet != null;
  hiddenLayerSizesInput.disabled = policyNet != null;
  trainButton.disabled = policyNet == null;
  testButton.disabled = policyNet == null;
  renderDuringTrainingCheckbox.checked = renderDuringTraining;
}

export async function setUpUI() {
  const cartPole = new CartPole(true);

  if (await SaveablePolicyNetwork.checkStoredModelStatus() != null) {
    policyNet = await SaveablePolicyNetwork.loadModel();
    logStatus('Loaded policy network from IndexedDB.');
    hiddenLayerSizesInput.value = policyNet.hiddenLayerSizes();
  }
  await updateUIControlState();

  renderDuringTrainingCheckbox.addEventListener('change', () => {
    renderDuringTraining = renderDuringTrainingCheckbox.checked;
  });

  createModelButton.addEventListener('click', async () => {
    try {
      const hiddenLayerSizes =
          hiddenLayerSizesInput.value.trim().split(',').map(v => {
            const num = Number.parseInt(v.trim());
            if (!(num > 0)) {
              throw new Error(
                  `잘못된 은닉층 크기: ` +
                  `${hiddenLayerSizesInput.value}`);
            }
            return num;
          });
      policyNet = new SaveablePolicyNetwork(hiddenLayerSizes);
      console.log('SaveablePolicyNetwork의 객체 생성 완료');
      await updateUIControlState();
    } catch (err) {
      logStatus(`에러: ${err.message}`);
    }
  });

  deleteStoredModelButton.addEventListener('click', async () => {
    if (confirm(`정말로 로컬에 저장된 모델을 삭제하시겠습니까?`)) {
      await policyNet.removeModel();
      policyNet = null;
      await updateUIControlState();
    }
  });

  trainButton.addEventListener('click', async () => {
    if (trainButton.textContent === '중지') {
      stopRequested = true;
    } else {
      disableModelControls();

      try {
        const trainIterations = Number.parseInt(numIterationsInput.value);
        if (!(trainIterations > 0)) {
          throw new Error(`잘못된 반복 횟수: ${trainIterations}`);
        }
        const gamesPerIteration = Number.parseInt(gamesPerIterationInput.value);
        if (!(gamesPerIteration > 0)) {
          throw new Error(
              `잘못된 반복당 게임수: ${gamesPerIteration}`);
        }
        const maxStepsPerGame = Number.parseInt(maxStepsPerGameInput.value);
        if (!(maxStepsPerGame > 1)) {
          throw new Error(`잘못된 게임 당 최대 스텝수: ${maxStepsPerGame}`);
        }
        const discountRate = Number.parseFloat(discountRateInput.value);
        if (!(discountRate > 0 && discountRate < 1)) {
          throw new Error(`잘못된 할인 계수: ${discountRate}`);
        }
        const learningRate = Number.parseFloat(learningRateInput.value);

        logStatus(
            '정책 네트워크 훈련 중... 잠시 기다려 주세요. ' +
            '마지막 반복이 종료되면 네트워크를 IndexedDB에 저장합니다.');
        const optimizer = tf.train.adam(learningRate);

        meanStepValues = [];
        onIterationEnd(0, trainIterations);
        let t0 = new Date().getTime();
        stopRequested = false;
        for (let i = 0; i < trainIterations; ++i) {
          const gameSteps = await policyNet.train(
              cartPole, optimizer, discountRate, gamesPerIteration,
              maxStepsPerGame);
          const t1 = new Date().getTime();
          const stepsPerSecond = sum(gameSteps) / ((t1 - t0) / 1e3);
          t0 = t1;
          trainSpeed.textContent = `${stepsPerSecond.toFixed(1)} 스텝/초`
          meanStepValues.push({x: i + 1, y: mean(gameSteps)});
          console.log(`텐서 개수: ${tf.memory().numTensors}`);
          plotSteps();
          onIterationEnd(i + 1, trainIterations);
          await tf.nextFrame();
          await policyNet.saveModel();
          await updateUIControlState();

          if (stopRequested) {
            logStatus('훈련을 중지합니다.');
            break;
          }
        }
        if (!stopRequested) {
          logStatus('훈련이 완료되었습니다.');
        }
      } catch (err) {
        logStatus(`에러: ${err.message}`);
      }
      enableModelControls();
    }
  });

  testButton.addEventListener('click', async () => {
    disableModelControls();
    let isDone = false;
    const cartPole = new CartPole(true);
    cartPole.setRandomState();
    let steps = 0;
    stopRequested = false;
    while (!isDone) {
      steps++;
      tf.tidy(() => {
        const action = policyNet.getActions(cartPole.getStateTensor())[0];
        logStatus(
            `테스트 진행 중. ` +
            `행동: ${action === 1 ? '<--' : ' -->'} (Step ${steps})`);
        isDone = cartPole.update(action);
        renderCartPole(cartPole, cartPoleCanvas);
      });
      await tf.nextFrame();
      if (stopRequested) {
        break;
      }
    }
    if (stopRequested) {
      logStatus(`${steps} 스텝 후에 테스트가 중지되었습니다.`);
    } else {
      logStatus(`테스트가 완료되었습니다. ${steps} 스텝을 진행했습니다.`);
    }
    console.log(`텐서 개수: ${tf.memory().numTensors}`);
    enableModelControls();
  });
}
