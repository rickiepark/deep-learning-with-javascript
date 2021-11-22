
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

import {TEXT_DATA_URLS, TextData} from './data.js';
import {SaveableLSTMTextGenerator} from './index.js';

// UI 콘트롤
const testText = document.getElementById('test-text');
const createOrLoadModelButton = document.getElementById('create-or-load-model');
const deleteModelButton = document.getElementById('delete-model');
const trainModelButton = document.getElementById('train-model');
const generateTextButton = document.getElementById('generate-text');

const appStatus = document.getElementById('app-status');
const textGenerationStatus = document.getElementById('text-generation-status');
const loadTextDataButton = document.getElementById('load-text-data');
const textDataSelect = document.getElementById('text-data-select');

const lstmLayersSizesInput = document.getElementById('lstm-layer-sizes');

const examplesPerEpochInput = document.getElementById('examples-per-epoch');
const batchSizeInput = document.getElementById('batch-size');
const epochsInput = document.getElementById('epochs');
const validationSplitInput = document.getElementById('validation-split');
const learningRateInput = document.getElementById('learning-rate');

const generateLengthInput = document.getElementById('generate-length');
const temperatureInput = document.getElementById('temperature');
const seedTextInput = document.getElementById('seed-text');
const generatedTextInput = document.getElementById('generated-text');

const modelAvailableInfo = document.getElementById('model-available');

const sampleLen = 40;
const sampleStep = 3;

// TextData 객체
let textData;

// SaveableLSTMTextGenerator 객체
let textGenerator;

function logStatus(message) {
  appStatus.textContent = message;
}

let batchLossValues;
let epochLossValues;

/**
 * 훈련이 시작될 때 호출되는 함수
 */
export function onTrainBegin() {
  batchLossValues = [];
  epochLossValues = [];
  logStatus('모델 훈련을 시작합니다...');
}

/**
 * 훈련에서 배치가 끝날 때 호출되는 함수
 *
 * @param {number} loss 현재 배치의 손실 값
 * @param {number} progress 0~1 사이의 숫자로 나타난 전체 훈련 진행 상황
 * @param {number} examplesPerSec 초당 샘플 수로 나타난 배치에서 훈련 속도
 */
export function onTrainBatchEnd(logs, progress, examplesPerSec) {
  logStatus(
      `모델 훈련: ${(progress * 1e2).toFixed(1)}% 완료... ` +
      `(${examplesPerSec.toFixed(0)} 샘플/초)`);
  batchLossValues.push(logs);
  const container = document.getElementById('batch-loss-canvas');
  tfvis.show.history(container, batchLossValues, ['loss'], {
    height: 300,
    zoomToFit: true,
    xLabel: 'Batch',
  });
}

export function onTrainEpochEnd(logs) {
  epochLossValues.push(logs);
  const container = document.getElementById('epoch-loss-canvas');
  tfvis.show.history(container, epochLossValues, ['loss', 'val_loss'], {
    height: 300,
    zoomToFit: true,
    xLabel: 'Epoch',
  });
}

/**
 * 텍스트 생성이 시작될 때 호출되는 함수
 *
 * @param {string} seedSentence: 텍스트 생성을 위해 사용할 시드 문장
 */
export function onTextGenerationBegin() {
  generatedTextInput.value = '';
  logStatus('텍스트를 생성합니다...');
}

/**
 * 텍스트 생성에서 문자가 결정될 때마다 호출되는 함수
 *
 * @param {string} char 방금 생성된 문자
 */
export async function onTextGenerationChar(char) {
  generatedTextInput.value += char;
  generatedTextInput.scrollTop = generatedTextInput.scrollHeight;
  const charCount = generatedTextInput.value.length;
  const generateLength = parseInt(generateLengthInput.value);
  const status = `텍스트 생성: ${charCount}/${generateLength} 완료...`;
  logStatus(status);
  textGenerationStatus.textContent = status;
  await tf.nextFrame();
}

export function setUpUI() {
  /**
   * 로컬(IndexedDB)에 저장된 모델의 상태 갱신
   */
  async function refreshLocalModelStatus() {
    const modelInfo = await textGenerator.checkStoredModelStatus();
    if (modelInfo == null) {
      modelAvailableInfo.innerText =
          `로컬에 저장된 모델(${textGenerator.modelIdentifier()})이 없습니다.`;
      createOrLoadModelButton.textContent = '모델 생성';
      deleteModelButton.disabled = true;
      enableModelParameterControls();
    } else {
      modelAvailableInfo.innerText =
          `@ ${modelInfo.dateSaved.toISOString()}에 저장한 모델이 있습니다.`;
      createOrLoadModelButton.textContent = '모델 로드';
      deleteModelButton.disabled = false;
      disableModelParameterControls();
    }
    createOrLoadModelButton.disabled = false;
  }

  function disableModelButtons() {
    createOrLoadModelButton.disabled = true;
    deleteModelButton.disabled = true;
    trainModelButton.disabled = true;
    generateTextButton.disabled = true;
  }

  function enableModelButtons() {
    createOrLoadModelButton.disabled = false;
    deleteModelButton.disabled = false;
    trainModelButton.disabled = false;
    generateTextButton.disabled = false;
  }

  /**
   * `textGenerator`를 사용해 랜덤한 텍스트를 생성하고 문자가 하나씩 생성될 때마다 화면에 출력합니다.
   */
  async function generateText() {
    try {
      disableModelButtons();

      if (textGenerator == null) {
        logStatus('에러: 먼저 텍스트 데이터셋을 로드하세요.');
        return;
      }
      const generateLength = parseInt(generateLengthInput.value);
      const temperature = parseFloat(temperatureInput.value);
      if (!(generateLength > 0)) {
        logStatus(
            `에러: 잘못된 생성 길이: ${generateLength}. ` +
            `생성 길이는 양수여야 합니다.`);
        enableModelButtons();
        return;
      }
      if (!(temperature > 0 && temperature <= 1)) {
        logStatus(
            `에러: 잘못된 온도: ${temperature}. ` +
            `온도는 양수여야 합니다.`);
        enableModelButtons();
        return;
      }

      let seedSentence;
      let seedSentenceIndices;
      if (seedTextInput.value.length === 0) {
        // 시드 문장이 지정되지 않아서 데이터에서 만듭니다.
        [seedSentence, seedSentenceIndices] = textData.getRandomSlice();
        seedTextInput.value = seedSentence;
      } else {
        seedSentence = seedTextInput.value;
        if (seedSentence.length < textData.sampleLen()) {
          logStatus(
              `에러: 시드 텍스트의 길이는 최소한 ` +
              `${textData.sampleLen()}이 되어야 합니다. 하지만 ` +
              `${seedSentence.length}가 입력되었습니다.`);
          enableModelButtons();
          return;
        }
        seedSentence = seedSentence.slice(
            seedSentence.length - textData.sampleLen(), seedSentence.length);
        seedSentenceIndices = textData.textToIndices(seedSentence);
      }

      const sentence = await textGenerator.generateText(
          seedSentenceIndices, generateLength, temperature);
      generatedTextInput.value = sentence;
      const status = '텍스트 생성 완료.';
      logStatus(status);
      textGenerationStatus.value = status;

      enableModelButtons();

      return sentence;
    } catch (err) {
      logStatus(`에러: 텍스트 생성 실패: ${err.message}, ${err.stack}`);
    }
  }

  function disableModelParameterControls() {
    lstmLayersSizesInput.disabled = true;
  }

  function enableModelParameterControls() {
    lstmLayersSizesInput.disabled = false;
  }

  function updateModelParameterControls(lstmLayerSizes) {
    lstmLayersSizesInput.value = lstmLayerSizes;
  }

  function updateTextInputParameters() {
    Object.keys(TEXT_DATA_URLS).forEach(key => {
      var opt = document.createElement('option');
      opt.value = key;
      opt.innerHTML = TEXT_DATA_URLS[key].needle;
      textDataSelect.appendChild(opt);
    });
  }

  function hashCode(str) {
    let hash = 5381, i = str.length;
    while (i) {
      hash = (hash * 33) ^ str.charCodeAt(--i);
    }
    return hash >>> 0;
  }

  /**
   * UI 상태 초기화
   */
  disableModelParameterControls();

  /**
   * 텍스트 입력창 업데이트
   */
  updateTextInputParameters();

  /**
   * UI 콜백 연결
   */
  loadTextDataButton.addEventListener('click', async () => {
    textDataSelect.disabled = true;
    loadTextDataButton.disabled = true;
    let dataIdentifier = textDataSelect.value;
    const url = TEXT_DATA_URLS[dataIdentifier].url;
    if (testText.value.length === 0) {
      try {
        logStatus(`다음 URL에서 텍스트 데이터를 로딩합니다: ${url} ...`);
        const response = await fetch(url);
        const textString = await response.text();
        testText.value = textString;
        logStatus(
            `텍스트 데이터 로드 완료 ` +
            `(길이=${(textString.length / 1024).toFixed(1)}k). ` +
            `이제, 모델을 로드하거나 만드세요.`);
      } catch (err) {
        logStatus('텍스트 데이터 로드 실패: ' + err.message);
      }
      if (testText.value.length === 0) {
        logStatus('에러: 빈 텍스트 데이터');
        return;
      }
    } else {
      dataIdentifier = hashCode(testText.value);
    }
    textData =
        new TextData(dataIdentifier, testText.value, sampleLen, sampleStep);
    textGenerator = new SaveableLSTMTextGenerator(textData);
    await refreshLocalModelStatus();
  });

  createOrLoadModelButton.addEventListener('click', async () => {
    createOrLoadModelButton.disabled = true;
    if (textGenerator == null) {
      createOrLoadModelButton.disabled = false;
      logStatus('에러: 먼저 텍스트 데이터셋을 로드하세요.');
      return;
    }

    if (await textGenerator.checkStoredModelStatus()) {
      // 로컬에 저장된 모델 로드
      logStatus('IndexedDB에서 모델을 로드합니다... 잠시 기다려 주세요.');
      await textGenerator.loadModel();
      updateModelParameterControls(textGenerator.lstmLayerSizes());
      logStatus(
          'IndexedDB 모델을 로드하였습니다. ' +
          '이제 모델을 더 훈련하거나 텍스트를 생성할 수 있습니다.');
    } else {
      // 처음부터 모델을 만듭니다.
      logStatus('모델 생성 중... 잠시 기다려 주세요.');
      const lstmLayerSizes = lstmLayersSizesInput.value.trim().split(',').map(
          s => parseInt(s));

      // LSTM 층 크기 확인
      if (lstmLayerSizes.length === 0) {
        logStatus('에러: 잘못된 LSTM 층 크기');
        return;
      }
      for (let i = 0; i < lstmLayerSizes.length; ++i) {
        const lstmLayerSize = lstmLayerSizes[i];
        if (!(lstmLayerSize > 0)) {
          logStatus(
              `에러: lstmLayerSizes는 양의 정수여야 합니다. ` +
              `하지만  ${i + 1}번째 층의 크기가 ${lstmLayerSize}입니다.`);
          return;
        }
      }

      await textGenerator.createModel(lstmLayerSizes);
      logStatus(
          '모델 생성 완료. ' +
          '이제 모델을 훈련하거나 텍스트를 생성할 수 있습니다.');
    }

    trainModelButton.disabled = false;
    generateTextButton.disabled = false;
  });

  deleteModelButton.addEventListener('click', async () => {
    if (textGenerator == null) {
      logStatus('에러: 먼저 텍스트 데이터셋을 로드하세요.');
      return;
    }
    if (confirm(
            `정말 모델 ` +
            `'${textGenerator.modelIdentifier()}'을 삭제하시겠어요?`)) {
      console.log(await textGenerator.removeModel());
      await refreshLocalModelStatus();
    }
  });

  trainModelButton.addEventListener('click', async () => {
    if (textGenerator == null) {
      logStatus('에러: 먼저 텍스트 데이터셋을 로드하세요.');
      return;
    }

    const numEpochs = parseInt(epochsInput.value);
    if (!(numEpochs > 0)) {
      logStatus(`에러: 잘못된 에포크 횟수: ${numEpochs}`);
      return;
    }
    const examplesPerEpoch = parseInt(examplesPerEpochInput.value);
    if (!(examplesPerEpoch > 0)) {
      logStatus(`에러: 잘못된 에포크 당 샘플 개수: ${examplesPerEpoch}`);
      return;
    }
    const batchSize = parseInt(batchSizeInput.value);
    if (!(batchSize > 0)) {
      logStatus(`에러: 잘못된 배치 개수: ${batchSize}`);
      return;
    }
    const validationSplit = parseFloat(validationSplitInput.value);
    if (!(validationSplit >= 0 && validationSplit < 1)) {
      logStatus(`에러: 잘못된 검증 세트 비율: ${validationSplit}`);
      return;
    }
    const learningRate = parseFloat(learningRateInput.value);
    if (!(learningRate > 0)) {
      logStatus(`에러: 잘못된 학습률: ${learningRate}`);
      return;
    }

    textGenerator.compileModel(learningRate);
    disableModelButtons();
    await textGenerator.fitModel(
        numEpochs, examplesPerEpoch, batchSize, validationSplit);
    console.log(await textGenerator.saveModel());
    await refreshLocalModelStatus();
    enableModelButtons();

    await generateText();
  });

  generateTextButton.addEventListener('click', async () => {
    if (textGenerator == null) {
      logStatus('에러: 먼저 텍스트 데이터셋을 로드하세요.');
      return;
    }
    await generateText();
  });
}
