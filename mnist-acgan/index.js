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

/**
 * 이 파일은 사전 훈련된 ACGAN의 생성자를 로드하고 생성된 가짜 MNIST 이미지를 보여줍니다.
 *
 * 사전 훈련된 생성자 모델은 두 가지 방법으로 준비합니다:
 *   1. 이 파일과 같은 폴더에 있는 `gan.js` 훈련 스크립트를 실행합니다.
 *   2. HTTPS를 통해 호스팅된 모델을 가져옵니다.
 */

import * as ta from './timeago.js';

// 가짜(생성된 이미지)와 비교하기 위해 데이터셋을 로드합니다.
import {loadMnistData, sampleFromMnistData} from './web-data.js';

const status = document.getElementById('status');
const loadHostedModel = document.getElementById('load-hosted-model');
const testModel = document.getElementById('test');
const zSpaceToggleButton = document.getElementById('toggle-sliders');
const slidersContainer = document.getElementById('sliders-container');
const fakeImagesSpan = document.getElementById('fake-images-span');
const fakeCanvas = document.getElementById('fake-canvas');
const realCanvas = document.getElementById('real-canvas');

/**
 * 잠재 벡터를 생성하고 슬라이더로 보여줍니다.
 *
 * @param {bool} fixedLatent 잠재 벡터에 고정된 값을 사용할지 여부 (모든 차원을 0.5로 설정)
 */
function generateLatentVector(fixedLatent) {
  const latentDims = latentSliders.length;

  // 랜덤한 잠재 벡터(z-공간 벡터)를 생성합니다.
  const latentValues = [];
  for (let i = 0; i < latentDims; ++i) {
    const latentValue = fixedLatent === true ? 0.5 : Math.random();
    latentValues.push(latentValue);
    latentSliders[i].value = latentValue;
  }
}

/**
 * 슬라이더에서 잠재 공간 벡터 값을 읽습니다.
 *
 * @param {number} numRepeats 하나의 잠재 벡터를 반복할 횟수. 가짜 MNIST 이미지 배치를 생성하기 위해 사용합니다.
 * @returns [numRepeats, latentDim] 크기의 잠재 공간 벡터
 */
function getLatentVectors(numRepeats) {
  return tf.tidy(() => {
    const latentDims = latentSliders.length;
    const zs = [];
    for (let i = 0; i < latentDims; ++i) {
      zs.push(+latentSliders[i].value);
    }
    const singleLatentVector = tf.tensor2d(zs, [1, latentDims]);
    return singleLatentVector.tile([numRepeats, 1]);
  });
}

/**
 * ACGAN의 생성자를 사용해 이미지를 생성합니다.
 *
 * @param {tf.Model} generator ACGAN의 생성자
 */
async function generateAndVisualizeImages(generator) {
  tf.util.assert(
      generator.inputs.length === 2,
      `2개의 심볼릭 입력을 가진 모델이어야 합니다. ` +
          `현재 모델의 입력 개수: ${generator.inputs.length}`);

  const combinedFakes = tf.tidy(() => {
    const latentVectors = getLatentVectors(10);

    // 각 숫자마다 가짜 이미지를 생성합니다.
    const sampledLabels = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 1]);
    // 출력 픽셀 값의 범위는 [-1, 1]입니다. 이를 [0, 1] 범위로 정규화합니다.
    const t0 = tf.util.now();
    const generatedImages =
        generator.predict([latentVectors, sampledLabels]).add(1).div(2);
    generatedImages.dataSync();  // 정확한 시간 측정을 위해
    const elapsed = tf.util.now() - t0;
    fakeImagesSpan.textContent =
        `가짜 이미지 (생성에 걸린 시간: ${elapsed.toFixed(2)} ms)`;
    // 이미지를 수평으로 연결하여 하나의 이미지로 만듭니다.
    return tf.concat(tf.unstack(generatedImages), 1);
  });

  await tf.browser.toPixels(combinedFakes, fakeCanvas);
  tf.dispose(combinedFakes);
}

/** 진짜 MNIST 이미지 샘플을 갱신합니다 */
async function drawReals() {
  const combinedReals = sampleFromMnistData(10);
  await tf.browser.toPixels(combinedReals, realCanvas);
  tf.dispose(combinedReals);
}

/** 잠재 벡터 값을 위한 슬라이더를 담은 배열 */
let latentSliders;

/**
 * 잠재 공간을 위한 슬라이더를 만듭니다.
 *
 * @param {tf.Model} generator 훈련된 ACGAN의 생성자
 */
function createSliders(generator) {
  const latentDims = generator.inputs[0].shape[1];
  latentSliders = [];
  for (let i = 0; i < latentDims; ++i) {
    const slider = document.createElement('input');
    slider.setAttribute('type', 'range');
    slider.min = 0;
    slider.max = 1;
    slider.step = 0.01;
    slider.value = 0.5;
    slider.addEventListener('change', () => {
      generateAndVisualizeImages(generator);
    });

    slidersContainer.appendChild(slider);
    latentSliders.push(slider);
  }
  slidersContainer.style.display = 'none';
  zSpaceToggleButton.disabled = false;
  zSpaceToggleButton.textContent = `z-벡터 슬라이더 보기 (${latentDims} 차원)`;
}

async function showGeneratorInitially(generator) {
  generator.summary();

  // z-공간(잠재 공간)을 위한 슬라이더를 만듭니다.
  createSliders(generator);

  generateLatentVector(true);
  await generateAndVisualizeImages(generator);
  await drawReals();

  testModel.disabled = false;
}

async function init() {
  // 웹페이지에 출력하기 위해 MNIST 데이터를 로드합니다.
  status.textContent = 'MNIST 데이터 로딩 중...';
  await loadMnistData();

  const LOCAL_MEATADATA_PATH = 'generator/acgan-metadata.json';
  const LOCAL_MODEL_PATH = 'generator/model.json';

  // 원격에 저장된 사전 훈련된 생성자
  const HOSTED_MODEL_URL =
      'https://storage.googleapis.com/tfjs-examples/mnist-acgan/dist/generator/model.json';

  // 로컬에 저장된 모델을 로드해 봅니다. 실패 하면 "원격 모델 로드하기" 버튼을 활성화합니다.
  let model;
  try {
    status.textContent = '메타데이터 로딩';
    const metadata =
        await (await fetch(LOCAL_MEATADATA_PATH, {cache: 'no-cache'})).json();

    status.textContent = `${LOCAL_MODEL_PATH}에서 모델을 로딩합니다...`;
    model = await tf.loadLayersModel(
        tf.io.browserHTTPRequest(LOCAL_MODEL_PATH, {cache: 'no-cache'}));
    await showGeneratorInitially(model);

    if (metadata.completed) {
      status.textContent =
          `Node.js에서 ACGAN 훈련이 완료되었습니다(${metadata.totalEpochs} 에포크).`;
    } else {
      status.textContent = `Node.js에서 ACGAN 훈련이 진행중입니다(에포크 ` +
          `${metadata.currentEpoch + 1}/${metadata.totalEpochs})... `;
    }
    if (metadata.currentEpoch < 10) {
      status.textContent +=
          '(노트: 처음 몇 에포크 동안은 생성자 결과가 나쁠 수 있습니다. ' +
          '하지만 훈련이 진행됨에 따라 나아집니다.) '
    }
    if (metadata.lastUpdated != null) {
      status.textContent +=
          ` (저장된 모델이 ` +
          `${ta.timeago().ago(new Date(metadata.lastUpdated))}에 마지막으로 업데이트되었습니다). `;
    }
    status.textContent +=
        '로컬에 저장된 모델을 로드했습니다! 이제 "이미지 생성" 버튼을 클릭하거나 ' +
        'z-벡터 슬라이더를 조정할 수 있습니다.';
  } catch (err) {
    console.error(err);
    status.textContent =
        '로컬에 저장된 모델을 로드하는데 실패했습니다. ' +
        '"원격 모델 로드하기" 버튼을 클릭하세요.';
  }

  loadHostedModel.addEventListener('click', async () => {
    try {
      status.textContent = `${HOSTED_MODEL_URL}에서 모델을 로드합니다...`;
      model = await tf.loadLayersModel(HOSTED_MODEL_URL);
      loadHostedModel.disabled = true;

      await showGeneratorInitially(model);
      status.textContent =
          `${HOSTED_MODEL_URL}에서 모델을 로드하는데 성공했습니다. ` +
          `이제 "이미지 생성" 버튼을 클릭하거나 z-벡터 슬라이더를 조정할 수 있습니다.`;
    } catch (err) {
      console.error(err);
      status.textContent =
          `${HOSTED_MODEL_URL}에서 모델을 로드하는데 실패했습니다.`;
    }
  });

  testModel.addEventListener('click', async () => {
    generateLatentVector(false);
    await generateAndVisualizeImages(model);
    drawReals();
  });

  zSpaceToggleButton.addEventListener('click', () => {
    if (slidersContainer.style.display === 'none') {
      slidersContainer.style.display = 'block';
      zSpaceToggleButton.textContent =
          zSpaceToggleButton.textContent.replace('Show ', 'Hide ');
    } else {
      slidersContainer.style.display = 'none';
      zSpaceToggleButton.textContent =
          zSpaceToggleButton.textContent.replace('Hide ', 'Show ');
    }
  });
}

init();
