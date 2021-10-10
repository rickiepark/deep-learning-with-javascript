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
 * 이 파일은 사전 훈련된 simple-object-detection 모델로 추론을 실행합니다.
 *
 * 모델은 `train.js`에 정의되어 있고 훈련되었습니다.
 * 모델 훈련과 추론에 사용한 데이터는 프로그래밍적으로 합성한 것입니다.
 * 자세한 내용은 `synthetic_images.js` 파일을 참고하세요.
 */

import {ObjectDetectionImageSynthesizer} from './synthetic_images.js';

const canvas = document.getElementById('data-canvas');
const status = document.getElementById('status');
const testModel = document.getElementById('test');
const loadHostedModel = document.getElementById('load-hosted-model');
const inferenceTimeMs = document.getElementById('inference-time-ms');
const trueObjectClass = document.getElementById('true-object-class');
const predictedObjectClass = document.getElementById('predicted-object-class');

const TRUE_BOUNDING_BOX_LINE_WIDTH = 2;
const TRUE_BOUNDING_BOX_STYLE = 'rgb(255,0,0)';
const PREDICT_BOUNDING_BOX_LINE_WIDTH = 2;
const PREDICT_BOUNDING_BOX_STYLE = 'rgb(0,0,255)';

function drawBoundingBoxes(canvas, trueBoundingBox, predictBoundingBox) {
  tf.util.assert(
      trueBoundingBox != null && trueBoundingBox.length === 4,
      `trueBoundingBox의 길이는 4를 기대합니다, ` +
          `하지만 현재는 ${trueBoundingBox}입니다.`);
  tf.util.assert(
      predictBoundingBox != null && predictBoundingBox.length === 4,
      `predictBoundingBox의 길이는 4를 기대합니다, ` +
          `하지만 현재는 ${trueBoundingBox}입니다.`);

  let left = trueBoundingBox[0];
  let right = trueBoundingBox[1];
  let top = trueBoundingBox[2];
  let bottom = trueBoundingBox[3];

  const ctx = canvas.getContext('2d');
  ctx.beginPath();
  ctx.strokeStyle = TRUE_BOUNDING_BOX_STYLE;
  ctx.lineWidth = TRUE_BOUNDING_BOX_LINE_WIDTH;
  ctx.moveTo(left, top);
  ctx.lineTo(right, top);
  ctx.lineTo(right, bottom);
  ctx.lineTo(left, bottom);
  ctx.lineTo(left, top);
  ctx.stroke();

  ctx.font = '15px Arial';
  ctx.fillStyle = TRUE_BOUNDING_BOX_STYLE;
  ctx.fillText('true', left, top);

  left = predictBoundingBox[0];
  right = predictBoundingBox[1];
  top = predictBoundingBox[2];
  bottom = predictBoundingBox[3];

  ctx.beginPath();
  ctx.strokeStyle = PREDICT_BOUNDING_BOX_STYLE;
  ctx.lineWidth = PREDICT_BOUNDING_BOX_LINE_WIDTH;
  ctx.moveTo(left, top);
  ctx.lineTo(right, top);
  ctx.lineTo(right, bottom);
  ctx.lineTo(left, bottom);
  ctx.lineTo(left, top);
  ctx.stroke();

  ctx.font = '15px Arial';
  ctx.fillStyle = PREDICT_BOUNDING_BOX_STYLE;
  ctx.fillText('predicted', left, bottom);
}

/**
 * 입력 이미지를 합성하고 추론을 수행한 다음 결과를 시각화합니다.
 *
 * @param {tf.Model} model 추론에 사용할 모델
 */
async function runAndVisualizeInference(model) {
  // 입력 이미지를 합성하고 캔바스에 그립니다.
  const synth = new ObjectDetectionImageSynthesizer(canvas, tf);

  const numExamples = 1;
  const numCircles = 10;
  const numLineSegments = 10;
  const {images, targets} = await synth.generateExampleBatch(
      numExamples, numCircles, numLineSegments);

  const t0 = tf.util.now();
  // 모델의 추론을 수행합니다.
  const modelOut = await model.predict(images).data();
  inferenceTimeMs.textContent = `${(tf.util.now() - t0).toFixed(1)}`;

  // 정답 바운딩 박스와 예측한 바운딩 박스를 그립니다.
  const targetsArray = Array.from(await targets.data());
  const boundingBoxArray = targetsArray.slice(1);
  drawBoundingBoxes(canvas, boundingBoxArray, modelOut.slice(1));

  // 객체의 정답 클래스와 예측 클래스를 출력합니다.
  const trueClassName = targetsArray[0] > 0 ? '직사각형' : '삼각형';
  trueObjectClass.textContent = trueClassName;

  // 모델이 객체의 예측 클래스를 나타내는 숫자를 출력합니다.
  // 삼각형은 0이고 직사각형은 224(canvas.width)를 예측하도록 훈련됩니다.
  // 이는 모델이 클래스 손실과 바운딩 박스 손실을 합쳐서 하나의 손실 값을 만들기 위해서입니다.
  // 따라서 추론 시에는 224(canvas.width)의 절반이 임곗값이 됩니다.
  const shapeClassificationThreshold = canvas.width / 2;
  const predictClassName =
      (modelOut[0] > shapeClassificationThreshold) ? '직사각형' : '삼각형';
  predictedObjectClass.textContent = predictClassName;

  if (predictClassName === trueClassName) {
    predictedObjectClass.classList.remove('shape-class-wrong');
    predictedObjectClass.classList.add('shape-class-correct');
  } else {
    predictedObjectClass.classList.remove('shape-class-correct');
    predictedObjectClass.classList.add('shape-class-wrong');
  }

  // 텐서 메모리 삭제
  tf.dispose([images, targets]);
}

async function init() {
  const LOCAL_MODEL_PATH = 'object_detection_model/model.json';
  const HOSTED_MODEL_PATH =
      'https://storage.googleapis.com/tfjs-examples/simple-object-detection/dist/object_detection_model/model.json';

  // 로컬에 저장된 모델을 로드합니다.
  // 실패시 "호스팅된 모델을 로드합니다" 버튼을 활성화합니다.
  let model;
  try {
    model = await tf.loadLayersModel(LOCAL_MODEL_PATH);
    model.summary();
    testModel.disabled = false;
    status.textContent = '로컬에 저장된 모델을 로드했습니다! 이제 "모델 테스트"를 클릭하세요!';
    runAndVisualizeInference(model);
  } catch (err) {
    status.textContent = '로컬에 저장된 모델을 로드하는데 실패했습니다. ' +
        '"호스팅된 모델을 로드합니다"를 클릭하세요';
    loadHostedModel.disabled = false;
  }

  loadHostedModel.addEventListener('click', async () => {
    try {
      status.textContent = `${HOSTED_MODEL_PATH}에서 모델을 로드하는 중...`;
      model = await tf.loadLayersModel(HOSTED_MODEL_PATH);
      model.summary();
      loadHostedModel.disabled = true;
      testModel.disabled = false;
      status.textContent =
          `호스팅된 모델을 로드했습니다! 이제 "모델 테스트"를 클릭하세요!`;
      runAndVisualizeInference(model);
    } catch (err) {
      status.textContent =
          `${HOSTED_MODEL_PATH}에서 모델을 로드하는데 실패했습니다.`;
    }
  });

  testModel.addEventListener('click', () => runAndVisualizeInference(model));
}

init();
