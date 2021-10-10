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

const fs = require('fs');
const path = require('path');

const argparse = require('argparse');
const canvas = require('canvas');
const tf = require('@tensorflow/tfjs');
const synthesizer = require('./synthetic_images');

const CANVAS_SIZE = 224;  // MobileNet의 입력 크기와 같아야 합니다.

// 미세 튜닝할 때 동결 해제할 층의 이름
const topLayerGroupNames = ['conv_pw_9', 'conv_pw_10', 'conv_pw_11'];

// 헤드가 없는 베이스 모델의 최상위 층 이름
const topLayerName =
    `${topLayerGroupNames[topLayerGroupNames.length - 1]}_relu`;

// `yTrue`의 첫 번째 열(0-1 도형 표시자)의 스케일을 조정하여 도형과 바운딩 박스를 합친
// 최종 손실 값에 공평하게 기여하게 만듭니다.
const LABEL_MULTIPLIER = [CANVAS_SIZE, 1, 1, 1, 1];

/**
 * 객체 탐지를 위한 사용자 정의 손실 함수
 *
 * 이 손실 함수는 손실 두 개를 합한 것입니다.
 * - 도형 손실은 binaryCrossentropy로 계산하고 바운딩 박스 손실의 스케일에 맞추기 위해
 *   `classLossMultiplier`를 곱합니다.
 * - 바운딩 박스 손실은 정답 바운딩 박스와 예측한 바운딩 박스 사이의 meanSquaredError로 계산합니다.
 * @param {tf.Tensor} yTrue 진짜 레이블. 크기: [batchSize, 5].
 *   첫 번째 열은 타깃이 삼각형(0) 또는 직사각형(1)인지
 *   나타내는 0-1 표시자입니다. 남은 네 열은 도형의 바운딩 박스입니다(픽셀 단위):
 *   [왼쪽, 오른쪽, 위, 아래]
 *   바운딩 박스 크기는 [0, CANVAS_SIZE) 범위입니다.
 * @param {tf.Tensor} yPred 예측한 레이블. 크기: `yTrue`와 같음.
 * @return {tf.Tensor} 손실 값
 */
function customLossFunction(yTrue, yPred) {
  return tf.tidy(() => {
    // `yTrue`의 첫 번째 열(0-1 도형 표시자)의 스케일을 조정하여 도형과 바운딩 박스를 합친
    // 최종 손실 값에 공평하게 기여하게 만듭니다.
    return tf.metrics.meanSquaredError(yTrue.mul(LABEL_MULTIPLIER), yPred);
  });
}

/**
 * MobileNet을 로드하고 헤드를 제거한 다음 모든 층을 동결합니다.
 *
 * 헤드 삭제와 층 동결은 전이 학습을 위한 준비 작업입니다.
 *
 * 또한 미세 튜닝 단계에서 동결 해제할 층 이름을 구합니다.
 *
 * @return {tf.Model} 모든 층이 동결된 헤드가 없는 MobileNet
 */
async function loadTruncatedBase() {
  const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // 중간 활성화 값을 출력하는 모델을 반환합니다.
  const fineTuningLayers = [];
  const layer = mobilenet.getLayer(topLayerName);
  const truncatedBase =
      tf.model({inputs: mobilenet.inputs, outputs: layer.output});
  // 모델의 층을 동결합니다.
  for (const layer of truncatedBase.layers) {
    layer.trainable = false;
    for (const groupName of topLayerGroupNames) {
      if (layer.name.indexOf(groupName) === 0) {
        fineTuningLayers.push(layer);
        break;
      }
    }
  }

  tf.util.assert(
      fineTuningLayers.length > 1,
      `${topLayerGroupNames}로 시작하는 층을 찾지 못했습니다.`);
  return {truncatedBase, fineTuningLayers};
}

/**
 * 객체 탐지를 위해 헤드가 없는 베이스 모델 위에 놓을 새로운 헤드를 만듭니다.
 *
 * @param {tf.Shape} inputShape 새로운 모델의 입력 크기
 * @returns {tf.Model} 새로운 모델의 헤드
 */
function buildNewHead(inputShape) {
  const newHead = tf.sequential();
  newHead.add(tf.layers.flatten({inputShape}));
  newHead.add(tf.layers.dense({units: 200, activation: 'relu'}));
  // 다섯 개의 출력 유닛:
  //   - 첫 번짼느 도형 표시자: 타깃 도형이 삼각형인지 직사각형인지 예측합니다.
  //   - 남은 네 개의 유닛은 바운딩 박스 예측에 사용됩니다:
  //     픽셀 단위의 [왼쪽, 오른쪽, 위, 아래]
  newHead.add(tf.layers.dense({units: 5}));
  return newHead;
}

/**
 * MobileNet으로 객체 탐지 모델 만들기
 *
 * @returns {[tf.Model, tf.layers.Layer[]]}
 *   1. 간단한 객체 탐지를 위해 새로 만든 모델
 *   2. 미세 튜닝 단계에서 동결 해제할 층
 */
async function buildObjectDetectionModel() {
  const {truncatedBase, fineTuningLayers} = await loadTruncatedBase();

  // 새로운 헤드 모델을 만듭니다
  const newHead = buildNewHead(truncatedBase.outputs[0].shape.slice(1));
  const newOutput = newHead.apply(truncatedBase.outputs[0]);
  const model = tf.model({inputs: truncatedBase.inputs, outputs: newOutput});

  return {model, fineTuningLayers};
}

(async function main() {
  // 데이터 관련 설정
  const numCircles = 10;
  const numLines = 10;

  const parser = new argparse.ArgumentParser();
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: '훈련에 tfjs-node-gpu를 사용합니다(CUDA와 CuDNN 필요)'
  });
  parser.addArgument(
      '--numExamples',
      {type: 'int', defaultValue: 2000, help: '훈련 샘플 개수'});
  parser.addArgument('--validationSplit', {
    type: 'float',
    defaultValue: 0.15,
    help: '훈련에 사용할 검증 세트 비율'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 128,
    help: '훈련에 사용할 배치 크기'
  });
  parser.addArgument('--initialTransferEpochs', {
    type: 'int',
    defaultValue: 100,
    help: '전이 학습 초기 단계에서 수행할 훈련 에포크 횟수'
  });
  parser.addArgument('--fineTuningEpochs', {
    type: 'int',
    defaultValue: 100,
    help: '미세 튜닝 단계에서 수행할 훈련 에포크 횟수'
  });
  parser.addArgument('--logDir', {
    type: 'string',
    help: '텐서보드 로그 디렉토리. 모델이 훈련할 때 손실 값을 기록합니다.'
  });
  parser.addArgument('--logUpdateFreq', {
    type: 'string',
    defaultValue: 'batch',
    optionStrings: ['batch', 'epoch'],
    help: '텐서보드로 손실을 기록할 주기'
  });
  const args = parser.parseArgs();

  let tfn;
  if (args.gpu) {
    console.log('GPU로 훈련합니다.');
    tfn = require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('CPU로 훈련합니다.');
    tfn = require('@tensorflow/tfjs-node');
  }

  const modelSaveURL = 'file://./dist/object_detection_model';

  const tBegin = tf.util.now();
  console.log(`${args.numExamples}개의 훈련 샘플 생성 중...`);
  const synthDataCanvas = canvas.createCanvas(CANVAS_SIZE, CANVAS_SIZE);
  const synth =
      new synthesizer.ObjectDetectionImageSynthesizer(synthDataCanvas, tf);
  const {images, targets} =
      await synth.generateExampleBatch(args.numExamples, numCircles, numLines);

  const {model, fineTuningLayers} = await buildObjectDetectionModel();
  model.compile({loss: customLossFunction, optimizer: tf.train.rmsprop(5e-3)});
  model.summary();

  // 전이 학습의 초기 단계
  console.log('단계 1 / 2: 초기 전이 학습');
  await model.fit(images, targets, {
    epochs: args.initialTransferEpochs,
    batchSize: args.batchSize,
    validationSplit: args.validationSplit,
    callbacks: args.logDir == null ? null : tfn.node.tensorBoard(args.logDir, {
      updateFreq: args.logUpdateFreq
    })
  });

  // 전이 학습의 미세 튜닝 단계
  // 미세 튜닝을 위해 층을 동결 해제합니다.
  for (const layer of fineTuningLayers) {
    layer.trainable = true;
  }
  model.compile({loss: customLossFunction, optimizer: tf.train.rmsprop(2e-3)});
  model.summary();

  // 미세 튜닝을 수행합니다.
  // CPU/GPU 메모리 부족을 피하기 위해 배치 크기를 줄입니다. This has
  // 이는 미세 튜닝에서 층을 동결 해제한 것과 관련이 있습니다.
  // 역전파 동안에 많은 메모리를 소모하게 됩니다.
  console.log('단계 2 / 2: 미세 튜닝 단계');
  await model.fit(images, targets, {
    epochs: args.fineTuningEpochs,
    batchSize: args.batchSize / 2,
    validationSplit: args.validationSplit,
    callbacks: args.logDir == null ? null : tfn.node.tensorBoard(args.logDir, {
      updateFreq: args.logUpdateFreq
    })
  });

  // 모델을 저장합니다.
  // 먼저 기본 디렉토리 위치를 확인합니다.
  const modelSavePath = modelSaveURL.replace('file://', '');
  const dirName = path.dirname(modelSavePath);
  if (!fs.existsSync(dirName)) {
    fs.mkdirSync(dirName);
  }
  await model.save(modelSaveURL);
  console.log(`모델 훈련 시간: ${(tf.util.now() - tBegin) / 1e3} s`);
  console.log(`훈련된 모델 저장 위치: ${modelSaveURL}`);
  console.log(
      `\n이제 다음으로 브라우저에서 모델을 테스트하려면 다음 명령을 실행하세요:`);
  console.log(`\n  cd ..; npx http-server`);
})();
