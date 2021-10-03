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

import {ControllerDataset} from './controller_dataset.js';
import * as ui from './ui.js';

// 예측하려는 클래스 개수.
// 이 예에서는 위, 아래, 왼쪽, 오른쪽에 해당하는 네 개의 클래스를 예측합니다.
const NUM_CLASSES = 4;

// 웹캠 이미지로부터 텐서를 생성합니다.
let webcam;

// 활성화 출력을 저장할 데이터셋 객체
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let truncatedMobileNet;
let model;

// MobileNet을 로드하고 분류 모델의 입력으로 사용할 중간 활성화를 출력하는 모델을 반환합니다.
async function loadTruncatedMobileNet() {
  const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // 중간 층의 활성화를 출력하는 모델을 반환합니다.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// UI 버튼을 누를 때 웹캠에서 프레임을 읽어 해당 클래스에 연결합니다.
// 위, 아래, 왼쪽, 오른쪽의 레이블이 각각 0, 1, 2, 3입니다.
ui.setExampleHandler(async label => {
  let img = await getImage();

  controllerDataset.addExample(truncatedMobileNet.predict(img), label);

  // 프리뷰 썸네일을 그립니다.
  ui.drawThumb(img, label);
  img.dispose();
})

/**
 * 분류기를 준비하고 훈련합니다.
 */
async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('훈련하기 전에 샘플을 추가하세요!');
  }

  // 두 개 층을 가진 완전 연결 신경망을 만듭니다.
  // MobileNet 모델에 층을 추가하지 않고 별도의 모델을 만듦으로써
  // MobileNet 모델의 가중치를 동결하고 새로운 모델의 가중치만 훈련합니다.
  model = tf.sequential({
    layers: [
      // 밀집 층에 사용할 수 있도록 입력을 벡터로 펼칩니다.
      // 기술적으로는 층이지만 텐서의 크기 변경만 수행합니다(훈련 파라미터가 없습니다).
      tf.layers.flatten(
          {inputShape: truncatedMobileNet.outputs[0].shape.slice(1)}),
      // 층 1.
      tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // 층 2. 마지막 층의 유닛 개수는 예측하려는 클래스 개수와 같아야 합니다.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // 모델 훈련을 위한 옵티마이저를 만듭니다.
  const optimizer = tf.train.adam(ui.getLearningRate());
  // 예측 확률 분포(입력이 각 클래스에 속할 확률)와 레이블(정답 클래스는 100% 확률) 사이의
  // 에러를 측정하기 위해 다중 분류에서 사용하는 손실 함수인
  // categoricalCrossentropy를 사용합니다.
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // 전체 데이터셋의 일정 비율을 배치 크기로 설정합니다. 수집된 샘플의 개수는
  // 사용자가 얼마나 많은 샘플을 모으는지에 따라 다르기 때문입니다.
  // 이렇게 하면 배치 크기를 유연하게 설정할 수 있습니다.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `배치 크기 비율이 0 또는 NaN입니다. 0이 아닌 비율을 선택하세요.`);
  }

  // 모델 훈련! Model.fit()이 xs & ys을 섞으므로 직접 섞을 필요가 없습니다.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus('손실: ' + logs.loss.toFixed(5));
      }
    }
  });
}

let isPredicting = false;

async function predict() {
  ui.isPredicting();
  while (isPredicting) {
    // 웹캠에서 프레임을 캡쳐합니다.
    const img = await getImage();

    // MobileNet의 중간층 활성화를 예측으로 출력합니다.
    // 즉, 입력 이미지의 "임베딩"을 출력합니다.
    const embeddings = truncatedMobileNet.predict(img);

    // MobileNet에서 출력한 임베딩을 입력으로 사용해 새로 훈련된 모델에서 예측을 만듭니다.
    const predictions = model.predict(embeddings);

    // 최대 확률을 가진 인덱스를 찾습니다.
    // 이 숫자가 모델이 생각하는 입력에 대해 가장 가능성 있는 클래스입니다.
    const predictedClass = predictions.as1D().argMax();
    const classId = (await predictedClass.data())[0];
    img.dispose();

    ui.predictClass(classId);
    await tf.nextFrame();
  }
  ui.donePredicting();
}

/**
 * 웹캠에서 프레임을 캡쳐하고 -1과 1 사이로 정규화합니다.
 * [1, w, h, c] 크기의 배치 이미지(샘플이 1개인 배치)를 반환합니다.
 */
async function getImage() {
  const img = await webcam.capture();
  const processedImg =
      tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
  img.dispose();
  return processedImg;
}

document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('훈련중...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
});
document.getElementById('predict').addEventListener('click', () => {
  ui.startPacman();
  isPredicting = true;
  predict();
});

async function init() {
  try {
    webcam = await tf.data.webcam(document.getElementById('webcam'));
  } catch (e) {
    console.log(e);
    document.getElementById('no-webcam').style.display = 'block';
  }
  truncatedMobileNet = await loadTruncatedMobileNet();

  ui.init();

  // 모델을 시운전 합니다. GPU에 가중치를 업로드하고 WebGL 프로그램을 컴파일합니다.
  // 이렇게 하면 처음 웹캠에서 데이터를 수집할 때 속도가 빨라집니다.
  const screenShot = await webcam.capture();
  truncatedMobileNet.predict(screenShot.expandDims(0));
  screenShot.dispose();
}

// 애플리케이션을 초기화합니다.
init();
