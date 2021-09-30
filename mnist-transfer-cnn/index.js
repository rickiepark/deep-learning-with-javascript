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

import * as loader from './loader.js';
import * as ui from './ui.js';
import * as util from './util.js';

const HOSTED_URLS = {
  model:
      'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json',
  train:
      'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/gte5.train.json',
  test:
      'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/gte5.test.json'
};

const LOCAL_URLS = {
  model: 'http://localhost:1235/resources/model.json',
  train: 'http://localhost:1235/resources/gte5.train.json',
  test: 'http://localhost:1235/resources/gte5.test.json'
};

class MnistTransferCNNPredictor {
  /**
   * MNIST CNN 전이 학습 데모 초기화
   */
  async init(urls) {
    this.urls = urls;
    this.model = await loader.loadHostedPretrainedModel(urls.model);

    // 모델을 로드한 후 요약 정보를 출력합니다.
    this.model.summary();
    tfvis.show.modelSummary(
        {name: 'Model Summary', tab: 'Model Info'}, this.model);

    this.imageSize = this.model.layers[0].batchInputShape[1];
    this.numClasses = 5;

    await this.loadRetrainData();
    this.prepTestExamples();

    return this;
  }

  async loadRetrainData() {
    ui.status('전이 학습을 위한 데이터를 로딩합니다...');
    this.gte5TrainData =
        await loader.loadHostedData(this.urls.train, this.numClasses);
    this.gte5TestData =
        await loader.loadHostedData(this.urls.test, this.numClasses);
    ui.status('전이 학습을 위한 데이터 로딩이 완료되었습니다.');
  }

  prepTestExamples() {
    // 인터랙티브한 테스트를 위해 몇 개의 MNIST 이미지 샘플을 하드코딩합니다.
    const testExamples = {};
    const digitCounts = {5: 0, 6: 0, 7: 0, 8: 0, 9: 0};
    const examplesPerDigit = 10;
    // `testExamples`에 5, 6, 7, 8, 9에서 한 샘플씩 넣습니다.
    for (let i = this.gte5TestData.data.length - 1; i >= 0; --i) {
      const datum = this.gte5TestData.data[i];
      const digit = datum.y + 5;
      if (digitCounts[digit] >= examplesPerDigit) {
        continue;
      }
      digitCounts[digit]++;
      const key = String(digit) + '_' + String(digitCounts[digit]);
      testExamples[key] = [];
      for (const row of datum.x) {
        testExamples[key] = testExamples[key].concat(row);
      }
      if (Object.keys(testExamples).length >= 5 * examplesPerDigit) {
        break;
      }
    }

    this.testExamples = testExamples;
  }

  // 로딩된 모델을 사용해 입력 이미지에 대한 예측을 수행합니다.
  predict(imageText) {
    tf.tidy(() => {
      try {
        const image = util.textToImageArray(imageText, this.imageSize);
        const predictOut = this.model.predict(image);
        const winner = predictOut.argMax(1);

        ui.setPredictResults(predictOut.dataSync(), winner.dataSync()[0] + 5);
      } catch (e) {
        ui.setPredictError(e.message);
      }
    });
  }

  // 로딩된 모델을 재훈련합니다.
  async retrainModel() {
    ui.status(
        '모델이 재훈련되는 동안 다른 것을 클릭하지 말고 기다리세요...',
        'blue');
    // 한 번 재훈련을 한 경우 모델을 다시 가져옵니다.
    if (this.already_retrained == undefined || this.already_retrained == True) {
      this.model = await loader.loadHostedPretrainedModel(this.urls.model);
    } else {
      this.already_retrained = True;
    }

    const trainingMode = ui.getTrainingMode();
    if (trainingMode === 'freeze-feature-layers') {
      console.log('모델의 특성 층을 동결합니다.');
      for (let i = 0; i < 7; ++i) {
        this.model.layers[i].trainable = false;
      }
    } else if (trainingMode === 'reinitialize-weights') {
      // 동일한 토폴로지의 모델을 만들지만 가중치를 다시 초기화합니다.
      const returnString = false;
      this.model = await tf.models.modelFromJSON({
        modelTopology: this.model.toJSON(null, returnString)
      });
    }
    this.model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: tf.train.adam(0.01),
      metrics: ['acc'],
    });

    // compile() 메서드 호출 후에 다시 summary() 메서드를 호출합니다.
    // 일부 모델의 가중치가 non-trainable로 바뀐 것을 볼 수 있습니다.
    this.model.summary();

    const batchSize = 128;
    const epochs = ui.getEpochs();

    const surfaceInfo = {name: trainingMode, tab: 'Transfer Learning'};
    console.log('model.fit() 실행 시작');
    await this.model.fit(this.gte5TrainData.x, this.gte5TrainData.y, {
      batchSize: batchSize,
      epochs: epochs,
      validationData: [this.gte5TestData.x, this.gte5TestData.y],
      callbacks: [
        ui.getProgressBarCallbackConfig(epochs),
        tfvis.show.fitCallbacks(surfaceInfo, ['val_loss', 'val_acc'], {
          zoomToFit: true,
          zoomToFitAccuracy: true,
          height: 200,
          callbacks: ['onEpochEnd'],
        }),
      ]
    });
    console.log('model.fit() 실행 완료');
  }
}

/**
 * 사전 훈련된 모델과 메타데이터를 로드합니다.
 * 화면에 예측과 재훈련 함수를 등록합니다.
 */
async function setupMnistTransferCNN() {
  if (await loader.urlExists(HOSTED_URLS.model)) {
    ui.status('모델 주소: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-pretrained-remote');
    button.addEventListener('click', async () => {
      const predictor = await new MnistTransferCNNPredictor().init(HOSTED_URLS);
      ui.prepUI(
          x => predictor.predict(x), () => predictor.retrainModel(),
          predictor.testExamples, predictor.imageSize);
    });
    button.style.display = 'inline-block';
  }

  if (await loader.urlExists(LOCAL_URLS.model)) {
    ui.status('모델 주소: ' + LOCAL_URLS.model);
    const button = document.getElementById('load-pretrained-local');
    button.addEventListener('click', async () => {
      const predictor = await new MnistTransferCNNPredictor().init(LOCAL_URLS);
      ui.prepUI(
          x => predictor.predict(x), () => predictor.retrainModel(),
          predictor.testExamples, predictor.imageSize);
    });
    button.style.display = 'inline-block';
  }

  ui.status('대기 중. 먼저 사전 훈련된 모델을 로드하세요.');
}

setupMnistTransferCNN();
