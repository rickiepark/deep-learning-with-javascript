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
 * TensorFlow.js 예제: LSTM 텍스트 생성
 *
 * 다음을 참고했습니다.
 * -
 * https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
 * - Andrej Karpathy. "The Unreasonable Effectiveness of Recurrent Neural
 * Networks" http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 */

import {TextData} from './data.js';
import * as model from './model.js';
import {onTextGenerationBegin, onTextGenerationChar, onTrainBatchEnd, onTrainBegin, onTrainEpochEnd, setUpUI} from './ui.js';

/**
 * LSTM 기반의 텍스트 생성을 관리하는 클래스
 *
 * 다음과 같은 작업을 수행합니다:
 *
 * - tf.layers API로 LSTM 모델을 만들고 주어진 입력 문자 시퀀스에서 다음 문자를 예측하도록 훈련합니다.
 * - LSTM 모델을 사용해 랜덤한 텍스트를 생성합니다.
 */
export class LSTMTextGenerator {
  /**
   * NeuralNetworkTextGenerator의 생성자
   *
   * @param {TextData} textData `TextData` 객체
   */
  constructor(textData) {
    this.textData_ = textData;
    this.charSetSize_ = textData.charSetSize();
    this.sampleLen_ = textData.sampleLen();
    this.textLen_ = textData.textLen();
  }

  /**
   * LSTM 모델을 만듭니다.
   *
   * @param {number | number[]} lstmLayerSizes LSTM 층의 크기. 하나의 숫자 또는 숫자 배열.
   */
  createModel(lstmLayerSizes) {
    this.model = model.createModel(
        this.sampleLen_, this.charSetSize_, lstmLayerSizes);
  }

  /**
   * 훈련을 위해 모델을 컴파일합니다.
   *
   * @param {number} learningRate 훈련에 사용할 학습률
   */
  compileModel(learningRate) {
    model.compileModel(this.model, learningRate);
  }

  /**
   * LSTM 모델을 훈련합니다.
   *
   * @param {number} numEpochs 모델을 훈련할 에포크 횟수
   * @param {number} examplesPerEpoch 에포크에서 사용할 샘플 수
   * @param {number} batchSize 훈련에 사용할 배치 크기
   * @param {number} validationSplit 훈련에 사용할 검증 세트 비율
   */
  async fitModel(numEpochs, examplesPerEpoch, batchSize, validationSplit) {
    let batchCount = 0;
    const batchesPerEpoch = examplesPerEpoch / batchSize;
    const totalBatches = numEpochs * batchesPerEpoch;
    let t = new Date().getTime();

    onTrainBegin();
    const callbacks = {
      onBatchEnd: async (batch, logs) => {
        // 초당 샘플 수로 현재 배치의 훈련 속도를 계산합니다.
        const t1 = new Date().getTime();
        const examplesPerSec = batchSize / ((t1 - t) / 1e3);
        t = t1;
        onTrainBatchEnd(logs, ++batchCount / totalBatches, examplesPerSec);
      },
      onEpochEnd: async (epoch, logs) => {
        onTrainEpochEnd(logs);
      }
    };

    await model.fitModel(
        this.model, this.textData_, numEpochs, examplesPerEpoch, batchSize,
        validationSplit, callbacks);
  }

  /**
   * LSTM 모델을 사용해 텍스트를 생성합니다.
   *
   * @param {number[]} sentenceIndices 문자 인덱스로 표현된 시드 문장
   * @param {number} length 생성할 텍스트 길이(문자 개수)
   * @param {number} temperature 온도 파라미터. 0보다 커야 합니다.
   * @returns {string} 생성된 텍스트
   */
  async generateText(sentenceIndices, length, temperature) {
    onTextGenerationBegin();
    return await model.generateText(
        this.model, this.textData_, sentenceIndices, length, temperature,
        onTextGenerationChar);
  }
};

/**
 * 모델 저장과 로딩을 위한 LSTMTextGenerator의 서브 클래스
 *
 * 모델을 브라우저의 IndexedDB에 저장하고 로드합니다.
 */
export class SaveableLSTMTextGenerator extends LSTMTextGenerator {
  /**
   * NeuralNetworkTextGenerator의 생성자
   *
   * @param {TextData} textData `TextData` 객체
   */
  constructor(textData) {
    super(textData);
    this.modelIdentifier_ = textData.dataIdentifier();
    this.MODEL_SAVE_PATH_PREFIX_ = 'indexeddb://lstm-text-generation';
    this.modelSavePath_ =
        `${this.MODEL_SAVE_PATH_PREFIX_}/${this.modelIdentifier_}`;
  }

  /**
   * 모델 식별자 가져오기
   *
   * @returns {string} 모델 식별자
   */
  modelIdentifier() {
    return this.modelIdentifier_;
  }

  /**
   * LSTM 모델이 로컬에 저장되어 있으면 로드하고 그렇지 않으면 만듭니다.
   *
   * @param {number | number[]} lstmLayerSizes LSTM 층의 크기. 하나의 숫자 또는 숫자 배열.
   */
  async loadModel(lstmLayerSizes) {
    const modelsInfo = await tf.io.listModels();
    if (this.modelSavePath_ in modelsInfo) {
      console.log(`기존 모델 로딩 중...`);
      this.model = await tf.loadLayersModel(this.modelSavePath_);
      console.log(`${this.modelSavePath_}에서 모델을 로드했습니다.`);
    } else {
      throw new Error(
          `${this.modelSavePath_}에서 모델을 찾을 수 없습니다. ` +
          `새로 모델을 만듭니다.`);
    }
  }

  /**
   * IndexedDB에 모델을 저장합니다.
   *
   * @returns 저장이 성공하면서 반환된 ModelInfo
   */
  async saveModel() {
    if (this.model == null) {
      throw new Error('모델을 만들기 전에 저장할 수 없습니다.');
    } else {
      return await this.model.save(this.modelSavePath_);
    }
  }

  /**
   * IndexedDB에 저장된 모델을 삭제합니다.
   */
  async removeModel() {
    if (await this.checkStoredModelStatus() == null) {
      throw new Error(
          '로컬에 저장된 모델이 없기 때문에 삭제할 수 없습니다.');
    }
    return await tf.io.removeModel(this.modelSavePath_);
  }

  /**
   * 로컬에 저장된 모델 상태를 체크합니다.
   *
   * @returns 로컬에 저장된 모델이 있다면 모델 정보를 JSON 객체로 반환하고 그렇지 않으면 `undefined`를 반환합니다.
   */
  async checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[this.modelSavePath_];
  }

  /**
   * 모델에 있는 LSTM 층의 크기를 가져옵니다.
   *
   * @returns {number | number[]} 모델의 LSTM 층의 크기 (즉, 유닛 개수)
   *   하니의 LSTM 층만 있다면 하나의 숫자가 반환됩니다. 그렇지 않으면 숫자 배열이 반환됩니다. is returned.
   */
  lstmLayerSizes() {
    if (this.model == null) {
      throw new Error('먼저 모델을 만드세요.');
    }
    const numLSTMLayers = this.model.layers.length - 1;
    const layerSizes = [];
    for (let i = 0; i < numLSTMLayers; i++) {
      layerSizes.push(this.model.layers[i].cell.units);
    }
    return layerSizes.length === 1 ? layerSizes[0] : layerSizes;
  }
}

setUpUI();
