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

if (typeof tf === 'undefined') {
  global.tf = require('@tensorflow/tfjs');
}
import * as dateFormat from './date_format.js';

/**
 * RNN 순차 출력의 마지막 타임 스텝을 얻기 위한 사용자 정의 층
 */
class GetLastTimestepLayer extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.supportMasking = true;
  }

  computeOutputShape(inputShape) {
    const outputShape = inputShape.slice();
    outputShape.splice(outputShape.length - 2, 1);
    return outputShape;
  }

  call(input) {
    if (Array.isArray(input)) {
      input = input[0];
    }
    const inputRank = input.shape.length;
    tf.util.assert(inputRank === 3, `잘못된 입력 랭크: ${inputRank}`);
    return input.gather([input.shape[1] - 1], 1).squeeze([1]);
  }

  static get className() {
    return 'GetLastTimestepLayer';
  }
}
tf.serialization.registerClass(GetLastTimestepLayer);

/**
 * 날짜 변환을 위한 LSTM 기반의 어텐션 모델을 만듭니다.
 *
 * @param {number} inputVocabSize 입력 어휘 사전 크기. 패딩 기호를 포함합니다.
 *   이 모델의 경우 어휘 사전은 입력 날짜 문자열에 등장하는 모든 고유한 문자의 집합입니다.
 * @param {number} outputVocabSize 출력 어휘 사전 크기. 패딩과 시작 기호를 포함합니다.
 *   이 모델의 경우 어휘 사전은 출력 날짜 문자열에 등장하는 모든 고유한 문자의 집합입니다.
 * @param {number} inputLength 최대 입력 길이(문자 개수).
 *   이 길이보다 짧은 입력 시퀀스는 끝에 패딩이 추가되어야 합니다.
 * @param {number} outputLength 출력 길이(문자 개수).
 * @return {tf.Model} 컴파일된 모델 객체
 */
export function createModel(
    inputVocabSize, outputVocabSize, inputLength, outputLength) {
  const embeddingDims = 64;
  const lstmUnits = 64;

  const encoderInput = tf.input({shape: [inputLength]});
  const decoderInput = tf.input({shape: [outputLength]});

  let encoder = tf.layers.embedding({
    inputDim: inputVocabSize,
    outputDim: embeddingDims,
    inputLength,
    maskZero: true
  }).apply(encoderInput);
  encoder = tf.layers.lstm({
    units: lstmUnits,
    returnSequences: true
  }).apply(encoder);

  const encoderLast = new GetLastTimestepLayer({
    name: 'encoderLast'
  }).apply(encoder);

  let decoder = tf.layers.embedding({
    inputDim: outputVocabSize,
    outputDim: embeddingDims,
    inputLength: outputLength,
    maskZero: true
  }).apply(decoderInput);
  decoder = tf.layers.lstm({
    units: lstmUnits,
    returnSequences: true
  }).apply(decoder, {initialState: [encoderLast, encoderLast]});

  let attention = tf.layers.dot({axes: [2, 2]}).apply([decoder, encoder]);
  attention = tf.layers.activation({
    activation: 'softmax',
    name: 'attention'
  }).apply(attention);

  const context = tf.layers.dot({
    axes: [2, 1],
    name: 'context'
  }).apply([attention, encoder]);
  const decoderCombinedContext =
      tf.layers.concatenate().apply([context, decoder]);
  let output = tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: lstmUnits,
      activation: 'tanh'
    })
  }).apply(decoderCombinedContext);
  output = tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: outputVocabSize,
      activation: 'softmax'
    })
  }).apply(output);

  const model = tf.model({
    inputs: [encoderInput, decoderInput],
    outputs: output
  });
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam'
  });
  return model;
}

/**
 * 날짜 변환을 위해 시퀀스-투-시퀀스 디코딩을 수행합니다.
 *
 * @param {tf.Model} model 시퀀스-투-시퀀스 디코딩을 위해 사용할 모델.
 *   두 개의 입력:
 *   1. `[numExamples, inputLength]` 크기의 인코더 입력
 *   2. `[numExamples, outputLength]` 크기의 디코더 입력
 *   하나의 출력:
 *   1. `[numExamples, outputLength, outputVocabularySize]` 크기의 디코더 소프트맥스 확률 출력
 * @param {string} inputStr 변환할 입력 날짜 문자열
 * @return {{outputStr: string, attention?: tf.Tensor}}
 *   - `outputStr` 필드는 출력 날짜 문자열입니다.
 *   - `getAttention`가 `true`이면, `[]` 크기의 `float32` `tf.Tensor`로 `attention` 필드가 채워집니다.
 */
export async function runSeq2SeqInference(
    model, inputStr, getAttention = false) {
  return tf.tidy(() => {
    const encoderInput = dateFormat.encodeInputDateStrings([inputStr]);
    const decoderInput = tf.buffer([1, dateFormat.OUTPUT_LENGTH]);
    decoderInput.set(dateFormat.START_CODE, 0, 0);

    for (let i = 1; i < dateFormat.OUTPUT_LENGTH; ++i) {
      const predictOut = model.predict(
          [encoderInput, decoderInput.toTensor()]);
      const output = predictOut.argMax(2).dataSync()[i - 1];
      predictOut.dispose();
      decoderInput.set(output, 0, i);
    }

    const output = {outputStr: ''};

    // 어텐션 행렬을 반환하는지에 따라 마지막 타임 스텝에 사용할 `tf.Model` 객체가 달라집니다.
    let finalStepModel = model;
    if (getAttention) {
      // 어텐션 행렬을 반환하려면 두 개의 출력을 가진 모델을 만듭니다.
      // - 첫 번째 출력은 원본 디코더 출력입니다.
      // - 두 번째 출력은 어텐션 행렬입니다.
      finalStepModel = tf.model({
        inputs: model.inputs,
        outputs: model.outputs.concat([model.getLayer('attention').output])
      });
    }

    const finalPredictOut = finalStepModel.predict(
        [encoderInput, decoderInput.toTensor()]);
    let decoderFinalOutput;  // 디코더의 최종 출력
    if (getAttention) {
      decoderFinalOutput = finalPredictOut[0];
      output.attention = finalPredictOut[1];
    } else {
      decoderFinalOutput = finalPredictOut;
    }
    decoderFinalOutput =
    decoderFinalOutput.argMax(2).dataSync()[dateFormat.OUTPUT_LENGTH - 1];

    for (let i = 1; i < decoderInput.shape[1]; ++i) {
      output.outputStr += dateFormat.OUTPUT_VOCAB[decoderInput.get(0, i)];
    }
    output.outputStr += dateFormat.OUTPUT_VOCAB[decoderFinalOutput];
    return output;
  });
}
