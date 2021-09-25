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

const tf = require('@tensorflow/tfjs-node');
const argparse = require('argparse');

const data = require('./data');
const model = require('./model');

async function run(epochs, batchSize, modelSavePath) {
  await data.loadData();

  const {images: trainImages, labels: trainLabels} = data.getTrainData();
  model.summary();

  let epochBeginTime;
  let millisPerStep;
  const validationSplit = 0.15;
  const numTrainExamplesPerEpoch =
      trainImages.shape[0] * (1 - validationSplit);
  const numTrainBatchesPerEpoch =
      Math.ceil(numTrainExamplesPerEpoch / batchSize);
  await model.fit(trainImages, trainLabels, {
    epochs: epochs,
    batchSize: batchSize,
    validationSplit: validationSplit
  });

  const {images: testImages, labels: testLabels} = data.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log(
      `\n평가 결과:\n` +
      `  손실 = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
      `정확도 = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`다음 경로에 모델을 저장합니다: ${modelSavePath}`);
  }
}

const parser = new argparse.ArgumentParser({
  description: 'TensorFlow.js-Node MNIST 예제',
  addHelp: true
});
parser.addArgument('--epochs', {
  type: 'int',
  defaultValue: 20,
  help: '모델을 훈련할 에포크 횟수'
});
parser.addArgument('--batch_size', {
  type: 'int',
  defaultValue: 128,
  help: '모델 훈련에 사용할 배치 크기'
})
parser.addArgument('--model_save_path', {
  type: 'string',
  help: '훈련이 끝난 후 모델을 저장할 경로'
});
const args = parser.parseArgs();

run(args.epochs, args.batch_size, args.model_save_path);
