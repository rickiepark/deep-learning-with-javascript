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

import * as argparse from 'argparse';
import * as fs from 'fs';
import * as path from 'path';
import * as shelljs from 'shelljs';

import {getDatasetStats, getNormalizedDatasets} from './data_housing';
import {createModel} from './model_housing';

// `--gpu` 플래그 지정에 따라 동적으로 tf를 임포트합니다.
let tf;

function parseArgs() {
  const parser = new argparse.ArgumentParser({
    description: 'TensorFlow.js 양자화 예제: 캘리포니아 주택 가격 모델 평가하기',
    addHelp: true
  });
  parser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 200,
    help: '모델을 훈련할 에포크 횟수'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 128,
    help: '훈련에 사용할 배치 크기'
  });
  parser.addArgument('--validationSplit', {
    type: 'float',
    defaultValue: 0.2,
    help: '훈련에 사용할 검증 세트 비율'
  });
  parser.addArgument('--evaluationSplit', {
    type: 'float',
    defaultValue: 0.1,
    help: '훈련 후 평가에 사용할 테스트 세트 비율'
  });
  parser.addArgument('--modelSavePath', {
    type: 'string',
    defaultValue: './models/housing/original',
    help: '훈련 후 모델을 저장할 경로'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'tfjs-node-gpu를 사용해 평가합니다(CUDA 가능 GPU, 지원 드라이버와 라이브러리가 필요).'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArgs();
  if (args.gpu) {
    tf = require('@tensorflow/tfjs-node-gpu');
  } else {
    tf = require('@tensorflow/tfjs-node');
  }

  const {count, featureMeans, featureStddevs, labelMean, labelStddev} =
      await getDatasetStats();
  const {trainXs, trainYs, valXs, valYs, evalXs, evalYs} =
      await getNormalizedDatasets(
          count, featureMeans, featureStddevs, labelMean, labelStddev,
          args.validationSplit, args.evaluationSplit);

  const model = createModel();
  model.summary();

  await model.fit(trainXs, trainYs,  {
    epochs: args.epochs,
    batchSize: args.batchSize,
    validationData: [valXs, valYs]
  });

  const evalOutput = model.evaluate(evalXs, evalYs);
  console.log(
      `\n평가 결과:\n` +
      `  손실 = ${evalOutput.dataSync()[0].toFixed(6)}`);

  if (args.modelSavePath != null) {
    if (!fs.existsSync(path.dirname(args.modelSavePath))) {
      shelljs.mkdir('-p', path.dirname(args.modelSavePath));
    }
    await model.save(`file://${args.modelSavePath}`);
    console.log(`모델 저장 경로: ${args.modelSavePath}`);
  }
}

if (require.main === module) {
  main();
}
