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

import {getDatasetStats, getNormalizedDatasets} from './data_housing';
import {compileModel} from './model_housing';

// `--gpu` 플래그 지정에 따라 동적으로 tf를 임포트합니다.
let tf;

function parseArgs() {
  const parser = new argparse.ArgumentParser({
    description: 'TensorFlow.js 양자화 예제: 캘리포니아 주택 가격 모델 평가하기',
    addHelp: true
  });
  parser.addArgument('modelSavePath', {
    type: 'string',
    help: '평가할 모델을 저장할 경로'
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

  const validationSplit = 0.2;
  const evaluationSplit = 0.1;
  const {evalXs, evalYs} =
      await getNormalizedDatasets(
          count, featureMeans, featureStddevs, labelMean, labelStddev,
          validationSplit, evaluationSplit);

  console.log(`${args.modelSavePath}에서 모델 로딩 중...`);
  const model = await tf.loadLayersModel(`file://${args.modelSavePath}`);
  compileModel(model);

  console.log(`평가 수행 중...`);
  const t0 = tf.util.now();
  const evalOutput = model.evaluate(evalXs, evalYs);
  const t1 = tf.util.now();
  console.log(`\n평가 시간: ${(t1 - t0).toFixed(2)} ms.`);
  console.log(
      `\n평가 결과:\n` +
      `  손실 = ${evalOutput.dataSync()[0].toFixed(6)}`);
}

if (require.main === module) {
  main();
}
