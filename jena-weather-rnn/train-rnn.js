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

/**
 * 온도 예측 문제를 위해 순환 신경망(RNN)을 훈련합니다.
 *
 * 이 스크립트 파일은 tfjs-node나 tfjs-node-gpu를 사용해 Node.js 환경에서 RNN을 훈련합니다.
 * (`--gpu` 플래그 참조)
 *
 * - 예나 데이터셋 로딩 방법은 data.js를 참고하세요.
 * - 자세한 모델 생성 방법과 훈련 로직은 models.js을 참고하세요.
 */

import {ArgumentParser} from 'argparse';

import {JenaWeatherData} from './data';
import {buildModel, getBaselineMeanAbsoluteError, trainModel} from './models';

global.fetch = require('node-fetch');

function parseArguments() {
  const parser =
      new ArgumentParser({description: '예나 날씨 문제를 위해 RNN을 훈련합니다.'});
  parser.addArgument('--modelType', {
    type: 'string',
    defaultValue: 'gru',
    optionStrings: ['baseline', 'gru', 'simpleRNN'],
    help: '훈련할 모델 종류. 상식 수준의 예측 오차를 계산하려면 "baseline"을 사용하세요'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'GPU 사용'
  });
  parser.addArgument('--lookBack', {
    type: 'int',
    defaultValue: 10 * 24 * 6,
    help: '특성 생성을 위해 사용할 과거 기간(행 개수)'
  });
  parser.addArgument('--step', {
    type: 'int',
    defaultValue: 6,
    help: '특성 생성을 위한 스텝 크기(행 개수)'
  });
  parser.addArgument('--delay', {
    type: 'int',
    defaultValue: 24 * 6,
    help: '얼마나 앞선 미래의 온도를 예측할 것인지(행 개수)'
  });
  parser.addArgument('--normalize', {
    defaultValue: true,
    help: '정규화된 특성 값을 사용합니다(기본값: true)'
  });
  parser.addArgument('--includeDateTime', {
    action: 'storeTrue',
    help: '날짜와 시간 특성을 사용합니다(기본값: false)'
  });
  parser.addArgument(
      '--batchSize',
      {type: 'int', defaultValue: 128, help: '훈련 배치 크기'});
  parser.addArgument(
      '--epochs',
      {type: 'int', defaultValue: 20, help: '훈련 에포크 횟수'});
  parser.addArgument( '--earlyStoppingPatience', {
    type: 'int',
    defaultValue: 2,
    help: 'earlyStopping 콜백에 사용할 patience 값'
   });
  parser.addArgument('--logDir', {
    type: 'string',
    help: '훈련하는 동안 손실과 정확도를 기록할 텐서보드 로그 디렉토리'
  });
  parser.addArgument('--logUpdateFreq', {
    type: 'string',
    defaultValue: 'batch',
    optionStrings: ['batch', 'epoch'],
    help: '손실과 정확도를 텐서보드에 기록할 빈도'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArguments();
  let tfn;
  if (args.gpu) {
    console.log('훈련에 GPU를 사용합니다.');
    tfn = require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('훈련에 CPU를 사용합니다.');
    tfn = require('@tensorflow/tfjs-node');
  }

  const jenaWeatherData = new JenaWeatherData();
  console.log(`예나 날씨 데이터를 로딩합니다...`);
  await jenaWeatherData.load();

  if (args.modelType === 'baseline') {
    console.log('상식 수준의 MAE를 계산 중입니다....');
    const baselineError = await getBaselineMeanAbsoluteError(
        jenaWeatherData, args.normalize, args.includeDateTime, args.lookBack,
        args.step, args.delay);
    console.log(
        `상식 수준의 MAE: ` +
        `${baselineError.toFixed(6)}`);
  } else {
    let numFeatures = jenaWeatherData.getDataColumnNames().length;
    const model = buildModel(
        args.modelType, Math.floor(args.lookBack / args.step), numFeatures);

    let callback = [];
    if (args.logDir != null) {
      console.log(
          `텐서보드에 로깅합니다. ` +
          `다음 명령으로 텐서보드 서버를 실행하세요:\n` +
          `  tensorboard --logdir ${args.logDir}`);
      callback.push(tfn.node.tensorBoard(args.logDir, {
        updateFreq: args.logUpdateFreq
      }));
    }
    if (args.earlyStoppingPatience != null) {
      console.log(
          `earlyStopping 콜백의 patience: ` +
          `${args.earlyStoppingPatience}.`);
      callback.push(tfn.callbacks.earlyStopping({
        patience: args.earlyStoppingPatience
      }));
    }

    await trainModel(
        model, jenaWeatherData, args.normalize, args.includeDateTime,
        args.lookBack, args.step, args.delay, args.batchSize, args.epochs,
        callback);
  }
}

if (require.main === module) {
  main();
}
