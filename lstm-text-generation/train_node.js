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
 * 다음 문자 예측 모델 훈련하기
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as https from 'https';

import * as argparse from 'argparse';

import {TextData, TEXT_DATA_URLS} from './data';
import {createModel, compileModel, fitModel, generateText} from './model';

/**
 * 필요하면 파일 다운로드하기
 *
 * @param {string} sourceURL 파일을 다운로드할 URL
 * @param {string} destPath 로컬에 저장할 파일 시스템의 경로
 */
export async function maybeDownload(sourceURL, destPath) {
  const fs = require('fs');
  return new Promise(async (resolve, reject) => {
    if (!fs.existsSync(destPath) || fs.lstatSync(destPath).size === 0) {
      const localZipFile = fs.createWriteStream(destPath);
      console.log(`${sourceURL}에서 ${destPath}로 파일을 다운로드하는 중...`);
      https.get(sourceURL, response => {
        response.pipe(localZipFile);
        localZipFile.on('finish', () => {
          localZipFile.close(() => resolve());
        });
        localZipFile.on('error', err => reject(err));
      });
    } else {
      return resolve();
    }
  });
}

function parseArgs() {
  const parser = argparse.ArgumentParser({
    description: 'lstm-text-generation 모델을 훈련합니다.'
  });
  parser.addArgument('textDatasetName', {
    type: 'string',
    choices: Object.keys(TEXT_DATA_URLS),
    help: '텍스트 데이터셋 이름'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'CUDA GPU를 사용해 훈련합니다.'
  });
  parser.addArgument('--sampleLen', {
    type: 'int',
    defaultValue: 60,
    help: '샘플 길이: 모델에 들어갈 입력 시퀀스의 길이(문자 개수)'
  });
  parser.addArgument('--sampleStep', {
    type: 'int',
    defaultValue: 3,
    help: '스텝 길이: 텍스트 데이터에서 추출한 한 샘플과 다음 샘플 사이에 건너 뛸 문자 개수'
  });
  parser.addArgument('--learningRate', {
    type: 'float',
    defaultValue: 1e-2,
    help: '훈련에 사용할 학습률'
  });
  parser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 150,
    help: '훈련 에포크 횟수'
  });
  parser.addArgument('--examplesPerEpoch', {
    type: 'int',
    defaultValue: 10000,
    help: '훈련 에포크에서 사용할 텍스트 샘플 개수'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 128,
    help: '훈련 배치 크기'
  });
  parser.addArgument('--validationSplit', {
    type: 'float',
    defaultValue: 0.0625,
    help: '검증 세트 비율'
  });
  parser.addArgument('--displayLength', {
    type: 'int',
    defaultValue: 120,
    help: '훈련 에포크가 끝날 때마다 출력할 샘플 텍스트 길이'
  });
  parser.addArgument('--savePath', {
    type: 'string',
    help: '모델을 저장할 경로 (옵션)'
  });
  parser.addArgument('--lstmLayerSize', {
    type: 'string',
    defaultValue: '128,128',
    help: 'LSTM 층 크기. 하나의 숫자 또는 콤마로 구분된 숫자 배열 ' +
    '(예를 들어, "256", "256,128")'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArgs();
  if (args.gpu) {
    console.log('GPU 사용');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('CPU 사용');
    require('@tensorflow/tfjs-node');
  }

  // 텍스트 데이터 객체 생성
  const textDataURL = TEXT_DATA_URLS[args.textDatasetName].url;
  const localTextDataPath = path.join(os.tmpdir(), path.basename(textDataURL));
  await maybeDownload(textDataURL, localTextDataPath);
  const text = fs.readFileSync(localTextDataPath, {encoding: 'utf-8'});
  const textData =
      new TextData('text-data', text, args.sampleLen, args.sampleStep);

  // `createModel()`에 전달하기 전에 lstmLayerSize를 문자열에서 숫자 배열로 바꿉니다.
  const lstmLayerSize = args.lstmLayerSize.indexOf(',') === -1 ?
      Number.parseInt(args.lstmLayerSize) :
      args.lstmLayerSize.split(',').map(x => Number.parseInt(x));

  const model = createModel(
      textData.sampleLen(), textData.charSetSize(), lstmLayerSize);
  compileModel(model, args.learningRate);

  // 모델 훈련 도중 테스트 출력을 위한 시드 텍스트
  const [seed, seedIndices] = textData.getRandomSlice();
  console.log(`Seed text:\n"${seed}"\n`);

  const DISPLAY_TEMPERATURES = [0, 0.25, 0.5, 0.75];

  let epochCount = 0;
  await fitModel(
      model, textData, args.epochs, args.examplesPerEpoch, args.batchSize,
      args.validationSplit, {
        onTrainBegin: async () => {
          epochCount++;
          console.log(`에포크: ${epochCount} / ${args.epochs}:`);
        },
        onTrainEnd: async () => {
          DISPLAY_TEMPERATURES.forEach(async temperature => {
            const generated = await generateText(
                model, textData, seedIndices, args.displayLength, temperature);
            console.log(
                `생성된 텍스트 (온도=${temperature}):\n` +
                `"${generated}"\n`);
          });
        }
      });

  if (args.savePath != null && args.savePath.length > 0) {
    await model.save(`file://${args.savePath}`);
    console.log(`모델 저장: ${args.savePath}`);
  }
}

main();
