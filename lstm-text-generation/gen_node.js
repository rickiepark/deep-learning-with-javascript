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
 * 훈련된 다음 문자 에측 모델을 사용해 텍스트를 생성합니다.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as argparse from 'argparse';

import * as tf from '@tensorflow/tfjs';

import {TextData, TEXT_DATA_URLS} from './data';
import {generateText} from './model';

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
  parser.addArgument('modelJSONPath', {
    type: 'string',
    help: '다음 문자 예측 모델이 저장된 디스크 경로 ' +
    '(e.g., ./my-model/model.json)'
  });
  parser.addArgument('--genLength', {
    type: 'int',
    defaultValue: 200,
    help: '생성할 텍스트 길이.'
  });
  parser.addArgument('--temperature', {
    type: 'float',
    defaultValue: 0.5,
    help: '텍스트 생성에 사용할 온도 값 ' +
    '높을수록 랜덤하게 보이는 결과를 생성합니다.'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'CUDA GPU를 사용합니다.'
  });
  parser.addArgument('--sampleStep', {
    type: 'int',
    defaultValue: 3,
    help: '스텝 길이: 텍스트 데이터에서 추출한 한 샘플과 다음 샘플 사이에 건너 뛸 문자 개수.'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArgs();

  if (args.gpu) {
    console.log('GPU 사용');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('CPU 시용');
    require('@tensorflow/tfjs-node');
  }

  // 모델 로드
  const model = await tf.loadLayersModel(`file://${args.modelJSONPath}`);

  const sampleLen = model.inputs[0].shape[1];

  // 텍스트 데이터 객체 만들기
  const textDataURL = TEXT_DATA_URLS[args.textDatasetName].url;
  const localTextDataPath = path.join(os.tmpdir(), path.basename(textDataURL));
  await maybeDownload(textDataURL, localTextDataPath);
  const text = fs.readFileSync(localTextDataPath, {encoding: 'utf-8'});
  const textData = new TextData('text-data', text, sampleLen, args.sampleStep);

  // 텍스트 데이터 객체에서 시드 텍스트 얻기
  const [seed, seedIndices] = textData.getRandomSlice();

  console.log(`시드 텍스트:\n"${seed}"\n`);

  const generated = await generateText(
      model, textData, seedIndices, args.genLength, args.temperature);

  console.log(`생성된 텍스트:\n"${generated}"\n`);
}

main();
