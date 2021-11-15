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

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as https from 'https';
import * as os from 'os';
import * as path from 'path';

import {OOV_INDEX, padSequences} from './sequence_utils';

// extract-zip은 `import` 명령이 안되는 것 같음.
const extract = require('extract-zip');

const DATA_ZIP_URL =
    'https://storage.googleapis.com/learnjs-data/imdb/imdb_tfjs_data.zip';
const METADATA_TEMPLATE_URL =
    'https://storage.googleapis.com/learnjs-data/imdb/metadata.json.zip';

/**
 * 로컬 파일에서 IMDB 데이터 로드하기
 *
 * @param {string} filePath 로컬 파일 시스템에 있는 데이터 파일
 * @param {string} numWords 어휘 사전의 단어 개수. 이를 넘는 단어 인덱스는 `OOV_INDEX`가 됩니다.
 * @param {string} maxLen 각 시퀀스의 길이. 이보다 긴 시퀀스는 앞부분이 잘려지고 짧은 시퀀스는 앞부분에 패딩이 추가됩니다.
 * @param {string} multihot 단어의 멀티-핫 인코딩을 사용할지 여부. 기본값: `false`
 * @return {tf.Tensor} `multihot`이 `false`(기본값)이면,
 *   데이터셋은 `[numExamples, maxLen]` 크기의 2D `int32` `tf.Tensor`로 표현됩니다.
 *   `true`이면 데이터셋은 `[numExamples, numWords]` 크기의 2D `float32` `tf.Tensor`로 표현됩니다.
 */
function loadFeatures(filePath, numWords, maxLen, multihot = false) {
  const buffer = fs.readFileSync(filePath);
  const numBytes = buffer.byteLength;

  let sequences = [];
  let seq = [];
  let index = 0;

  while (index < numBytes) {
    const value = buffer.readInt32LE(index);
    if (value === 1) {
      // 새로운 시퀀스 시작
      if (index > 0) {
        sequences.push(seq);
      }
      seq = [];
    } else {
      // 시퀀스는 계속됩니다.
      seq.push(value >= numWords ? OOV_INDEX : value);
    }
    index += 4;
  }
  if (seq.length > 0) {
    sequences.push(seq);
  }

  // 시퀀스 길이 통계 계산
  let minLength = Infinity;
  let maxLength = -Infinity;
  sequences.forEach(seq => {
    const length = seq.length;
    if (length < minLength) {
      minLength = length;
    }
    if (length > maxLength) {
      maxLength = length;
    }
  });
  console.log(`시퀀스 길이: min = ${minLength}; max = ${maxLength}`);

  if (multihot) {
    // `true`이면 시퀀스를 멀티-핫 벡터로 인코딩합니다.
    const buffer = tf.buffer([sequences.length, numWords]);
    sequences.forEach((seq, i) => {
      seq.forEach(wordIndex => {
        if (wordIndex !== OOV_INDEX) {
          buffer.set(1, i, wordIndex);
        }
      });
    });
    return buffer.toTensor();
  } else {
    const paddedSequences =
        padSequences(sequences, maxLen, 'pre', 'pre');
    return tf.tensor2d(
        paddedSequences, [paddedSequences.length, maxLen], 'int32');
  }
}

/**
 * 파일에서 IMDb 타깃을 로드합니다.
 *
 * @param {string} filePath 이진 타깃 파일의 경로
 * @return {tf.Tensor} `[numExamples, 1]` 크기의 `float32` `tf.Tensor` 타깃. 0 또는 1의 값을 가집니다.
 */
function loadTargets(filePath) {
  const buffer = fs.readFileSync(filePath);
  const numBytes = buffer.byteLength;

  let numPositive = 0;
  let numNegative = 0;

  let ys = [];
  for (let i = 0; i < numBytes; ++i) {
    const y = buffer.readUInt8(i);
    if (y === 1) {
      numPositive++;
    } else {
      numNegative++;
    }
    ys.push(y);
  }

  console.log(
      `${numPositive}개의 긍정적인 샘플과 ` +
      `${numNegative}개의 부정적인 샘플을 로드했습니다.`);
  return tf.tensor2d(ys, [ys.length, 1], 'float32');
}

/**
 * 필요하면 파일을 다운로드합니다.
 *
 * @param {string} sourceURL 파일을 다운로드할 URL
 * @param {string} destPath 파일을 저장할 로컬 파일 시스템의 경로
 */
async function maybeDownload(sourceURL, destPath) {
  return new Promise(async (resolve, reject) => {
    if (!fs.existsSync(destPath) || fs.lstatSync(destPath).size === 0) {
      const localZipFile = fs.createWriteStream(destPath);
      console.log(`파일 다운로드 중: ${sourceURL} ...`);
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

/**
 * 압축을 해제합니다.
 *
 * 만약 이미 파일이 압축해제 되었다면 이 함수는 아무런 일을 하지 않습니다.
 *
 * @param {string} sourcePath zip 파일 경로
 * @param {string} destDir 압축을 풀 디렉토리
 */
async function maybeExtract(sourcePath, destDir) {
  return new Promise((resolve, reject) => {
    if (fs.existsSync(destDir)) {
      return resolve();
    }
    console.log(`압축 해제 중: ${sourcePath} --> ${destDir}`);
    extract(sourcePath, {dir: destDir}, err => {
      if (err == null) {
        return resolve();
      } else {
        return reject(err);
      }
    });
  });
}

const ZIP_SUFFIX = '.zip';

/**
 * 파일을 다운로드하고 압축해제 하여 IMDb 데이터 얻기
 *
 * 파일이 이미 로컬 파일 시스템에 있다면 다운로드와 압축 해제는 실행되지 않습니다.
 */
async function maybeDownloadAndExtract() {
  const zipDownloadDest = path.join(os.tmpdir(), path.basename(DATA_ZIP_URL));
  await maybeDownload(DATA_ZIP_URL, zipDownloadDest);

  const zipExtractDir =
      zipDownloadDest.slice(0, zipDownloadDest.length - ZIP_SUFFIX.length);
  await maybeExtract(zipDownloadDest, zipExtractDir);
  return zipExtractDir;
}

/**
 * 필요하면 파일을 다운로드하고 압축을 해제하여 데이터를 로드합니다.
 *
 * @param {number} numWords 어휘 사전에 있는 단어 개수
 * @param {number} len 각 시퀀스의 길이. 긴 시퀀스는 앞부분이 잘리고, 짧은 시퀀스는 앞부분에 패딩이 추가됩니다.
 * @return
 *   xTrain: 훈련 데이터. `[numExamples, len]` 크기의 `int32` `tf.Tensor`.
 *   yTrain: 타깃 데이터. `[numExamples, 1]` 크기의 `float32` `tf.Tensor`. 0 또는 1입니다.
 *   xTest: `xTrain`과 같지만 테스트 데이터셋입니다.
 *   yTest: `yTrain`과 같지만 테스트 데이터셋입니다.
 */
export async function loadData(numWords, len, multihot = false) {
  const dataDir = await maybeDownloadAndExtract();

  const trainFeaturePath = path.join(dataDir, 'imdb_train_data.bin');
  const xTrain = loadFeatures(trainFeaturePath, numWords, len, multihot);
  const testFeaturePath = path.join(dataDir, 'imdb_test_data.bin');
  const xTest = loadFeatures(testFeaturePath, numWords, len, multihot);
  const trainTargetsPath = path.join(dataDir, 'imdb_train_targets.bin');
  const yTrain = loadTargets(trainTargetsPath);
  const testTargetsPath = path.join(dataDir, 'imdb_test_targets.bin');
  const yTest = loadTargets(testTargetsPath);

  tf.util.assert(
      xTrain.shape[0] === yTrain.shape[0],
      `xTrain과 yTrain의 샘플 개수가 맞지 않습니다.`);
  tf.util.assert(
      xTest.shape[0] === yTest.shape[0],
      `xTest과 yTest의 샘플 개수가 맞지 않습니다.`);
  return {xTrain, yTrain, xTest, yTest};
}

/**
 * 메타데이터 템플릿을 로드합니다. 필요하면 다운로드하고 압축을 해제합니다.
 *
 * @return 메타데이터 템플릿 JSON 객체
 */
export async function loadMetadataTemplate() {
  const baseName = path.basename(METADATA_TEMPLATE_URL);
  const zipDownloadDest = path.join(os.tmpdir(), baseName);
  await maybeDownload(METADATA_TEMPLATE_URL, zipDownloadDest);

  const zipExtractDir =
      zipDownloadDest.slice(0, zipDownloadDest.length - ZIP_SUFFIX.length);
  await maybeExtract(zipDownloadDest, zipExtractDir);

  return JSON.parse(fs.readFileSync(
      path.join(zipExtractDir,
                baseName.slice(0, baseName.length - ZIP_SUFFIX.length))));
}
