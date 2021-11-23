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

const path = require('path');
const fs = require('fs');
const http = require('http');
const zlib = require('zlib');
const mkdirp = require('mkdirp');


/**
 * 필요하면 파일을 다운로드합니다.
 *
 * @param {string} sourceURL 파일을 다운로드할 URL
 * @param {string} destPath 파일을 저장할 로컬 경로
 */
async function maybeDownload(sourceURL, destPath) {
  return new Promise(async (resolve, reject) => {
    if (!fs.existsSync(destPath) || fs.lstatSync(destPath).size === 0) {
      mkdirp(path.dirname(destPath), function(err) {
        if (err) {
          reject(err)
        }
      });
      const localZipFile = fs.createWriteStream(destPath);
      http.get(sourceURL, response => {
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
 * 파일 압축을 풉니다
 *
 * @param {string} sourcePath 압축 파일 경로
 * @param {string} destPath 압축 해제할 경로
 */
async function extract(sourcePath, destPath) {
  return new Promise((resolve, reject) => {
    const fileContents = fs.createReadStream(sourcePath);
    const writeStream = fs.createWriteStream(destPath);
    const unzip = zlib.createGunzip();
    fileContents.pipe(unzip).pipe(writeStream).on('finish', (err) => {
      if (err) {
        reject(err);
      } else {
        resolve();
      }
    });
  });
}



const DATA_URL =
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz';
const ZIP_PATH =
    path.resolve(path.join('./dataset', 'train-images-idx3-ubyte.gz'));
const UNZIP_PATH =
    path.resolve(path.join('./dataset', 'train-images-idx3-ubyte'));



(async function run() {
  try {
    console.log(
        `${DATA_URL}에서 데이터 파일을 다운로드하여 ${ZIP_PATH}에 저장합니다.`);
    await maybeDownload(DATA_URL, ZIP_PATH);
  } catch (e) {
    console.log('파일 다운로드 에러');
    console.log(e);
  }

  try {
    console.log('데이터 파일 압축 해제');
    await extract(ZIP_PATH, UNZIP_PATH);
  } catch (e) {
    console.log('파일 압축 해제 에러');
    console.log(e);
  }
}())
