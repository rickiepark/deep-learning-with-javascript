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
 * Provides methods and classes that support loading data from
 * both MNIST and Fashion MNIST datasets.
 */

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as http from 'http';
import * as https from 'https';
import * as path from 'path';
import * as util from 'util';
import * as zlib from 'zlib';

const exists = util.promisify(fs.exists);
const mkdir = util.promisify(fs.mkdir);
const readFile = util.promisify(fs.readFile);
const rename = util.promisify(fs.rename);

// MNIST와 패션 MNIST 데이터셋 스펙
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const LABEL_HEADER_MAGIC_NUM = 2049;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;

// 파일을 한 번 다운로드하고 파일 버퍼를 반환합니다.
export async function fetchOnceAndSaveToDiskWithBuffer(
    baseURL, destDir, filename) {

  return new Promise(async (resolve, reject) => {
    const url = `${baseURL}${filename}.gz`;
    const localPath = path.join(destDir, filename);
    if (await exists(localPath)) {
      resolve(readFile(localPath));
      return;
    }
    const file = fs.createWriteStream(filename);
    console.log(`  * 다운로드 중: ${url}`);
    let httpModule;
    if (url.indexOf('https://') === 0) {
      httpModule = https;
    } else if (url.indexOf('http://') === 0) {
      httpModule =  http;
    } else {
      return reject(`알 수 없는 URL 프로토콜: ${url}`);
    }

    httpModule.get(url, (response) => {
      const unzip = zlib.createGunzip();
      response.pipe(unzip).pipe(file);
      unzip.on('end', async () => {
        await rename(filename, localPath);
        resolve(readFile(localPath));
      });
    });
  });
}

function loadHeaderValues(buffer, headerLength) {
  const headerValues = [];
  for (let i = 0; i < headerLength / 4; i++) {
    // 헤더 데이터는 빅엔디안으로 저장되어 있습니다.
    headerValues[i] = buffer.readUInt32BE(i * 4);
  }
  return headerValues;
}

async function loadImages(baseURL, destDir, filename) {
  const buffer =
      await fetchOnceAndSaveToDiskWithBuffer(baseURL, destDir, filename);

  const headerBytes = IMAGE_HEADER_BYTES;
  const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  tf.util.assert(
      headerValues[0] === IMAGE_HEADER_MAGIC_NUM,
      () => `이미지 파일 헤더가 매직 넘버와 맞지 않습니다.`);
  tf.util.assert(
      headerValues[2] === IMAGE_HEIGHT,
      () => `파일 헤더에 있는 값(${headerValues[2]})이 ` +
      `이미지 높이 ${IMAGE_HEIGHT}와 맞지 않습니다.`);
  tf.util.assert(
      headerValues[3] === IMAGE_WIDTH,
      () => `파일 헤더에 있는 값(${headerValues[3]})이 ` +
      `이미지 너비 ${IMAGE_WIDTH}와 맞지 않습니다.`);

  const images = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Float32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      // 픽셀 값을 0~255 범위에서 0~1 범위로 정규화합니다.
      array[i] = buffer.readUInt8(index++) / 255;
    }
    images.push(array);
  }

  tf.util.assert(
      images.length === headerValues[1],
      () => `실제 이미지 길이(${images.length}가 ` +
      `헤더에 있는 값(${headerValues[1]})과 맞지 않습니다.`);
  return images;
}

async function loadLabels(baseURL, destDir, filename) {
  const buffer =
      await fetchOnceAndSaveToDiskWithBuffer(baseURL, destDir, filename);

  const headerBytes = LABEL_HEADER_BYTES;
  const recordBytes = LABEL_RECORD_BYTE;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  tf.util.assert(
      headerValues[0] === LABEL_HEADER_MAGIC_NUM,
      () => `레이블 파일 헤더가 매직 넘버와 맞지 않습니다.`);

  const labels = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Int32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      array[i] = buffer.readUInt8(index++);
    }
    labels.push(array);
  }

  tf.util.assert(
      labels.length === headerValues[1],
      () => `실제 레이블 길이(${images.length})가 ` +
      `헤더에 있는 값(${headerValues[1]})과 맞지 않습니다.`);
  return labels;
}

/** 훈련 데이터와 테스트 데이터 로딩을 위한 헬퍼 클래스 */
export class MnistDataset {
  // MNIST 데이터 상수:
  constructor() {
    this.dataset = null;
    this.trainSize = 0;
    this.testSize = 0;
    this.trainBatchIndex = 0;
    this.testBatchIndex = 0;
  }

  getBaseUrlAndFilePaths() {
    return {
      baseUrl: 'https://storage.googleapis.com/cvdf-datasets/mnist/',
      destDir: 'data-mnist',
      trainImages: 'train-images-idx3-ubyte',
      trainLabels: 'train-labels-idx1-ubyte',
      testImages: 't10k-images-idx3-ubyte',
      testLabels: 't10k-labels-idx1-ubyte'
    }
  }

  /** 훈련 데이터와 테스트 데이터를 로드합니다. */
  async loadData() {
    const baseUrlAndFilePaths = this.getBaseUrlAndFilePaths();
    const baseUrl = baseUrlAndFilePaths.baseUrl;
    const destDir = baseUrlAndFilePaths.destDir;
    if (!(await exists(destDir))) {
      await mkdir(destDir);
    }

    this.dataset = await Promise.all([
      loadImages(baseUrl, destDir, baseUrlAndFilePaths.trainImages),
      loadLabels(baseUrl, destDir, baseUrlAndFilePaths.trainLabels),
      loadImages(baseUrl, destDir, baseUrlAndFilePaths.testImages),
      loadLabels(baseUrl, destDir, baseUrlAndFilePaths.testLabels)
    ]);
    this.trainSize = this.dataset[0].length;
    this.testSize = this.dataset[2].length;
  }

  getTrainData() {
    return this.getData_(true);
  }

  getTestData() {
    return this.getData_(false);
  }

  getData_(isTrainingData) {
    let imagesIndex;
    let labelsIndex;
    if (isTrainingData) {
      imagesIndex = 0;
      labelsIndex = 1;
    } else {
      imagesIndex = 2;
      labelsIndex = 3;
    }
    const size = this.dataset[imagesIndex].length;
    tf.util.assert(
        this.dataset[labelsIndex].length === size,
        `이미지 개수(${size})와 ` +
            `레이블 개수(${this.dataset[labelsIndex].length})가 일치하지 않습니다.`);

    // 이미지 배치를 담을 배열을 만듭니다.
    const imagesShape = [size, IMAGE_HEIGHT, IMAGE_WIDTH, 1];
    const images = new Float32Array(tf.util.sizeFromShape(imagesShape));
    const labels = new Int32Array(tf.util.sizeFromShape([size, 1]));

    let imageOffset = 0;
    let labelOffset = 0;
    for (let i = 0; i < size; ++i) {
      images.set(this.dataset[imagesIndex][i], imageOffset);
      labels.set(this.dataset[labelsIndex][i], labelOffset);
      imageOffset += IMAGE_FLAT_SIZE;
      labelOffset += 1;
    }

    return {
      images: tf.tensor4d(images, imagesShape),
      labels: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
    };
  }
}

export class FashionMnistDataset extends MnistDataset {
  getBaseUrlAndFilePaths() {
    return {
      baseUrl: 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
      destDir: 'data-fashion-mnist',
      trainImages: 'train-images-idx3-ubyte',
      trainLabels: 'train-labels-idx1-ubyte',
      testImages: 't10k-images-idx3-ubyte',
      testLabels: 't10k-labels-idx1-ubyte'
    }
  }
}
