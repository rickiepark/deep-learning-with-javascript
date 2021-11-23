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

const tf = require('@tensorflow/tfjs');
const assert = require('assert');
const fs = require('fs');
const util = require('util');
const jimp = require('jimp');
const terminalImage = require('terminal-image');

const readFile = util.promisify(fs.readFile);

const DATASET_PATH = './dataset';
const TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte';
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_CHANNELS = 1;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS;


/**
 * 데이터셋 파일의 헤더를 읽습니다
 *
 * @param {Buffer} buffer
 * @param {number} headerLength
 *
 * @returns {number[]} MNIST 데이터 헤더
 */
function loadHeaderValues(buffer, headerLength) {
  const headerValues = [];
  for (let i = 0; i < headerLength / 4; i++) {
    // 헤더 데이터는 빅엔디안으로 저장되어 있습니다
    headerValues[i] = buffer.readUInt32BE(i * 4);
  }
  return headerValues;
}

/**
 * 파일에서 이미지를 로드하고 0-1 범위로 정규화합니다.
 *
 * 입력 파일은 MNIST/FashionMNIST 파일 포맷이어야 합니다.
 *
 * @param {string} filepath
 *
 * @returns {Float32Array[]} 형식화 배열로 표현된 이미지 배열
 */
async function loadImages(filepath) {
  if (!fs.existsSync(filepath)) {
    console.log(`데이터 파일: ${filepath}이 없습니다.
      README에 있는 다운로드 방법을 참고하세요.`);
    process.exit(1);
  }

  const buffer = await readFile(filepath)

  const headerBytes = IMAGE_HEADER_BYTES;
  const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  assert.equal(headerValues[0], IMAGE_HEADER_MAGIC_NUM);
  assert.equal(headerValues[2], IMAGE_HEIGHT);
  assert.equal(headerValues[3], IMAGE_WIDTH);

  const images = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Float32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      // 0-255 픽셀 값을 0-1 범위로 정규화합니다.
      array[i] = buffer.readUInt8(index++) / 255;
    }
    images.push(array);
  }

  assert.equal(images.length, headerValues[1]);
  tf.util.shuffle(images);
  return images;
}

/**
 * (형식화 배열로 표현된) 이미지 배열을 받아 텐서를 반환합니다.
 *
 * @param {Float32Array[]} imagesData
 *
 * @returns {Tensor3d} 입력 이미지의 텐서
 */
function batchImages(imagesData) {
  const numImages = imagesData.length;
  const flat = [];
  for (let i = 0; i < numImages; i++) {
    const image = imagesData[i];
    for (let j = 0; j < image.length; j++) {
      flat.push(image[j]);
    }
  }

  const batchedTensor =
      tf.tensor3d(flat, [numImages, IMAGE_WIDTH, IMAGE_HEIGHT], 'float32');

  return batchedTensor;
}

/**
 * 형식화 배열로 표현된 이미지를 JIMP 객체로 바꿉니다.
 *
 * @param {Float32Array} imageData
 *
 * @returns {Promise[Jimp]} 이미지를 표현한 Jimp 객체
 */
async function arrayToJimp(imageData) {
  const bufferLen = IMAGE_HEIGHT * IMAGE_WIDTH * 4;
  const buffer = new Uint8Array(bufferLen);

  let index = 0;
  for (let i = 0; i < IMAGE_HEIGHT; ++i) {
    for (let j = 0; j < IMAGE_WIDTH; ++j) {
      const inIndex = (i * IMAGE_WIDTH + j);
      const val = imageData[inIndex] * 255;
      buffer.set([Math.floor(val)], index++);
      buffer.set([Math.floor(val)], index++);
      buffer.set([Math.floor(val)], index++);
      buffer.set([255], index++);
    }
  }

  return new Promise((resolve, reject) => {
    new jimp(
        {data: buffer, width: IMAGE_WIDTH, height: IMAGE_HEIGHT},
        (err, img) => {
          if (err) {
            reject(err);
          } else {
            resolve(img);
          }
        });
  });
}

/**
 * 콘솔에 이미지를 출력합니다.
 *
 * @param {Float32Array} imageData
 */
async function previewImage(imageData) {
  const imageAsJimp = await arrayToJimp(imageData);
  const pngBuffer = await imageAsJimp.getBufferAsync(jimp.MIME_PNG);
  console.log(await terminalImage.buffer(pngBuffer));
}

module.exports = {
  DATASET_PATH,
  TRAIN_IMAGES_FILE,
  IMAGE_FLAT_SIZE,
  loadImages,
  previewImage,
  batchImages,
};
