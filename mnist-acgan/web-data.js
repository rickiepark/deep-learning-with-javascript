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

export const IMAGE_H = 28;
export const IMAGE_W = 28;
const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
    'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * 스프라이트된 MNIST 데이터셋을 가져와 tf.Tensor로 변환하는 클래스
 */
export class MnistData {
  constructor() {}

  async load() {
    // MNIST 스프라이트 이미지를 요청합니다.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer =
            new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
              datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
              IMAGE_SIZE * chunkSize);
          ctx.drawImage(
              img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
              chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // 흑백 이미지이기 때문에 모든 채널이 동일한 값을 가집니다. 여기서는 빨강 채널을 사용합니다.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] =
        await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // 이미지와 레이블을 잘라서 훈련 세트와 테스트 세트로 만듭니다.
    this.trainImages =
        this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels =
        this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels =
        this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  /**
   * 모든 훈련 데이터를 데이터 텐서와 레이블 텐서로 준비합니다.
   *
   * @returns
   *   xs: `[numTrainExamples, 28, 28, 1]` 크기의 데이터 텐서.
   *   labels: `[numTrainExamples, 10]` 크기의 원-핫 인코딩된 레이블 텐서.
   */
  getTrainData() {
    const xs = tf.tensor4d(
        this.trainImages,
        [this.trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
    const labels = tf.tensor2d(
        this.trainLabels, [this.trainLabels.length / NUM_CLASSES, NUM_CLASSES]);
    return {xs, labels};
  }

  /**
   * 모든 테스트 데이터를 데이터 텐서와 레이블 텐서로 준비합니다.
   *
   * @param {number} numExamples 가져올 샘플 개수. 지정하지 않으면 모든 테스트 샘플을 가져옵니다.
   * @returns
   *   xs: `[numTestExamples, 28, 28, 1]` 크기의 데이터 텐서.
   *   labels: `[numTestExamples, 10]` 크기의 원-핫 인코딩된 레이블 텐서.
   */
  getTestData(numExamples) {
    let xs = tf.tensor4d(
        this.testImages,
        [this.testImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
    let labels = tf.tensor2d(
        this.testLabels, [this.testLabels.length / NUM_CLASSES, NUM_CLASSES]);

    if (numExamples != null) {
      xs = xs.slice([0, 0, 0, 0], [numExamples, IMAGE_H, IMAGE_W, 1]);
      labels = labels.slice([0, 0], [numExamples, NUM_CLASSES]);
    }
    return {xs, labels};
  }
}

let mnistImages;
let mnistLabels;
let mnistNumExamples;
let mnistIndices;

/** MNIST 데이터 로드하기 */
export async function loadMnistData() {
  const mnistData = new MnistData();
  await mnistData.load();
  const mnistSamples = mnistData.getTrainData();
  mnistImages = mnistSamples.xs;
  mnistLabels = await mnistSamples.labels.argMax(-1).data();

  mnistNumExamples = mnistLabels.length;
  mnistIndices = [];
  for (let i = 0; i < mnistNumExamples; ++i) {
    mnistIndices.push(i);
  }
}

/**
 * MNIST 데이터셋의 클래스마다 여러 개의 샘플을 뽑습니다.
 *
 * @param {number} numExamplesPerClass 클래스별 샘플 개수
 * @returns {tf.Tensor} [numExamplesPerClass * 10, 28, 28, 1] 크기의 4D 텐서
 */
export function sampleFromMnistData(numExamplesPerClass) {
  tf.util.assert(
      numExamplesPerClass <= mnistNumExamples / 10,
      `클래스 당 요청한 샘플이 너무 많습니다: ` +
          `(${numExamplesPerClass} > ${mnistNumExamples / 10})`);

  tf.util.shuffle(mnistIndices);
  const indicesByClass = [];
  for (let i = 0; i < NUM_CLASSES; ++i) {
    indicesByClass.push([]);
  }

  for (let i = 0; i < mnistIndices.length; ++i) {
    if (indicesByClass[mnistLabels[mnistIndices[i]]].length >=
        numExamplesPerClass) {
      continue;
    }
    indicesByClass[mnistLabels[mnistIndices[i]]].push(mnistIndices[i]);

    let minLength = Infinity;
    indicesByClass.forEach(indicesArray => {
      if (indicesArray.length < minLength) {
        minLength = indicesArray.length;
      }
    });
    if (minLength >= numExamplesPerClass) {
      break;
    }
  }

  return tf.tidy(() => {
    let rowsToCombine = [];
    indicesByClass.forEach(classIndices => {
      const classImages = tf.gather(mnistImages, classIndices);
      const rowOfExamples = tf.concat(classImages.unstack(), 0);
      rowsToCombine.push(rowOfExamples);
    });
    return tf.concat(rowsToCombine, 1);
  });
}
