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

// MNIST 이미지를 나타내는 이미지 벡터(길이 784)를 사람이 읽을 수 있는 텍스트 표현으로 바꿉니다.
//
// Args:
//   imageVector: `imageSize * imageSize` 길이의 숫자 배열
//
// Returns:
//   이미지를 나타내는 문자열
export function imageVectorToText(imageVector, imageSize) {
  if (imageVector.length !== imageSize * imageSize) {
    throw new Error(
        '이미지 벡터 길이가 잘못되었습니다(예상 길이: ' + imageSize * imageSize +
        '; 현재 길이: ' + imageVector.length + ')');
  }
  let text = '';
  for (let i = 0; i < imageSize * imageSize; ++i) {
    if (i % imageSize === 0 && i > 0) {
      text += '\n';
    }
    const numString = imageVector[i].toString();
    text +=
        ' '.repeat(numString.length < 4 ? 4 - numString.length : 0) + numString;
  }
  return text;
}

// MNIST 이미지의 텍스트 표현을 [1, imageSize, imageSize, 1] 크기의 텐서로 바꿉니다.
//
// Args:
//   text: MNIST 이미지를 나타내는 문자열
//
// Returns:
//   배치 크기가 1인 이미지를 나타내는 텐서 객체
//     크기: [1, imageSize, imageSize, 1].
export function textToImageArray(text, imageSize) {
  // 행으로 나눕니다.
  const pixels = [];
  const rows = text.split('\n');
  for (const row of rows) {
    const tokens = row.split(' ');
    for (const token of tokens) {
      if (token.length > 0) {
        pixels.push(Number.parseInt(token) / 255);
      }
    }
  }
  if (pixels.length !== imageSize * imageSize) {
    throw new Error(
        '이미지 벡터 길이가 잘못되었습니다(예상 길이: ' + imageSize * imageSize +
        '; 현재 길이: ' + pixels.length + ')');
  }
  return tf.tensor4d(pixels, [1, imageSize, imageSize, 1]);
}

export function indexToOneHot(index, numClasses) {
  const oneHot = [];
  for (let i = 0; i < numClasses; ++i) {
    oneHot.push(i === index ? 1 : 0);
  }
  return oneHot;
}

export function convertDataToTensors(data, numClasses) {
  const numExamples = data.length;
  const imgRows = data[0].x.length;
  const imgCols = data[0].x[0].length;
  const xs = [];
  const ys = [];
  data.map(example => {
    xs.push(example.x);
    ys.push(this.indexToOneHot(example.y, numClasses));
  });
  let xsTensor = tf.reshape(
      tf.tensor3d(xs, [numExamples, imgRows, imgCols]),
      [numExamples, imgRows, imgCols, 1]);
  xsTensor = tf.mul(tf.scalar(1 / 255), xsTensor);
  const ysTensor = tf.tensor2d(ys, [numExamples, numClasses]);
  return {x: xsTensor, y: ysTensor};
}
