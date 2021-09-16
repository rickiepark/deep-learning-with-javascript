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

const BASE_URL =
    'https://gist.githubusercontent.com/ManrajGrover/6589d3fd3eb9a0719d2a83128741dfc1/raw/d0a86602a87bfe147c240e87e6a9641786cafc19/';

/**
 *
 * @param {Array<Object>} data 다운로드된 데이터
 *
 * @returns {Promise.Array<number[]>} 실수로 파싱된 데이터를 반환합니다.
 */
async function parseCsv (data) {
  return new Promise(resolve => {
    data = data.map((row) => {
      return Object.keys(row).sort().map(key => parseFloat(row[key]));
    });
    resolve(data);
  });
};

/**
 * csv 파일을 다운로드하여 반환합니다.
 *
 * @param {string} filename 로딩할 파일 이름
 *
 * @returns {Promise.Array<number[]>} 파싱된 csv 데이터를 반환합니다
 */
export async function loadCsv(filename) {
  return new Promise(resolve => {
    const url = `${BASE_URL}${filename}.csv`;

    console.log(`  * 다음 주소에서 데이터 다운로드 중: ${url}`);
    Papa.parse(url, {
      download: true,
      header: true,
      complete: (results) => {
        resolve(parseCsv(results['data']));
      }
    })
  });
};

/**
 * Fisher-Yates 알고리즘을 사용하여 데이터와 타깃을 섞습니다.
 */
export async function shuffle(data, label) {
  let counter = data.length;
  let temp = 0;
  let index = 0;
  while (counter > 0) {
    index = (Math.random() * counter) | 0;
    counter--;
    // 데이터
    temp = data[counter];
    data[counter] = data[index];
    data[index] = temp;
    // 레이블
    temp = label[counter];
    label[counter] = label[index];
    label[index] = temp;
  }
};

/**
 * 벡터의 산술 평균을 계산합니다.
 *
 * @param {Array} vector 숫자 배열로 표현된 벡터
 *
 * @returns {number} 산술 평균
 */
function mean(vector) {
  let sum = 0;
  for (const x of vector) {
    sum += x;
  }
  return sum / vector.length;
};

/**
 * 벡터의 표준 편차를 계산합니다.
 *
 * @param {Array} vector 숫자 배열로 표현된 벡터
 *
 * @returns {number} 표준 편차
 */
function stddev(vector) {
  let squareSum = 0;
  const vectorMean = mean(vector);
  for (const x of vector) {
    squareSum += (x - vectorMean) * (x - vectorMean);
  }
  return Math.sqrt(squareSum / (vector.length - 1));
};

/**
 * 벡터를 평균과 표준 편차를 사용해 정규화합니다.
 *
 * @param {Array} vector 정규화할 벡터
 * @param {number} vectorMean 평균
 * @param {number} vectorStddev 표준 편차
 *
 * @returns {Array} 정규화된 벡터
 */
const normalizeVector = (vector, vectorMean, vectorStddev) => {
  return vector.map(x => (x - vectorMean) / vectorStddev);
};

/**
 * 데이터셋을 정규화합니다.
 *
 * @param {Array} dataset 정규화할 데이텃pㅅ
 * @param {boolean} isTrainData 훈련 데이터인지 여부
 * @param {Array} vectorMeans 데이터셋에 있는 각 열의 평균
 * @param {Array} vectorStddevs 데이터셋에 있는 각 열의 표준 편차
 *
 * @returns {Object} 정규화된 데이터셋, 각 열의 평균과 표준 편차를 담고 있는 객체
 */
export function normalizeDataset(
    dataset, isTrainData = true, vectorMeans = [], vectorStddevs = []) {
      const numFeatures = dataset[0].length;
      let vectorMean;
      let vectorStddev;

      for (let i = 0; i < numFeatures; i++) {
        const vector = dataset.map(row => row[i]);

        if (isTrainData) {
          vectorMean = mean(vector);
          vectorStddev = stddev(vector);

          vectorMeans.push(vectorMean);
          vectorStddevs.push(vectorStddev);
        } else {
          vectorMean = vectorMeans[i];
          vectorStddev = vectorStddevs[i];
        }

        const vectorNormalized =
            normalizeVector(vector, vectorMean, vectorStddev);

        vectorNormalized.forEach((value, index) => {
          dataset[index][i] = value;
        });
      }

      return {dataset, vectorMeans, vectorStddevs};
    };

/**
 * 임곗값 0.5를 기준으로 텐서를 이진 값으로 바꿉니다.
 *
 * @param {tf.Tensor} y 변환할 텐서
 * @param {number} threshold 임곗값 (기본값: 0.5)
 * @returns {tf.Tensor} 이진 값으로 바뀐 텐서
 */
export function binarize(y, threshold) {
  if (threshold == null) {
    threshold = 0.5;
  }
  tf.util.assert(
      threshold >= 0 && threshold <= 1,
      `임곗값은 0보다 크거가 같고 1보다 작거나 같아야 합니다. 현재 값은 ${threshold}입니다.`);

  return tf.tidy(() => {
    const condition = y.greater(tf.scalar(threshold));
    return tf.where(condition, tf.onesLike(y), tf.zerosLike(y));
  });
}
