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

const HOUSING_CSV_URL = 'https://storage.googleapis.com/learnjs-data/csv-datasets/california_housing_train_10k.csv';

export const featureColumns = [
  'longitude', 'latitude', 'housing_median_age', 'total_rooms',
  'total_bedrooms', 'population', 'households',  'median_income'];
const labelColumn = 'median_house_value';

/**
 * 주택 CSV 데이터셋의 열 통계를 계산합니다.
 *
 * @return 다음 필드를 포함하는 객체:
 *   count {number} 행 개수
 *   featureMeans {number[]} 각 원소는 열의 평균. CSV 데이터셋에 있는 특성 열 순서대로 정렬됨.
 *   featureStddevs {number[]} 각 원소는 열의 표준편차. CSV 데이터셋에 있는 특성 열 순서대로 정렬됨.
 *   labelMean {number} 레이블 열의 평균
 *   labeStddev {number} 레이블 열의 표준편차
 */
export async function getDatasetStats() {
  const featureValues = {};
  featureColumns.forEach(feature => {
    featureValues[feature] = [];
  });
  const labelValues = [];

  const dataset = tf.data.csv(HOUSING_CSV_URL, {
    columnConfigs: {
      [labelColumn]: {
        isLabel: true
      }
    }
  });
  const iterator = await dataset.iterator();
  let count = 0;
  while (true) {
    const item = await iterator.next();
    if (item.done) {
      break;
    }
    featureColumns.forEach(feature => {
      if (item.value.xs[feature] == null) {
        throw new Error(`#{count}번째 샘플에 ${feature} 특성이 누락되어 있습니다.`);
      }
      featureValues[feature].push(item.value.xs[feature]);
    });
    labelValues.push(item.value.ys[labelColumn]);
    count++;
  }

  return tf.tidy(() => {
    const featureMeans = {};
    const featureStddevs = {};
    featureColumns.forEach(feature => {
      const {mean, variance} = tf.moments(featureValues[feature]);
      featureMeans[feature] = mean.arraySync();
      featureStddevs[feature] = tf.sqrt(variance).arraySync();
    });

    const moments = tf.moments(labelValues);
    const labelMean = moments.mean.arraySync();
    const labelStddev = tf.sqrt(moments.variance).arraySync();
    return {
      count,
      featureMeans,
      featureStddevs,
      labelMean,
      labelStddev
    };
  });
}

/**
 * z 점수로 정규화한 특성과 레이블 데이터셋을 반환합니다.
 * 훈련, 검증, 평가를 위해 세 개의 xs-ys 텐서 쌍으로 데이터셋을 분할합니다.
 *
 * @param {number} count CSV 데이터셋의 행 개수
 * @param {{[feature: string]: number}} featureMeans 특성의 평균
 * @param {[feature: string]: number} featureStddevs 특성의 표준편차
 * @param {number} labelMean 레이블의 평균
 * @param {number} labelStddev 레이블의 표준편차
 * @param {number} validationSplit 검증 분할 비율. 0< 그리고 <1
 * @param {number} evaluationSplit 평가 분할 비율. 0< 그리고 <1
 * @returns 다음 키를 포함한 객체:
 *   trainXs {tf.Tensor} 훈련 특성 텐서
 *   trainYs {tf.Tensor} 훈련 레이블 텐서
 *   valXs {tf.Tensor} 검증 특성 텐서
 *   valYs {tf.Tensor} 검증 레이블 텐서
 *   evalXs {tf.Tensor} 평가 특성 텐서
 *   evalYs {tf.Tensor} 평가 레이블 텐서
 */
export async function getNormalizedDatasets(
    count, featureMeans, featureStddevs, labelMean, labelStddev,
    validationSplit, evaluationSplit) {
  tf.util.assert(
      validationSplit > 0 && validationSplit < 1,
      () => `validationSplit는 0보다 크고 1보다 작아야 합니다. ` +
            `입력된 값: ${validationSplit}`);
  tf.util.assert(
      evaluationSplit > 0 && evaluationSplit < 1,
      () => `evaluationSplit는 0보다 크고 1보다 작아야 합니다. ` +
            `입력된 값: ${evaluationSplit}`);
  tf.util.assert(
      validationSplit + evaluationSplit < 1,
      () => `validationSplit와 evaluationSplit의 합이 1보다 작아야 합니다.`);

  const dataset = tf.data.csv(HOUSING_CSV_URL, {
    columnConfigs: {
      [labelColumn]: {
        isLabel: true
      }
    }
  });

  const featureValues = [];
  const labelValues = [];
  const indices = [];
  const iterator = await dataset.iterator();
  for (let i = 0; i < count; ++i) {
    const {value, done} = await iterator.next();
    if (done) {
      break;
    }
    featureColumns.map(feature => {
      featureValues.push(
          (value.xs[feature] - featureMeans[feature]) /
          featureStddevs[feature]);
    });
    labelValues.push((value.ys[labelColumn] - labelMean) / labelStddev);
    indices.push(i);
  }

  const xs = tf.tensor2d(featureValues, [count, featureColumns.length]);
  const ys = tf.tensor2d(labelValues, [count, 1]);

  // 셔플링 순서를 고정해서 훈련, 검증, 평가 분할을 일정하게 유지하기 위해 랜덤 시드를 지정합니다.
  Math.seedrandom('1337');
  tf.util.shuffle(indices);

  const numTrain = Math.round(count * (1 - validationSplit - evaluationSplit));
  const numVal = Math.round(count * validationSplit);
  const trainXs = xs.gather(indices.slice(0, numTrain));
  const trainYs = ys.gather(indices.slice(0, numTrain));
  const valXs = xs.gather(indices.slice(numTrain, numTrain + numVal));
  const valYs = ys.gather(indices.slice(numTrain, numTrain + numVal));
  const evalXs = xs.gather(indices.slice(numTrain + numVal));
  const evalYs = ys.gather(indices.slice(numTrain + numVal));

  return {trainXs, trainYs, valXs, valYs, evalXs, evalYs};

}
