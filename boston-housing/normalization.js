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

/**
 * 데이터 배열에 있는 각 열의 평균과 표준 편차를 계산합니다.
 *
 * @param {Tensor2d} data 각 열의 평균과 표준 편차를 독립적으로 계산하기 위한 데이터셋
  *
 * @returns {Object} 각 열의 평균과 표준 편차를 1d 텐서로 포함하고 있는 객체
 */
export function determineMeanAndStddev(data) {
  const dataMean = data.mean(0);
  const diffFromMean = data.sub(dataMean);
  const squaredDiffFromMean = diffFromMean.square();
  const variance = squaredDiffFromMean.mean(0);
  const dataStd = variance.sqrt();
  return {dataMean, dataStd};
}

/**
 * 평균과 표준 편차가 주어지면 평균을 빼고 표준 편차로 나누어 정규화합니다.
 *
 * @param {Tensor2d} data 정규화할 데이터. 크기: [batch, numFeatures].
 * @param {Tensor1d} dataMean 데이터의 평균. 크기 [numFeatures].
 * @param {Tensor1d} dataStd 데이터의 표준 편차. 크기 [numFeatures]
 *
 * @returns {Tensor2d} data와 동일한 크기의 텐서이지만,
 * 각 열은 평균이 0이고 단위 표준 편차를 가지도록 정규화되어 있습니다.
 */
export function normalizeTensor(data, dataMean, dataStd) {
  return data.sub(dataMean).div(dataStd);
}
