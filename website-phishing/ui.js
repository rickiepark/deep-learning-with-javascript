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

const statusElement = document.getElementById('status');
export function updateStatus(message) {
  statusElement.innerText = message;
};

export async function plotLosses(trainLogs) {
  return tfvis.show.history(
      document.getElementById('plotLoss'), trainLogs, ['loss', 'val_loss'], {
        width: 450,
        height: 320,
        xLabel: '에포크',
        yLabel: '손실',
      });
}

export async function plotAccuracies(trainLogs) {
  tfvis.show.history(
      document.getElementById('plotAccuracy'), trainLogs, ['acc', 'val_acc'], {
        width: 450,
        height: 320,
        xLabel: '에포크',
        yLabel: '정확도',
      });
}

const rocValues = [];
const rocSeries = [];

/**
 * ROC 곡선을 그립니다.
 * @param {number[]} fprs 거짓 양성 비율
 * @param {number[]} tprs 진짜 양성 비율. `fprs`와 길이가 같음.
 * @param {number} epoch 에포크 횟수
 */
export async function plotROC(fprs, tprs, epoch) {
  epoch++;  // 에로크를 1부터 시작하도록 바꿉니다.

  // 시리즈의 리스트에 시리즈 이름을 저장합니다.
  const seriesName = '에포크 ' +
      (epoch < 10 ? `00${epoch}` : (epoch < 100 ? `0${epoch}` : `${epoch}`))
  rocSeries.push(seriesName);

  const newSeries = [];
  for (let i = 0; i < fprs.length; i++) {
    newSeries.push({
      x: fprs[i],
      y: tprs[i],
    });
  }
  rocValues.push(newSeries);

  return tfvis.render.linechart(
      document.getElementById('rocCurve'),
      {values: rocValues, series: rocSeries},
      {
        width: 450,
        height: 320,
        xLabel: 'FPR',
        yLabel: 'TPR',
      },
  );
}
