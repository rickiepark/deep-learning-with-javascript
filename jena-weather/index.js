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
 * 날씨 예측 예제
 *
 * - tfjs-vis를 사용해 데이터를 시각화합니다.
 * - 간단한 모델(선형 회귀와 MLP)을 훈련하고 훈련 과정을 시각화합니다.
 */

import {JenaWeatherData} from './data.js';
import {buildModel, trainModel} from './models.js';
import {currBeginIndex, getDataVizOptions, logStatus, populateSelects, TIME_SPAN_RANGE_MAP, TIME_SPAN_STRIDE_MAP, updateDateTimeRangeSpan, updateScatterCheckbox} from './ui.js';

const dataChartContainer = document.getElementById('data-chart');
const trainModelButton = document.getElementById('train-model');
const modelTypeSelect = document.getElementById('model-type');
const includeDateTimeSelect =
    document.getElementById('include-date-time-features');
const epochsInput = document.getElementById('epochs');

let jenaWeatherData;

/**
 * 차트를 그립니다.
 *
 * 시각화는 다음 요소로 구성됩니다:
 *
 * - 시계열 데이터를 위한 드롭다운 메뉴
 * - "데이터 시리즈를 축으로 사용하기" 체크박스
 * - "데이터 정규화" 체크박스
 *
 * UI 콘트롤 상태에 따라 그려지는 차트는 다음과 같습니다.
 *
 * - 한 개 또는 두 개의 시계열 데이터에 대한 선 그래프
 * - 두 개의 시계열 데이터에 대한 산점도
 */
export function plotData() {
  logStatus('그래프를 렌더링합니다...');
  const {timeSpan, series1, series2, normalize, scatter} = getDataVizOptions();

  if (scatter && series1 !== 'None' && series2 !== 'None') {
    // 두 시계열 데이터에 대한 산점도를 그립니다.
    makeTimeSeriesScatterPlot(series1, series2, timeSpan, normalize);
  } else {
    // 한 개 또는 두 개의 시리즈를 그래프로 그립니다.
    makeTimeSeriesChart(
        series1, series2, timeSpan, normalize, dataChartContainer);
  }

  updateDateTimeRangeSpan(jenaWeatherData);
  updateScatterCheckbox();
  logStatus('차트 렌더링 완료.');
}

/**
 * 한 개 또는 두 개의 시리즈를 선 그래프로 그립니다.
 *
 * @param {string} series1 시계열 1의 이름 (x-축).
 * @param {string} series2 시계열 2의 이름 (y-축).
 * @param {string} timeSpan 기간. `TIME_SPAN_STRIDE_MAP`에 속한 것이어야 합니다.
 * @param {boolean} normalize 두 시계열 데이터를 정규화할지 여부.
 * @param {HTMLDivElement} chartConatiner 차트를 렌더링할 div 요소.
 */
function makeTimeSeriesChart(
    series1, series2, timeSpan, normalize, chartConatiner) {
  const values = [];
  const series = [];
  const includeTime = true;
  if (series1 !== 'None') {
    values.push(jenaWeatherData.getColumnData(
        series1, includeTime, normalize, currBeginIndex,
        TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]));
    series.push(normalize ? `${series1} (정규화)` : series1);
  }
  if (series2 !== 'None') {
    values.push(jenaWeatherData.getColumnData(
        series2, includeTime, normalize, currBeginIndex,
        TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]));
    series.push(normalize ? `${series2} (정규화)` : series2);
  }
  tfvis.render.linechart(chartConatiner, {values, series: series}, {
    width: chartConatiner.offsetWidth * 0.95,
    height: chartConatiner.offsetWidth * 0.3,
    xLabel: '시간',
    yLabel: series.length === 1 ? series[0] : '',
  });
}

/**
 * 두 시계열 데이터의 산점도를 그립니다.
 *
 * @param {string} series1 시계열 1의 이름 (x-축).
 * @param {string} series2 시계열 2의 이름 (y-축).
 * @param {string} timeSpan 기간. `TIME_SPAN_STRIDE_MAP`에 속한 것이어야 합니다.
 * @param {boolean} normalize 두 시계열 데이터를 정규화할지 여부.
 */
function makeTimeSeriesScatterPlot(series1, series2, timeSpan, normalize) {
  const includeTime = false;
  const xs = jenaWeatherData.getColumnData(
      series1, includeTime, normalize, currBeginIndex,
      TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]);
  const ys = jenaWeatherData.getColumnData(
      series2, includeTime, normalize, currBeginIndex,
      TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]);
  const values = [xs.map((x, i) => {
    return {x, y: ys[i]};
  })];
  let seriesLabel1 = series1;
  let seriesLabel2 = series2;
  if (normalize) {
    seriesLabel1 += ' (정규화)';
    seriesLabel2 += ' (정규화)';
  }
  const series = [`${seriesLabel1} - ${seriesLabel2}`];

  tfvis.render.scatterplot(dataChartContainer, {values, series}, {
    width: dataChartContainer.offsetWidth * 0.7,
    height: dataChartContainer.offsetWidth * 0.5,
    xLabel: seriesLabel1,
    yLabel: seriesLabel2
  });
}

trainModelButton.addEventListener('click', async () => {
  logStatus('모델을 훈련합니다...');
  trainModelButton.disabled = true;
  trainModelButton.textContent = '모델을 훈련합니다. 잠시 기다려 주세요...'

  const lookBack = 10 * 24 * 6;  // 10일 이전 데이터를 사용합니다.
  const step = 6;                // 1-시간 스텝.
  const delay = 24 * 6;          // 1일 후 날씨를 예측합니다.
  const batchSize = 128;
  const normalize = true;
  const includeDateTime = includeDateTimeSelect.checked;
  const modelType = modelTypeSelect.value;

  console.log('모델을 만듭니다...');
  let numFeatures = jenaWeatherData.getDataColumnNames().length;
  const model = buildModel(modelType, Math.floor(lookBack / step), numFeatures);

  // tfjs-vis 바이저로 모델의 요약 정보를 나타냅니다.
  const surface =
      tfvis.visor().surface({tab: modelType, name: '모델 요약'});
  tfvis.show.modelSummary(surface, model);

  const trainingSurface =
      tfvis.visor().surface({tab: modelType, name: '모델 훈련'});

  console.log('모델 훈련을 시작합니다...');
  const epochs = +epochsInput.value;
  await trainModel(
      model, jenaWeatherData, normalize, includeDateTime,
      lookBack, step, delay, batchSize, epochs,
      tfvis.show.fitCallbacks(trainingSurface, ['loss', 'val_loss'], {
        callbacks: ['onBatchEnd', 'onEpochEnd'],
        yLabel: 'Loss'
      }));

  logStatus('모델 훈련을 완료합니다...');

  if (modelType.indexOf('mlp') === 0) {
    visualizeModelLayers(
        modelType, [model.layers[1], model.layers[2]],
        ['Dense Layer 1', 'Dense Layer 2']);
  } else if (modelType.indexOf('linear-regression') === 0) {
    visualizeModelLayers(modelType, [model.layers[1]], ['Dense Layer 1']);
  }

  trainModelButton.textContent = '모델 훈련';
  trainModelButton.disabled = false;
});

/**
 * 모델 층을 시각화합니다.
 *
 * @param {string} tab 시각화를 출력할 tfjs-vis 바이저 탭 이름.
 * @param {tf.layers.Layer[]} layers 시각화할 층 배열.
 * @param {string[]} layerNames tfvis 서피스에 레이블로 사용할 층 이름. `layers`와 길이가 같아야 합니다.
 */
function visualizeModelLayers(tab, layers, layerNames) {
  layers.forEach((layer, i) => {
    const surface = tfvis.visor().surface({tab, name: layerNames[i]});
    tfvis.show.layer(surface, layer);
  });
}

async function run() {
  logStatus('예나 날씨 데이터를 로드합니다 (41.2 MB)...');
  jenaWeatherData = new JenaWeatherData();
  await jenaWeatherData.load();
  logStatus('예나 날씨 데이터를 로드했습니다.');
  console.log(
      'T (degC) 열의 표준 편차: ' +
      jenaWeatherData.getMeanAndStddev('T (degC)').stddev.toFixed(4));

  console.log('데이터 시리즈 드롭다운 박스를 생성합니다...');
  populateSelects(jenaWeatherData);

  console.log('데이터를 출력합니다...');
  plotData();
}

run();
