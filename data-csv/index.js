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

import * as ui from './ui.js';


// 보스턴 주택 CSV
const BOSTON_HOUSING_CSV_URL =
    'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv';
// 예나 기후 CSV
const JENA_CLIMATE_CSV_URL =
    'https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv';
// 드레스 판매 데이터 (https://www.openml.org/d/23381)
const DRESSES_SALES_CSV_URL =
    'https://storage.googleapis.com/learnjs-data/csv-datasets/dresses-sales-openml.csv';
// 뉴욕 주립대 캠퍼스 데이터
// https://data.ny.gov/Education/State-University-of-New-York-SUNY-Campus-Locations/3cij-nwhw
const SUNY_CSV_URL =
    'https://storage.googleapis.com/learnjs-data/csv-datasets/State_University_of_New_York__SUNY__Campus_Locations_with_Websites__Enrollment_and_Select_Program_Offerings.csv';


/**
 * UI에서 입력한 URL을 사용해 CSV 데이터셋 객체를 만듭니다.
 * 그다음 데이터셋에 있는 모든 원소를 순회하면서 카운트를 세고 화면을 업데이트합니다.
 */
async function countRowsHandler() {
  const url = ui.getQueryElement().value;
  ui.updateStatus(`${url}에 연결하기 위해 데이터 객체를 만듭니다.`);
  const myData = tf.data.csv(url);
  let i = 0;
  ui.updateRowCountOutput(`카운팅...`);
  const updateFn = x => {
    i += 1;
    if (i % 1000 === 0) {
      ui.updateStatus(`카운팅 ... 이 CSV 파일에는 지금까지 ${i}개의 행이 있습니다...`);
    }
  };
  try {
    ui.updateStatus('CSV에 있는 행을 카운트합니다.');
    // `tf.data.Dataset.forEachAsync()`는 비동기 함수입니다.
    // `await`이 없으면 `updataFn`을 실행할 때를 제어하지 못합니다.
    // 따라서 최종 카운트 상태를 "0개 행이 카운트되었습니다"로 업데이트한 후에 실행될 것입니다.
    await myData.forEachAsync(x => updateFn(x));
  } catch (e) {
    const errorMsg = `${url}의 데이터를 순회하는데 오류가 발생했습니다.  ` +
        `올바른 URL이 아니거나 CORS 요청을 지원하지 않을 수 있습니다.` +
        `  개발자 콘솔에서 CORS 오류를 확인하세요.` + e;
    ui.updateRowCountOutput(errorMsg);
    return;
  }
  ui.updateStatus(`행 카운팅 완료.`);
  ui.updateRowCountOutput(`이 CSV에는 ${i}개의 행이 있습니다.`);
};

/**
 * 화면에서 지정한 URL을 사용해 CSV Dataset 객체를 만듭니다.
 * 그다음 데이터셋 객체로 연결하여 열 이름을 가져오고 화면을 업데이트합니다.
 */
async function getColumnNamesHandler() {
  ui.updateColumnNamesOutput([]);
  const url = ui.getQueryElement().value;
  ui.updateStatus(`${url}에 있는 CSV 파일에 연결하는 중입니다.`);
  const myData = tf.data.csv(url);
  ui.updateStatus('연결되었습니다 ... 열 이름을 가져오는 중입니다.');
  ui.updateColumnNamesMessage('열 이름을 추출했습니다.');
  try {
    const columnNames = await myData.columnNames();
    ui.updateStatus('열 이름 추출 완료.');
    ui.updateColumnNamesMessage('');
    ui.updateColumnNamesOutput(columnNames);
  } catch (e) {
    const errorMsg = `${url}의 데이터를 순회하는데 오류가 발생했습니다.  ` +
        `올바른 URL이 아니거나 CORS 요청을 지원하지 않을 수 있습니다.` +
        `  개발자 콘솔에서 CORS 오류를 확인하세요.` + e;
    ui.updateColumnNamesMessage(errorMsg);
    return;
  }
};

/**
 * CSV 파일에 접근하여 하나의 행을 가져옵니다.
 * 가져올 행 인덱스는 화면에서 지정합니다.
 */
async function getSampleRowHandler() {
  const url = ui.getQueryElement().value;
  ui.updateStatus(`${url}에 있는 CSV 파일에 연결하는 중입니다.`);
  const myData = tf.data.csv(url);
  ui.updateStatus('연결되었습니다 ... 요청한 샘플을 가져오는 중입니다.');
  const sampleIndex = ui.getSampleIndex();
  if (sampleIndex < 0 || isNaN(sampleIndex)) {
    const msg = `인덱스가 음수이거나 NaN인 샘플은 가져올 수 없습니다. (요청 인덱스: ${
        sampleIndex}).`;
    ui.updateStatus(msg);
    ui.updateSampleRowMessage(msg);
    ui.updateSampleRowOutput([]);
    return;
  }
  let sample;
  try {
    sample = await myData.skip(sampleIndex).take(1).toArray();
  } catch (e) {
    let errorMsg = `${url}에서 샘플을 가져오는데 오류가 발생했습니다.  `;
    errorMsg +=
        '올바른 URL이 아니거나 CORS 요청을 지원하지 않을 수 있습니다.';
    errorMsg += '  개발자 콘솔에서 CORS 오류를 확인하세요.';
    errorMsg += e;
    ui.updateSampleRowMessage(errorMsg);
    return;
  }
  if (sample.length === 0) {
    // CSV 크기를 넘어선 샘플을 요청하면 빈 데이터를 반환합니다.
    const msg = `인덱스가 ${
        sampleIndex}인 샘플을 가져올 수 없습니다.  데이터셋의 크기보다 큽니다.`;
    ui.updateStatus(msg);
    ui.updateSampleRowMessage(msg);
    ui.updateSampleRowOutput([]);
    return;
  }
  ui.updateStatus(`인덱스가 ${sampleIndex}인 샘플 가져오기 완료.`);
  ui.updateSampleRowMessage('');
  ui.updateSampleRowOutput(sample[0]);
};

/** 메시지와 테이블 출력을 초기화합니다. */
const resetOutputMessages = () => {
  ui.updateRowCountOutput('"행 카운트"를 클릭하세요');
  ui.updateColumnNamesMessage('"열 이름 가져오기"를 클릭하세요');
  ui.updateColumnNamesOutput([]);
  ui.updateSampleRowMessage('인덱스를 선택하고 "샘플 행 가져오기"를 클릭하세요');
  ui.updateSampleRowOutput([]);
};

/** 버튼을 포함해 모든 UI 핸들러를 설정합니다. */
document.addEventListener('DOMContentLoaded', async () => {
  resetOutputMessages();

  // 미리 설정한 URL 버튼을 연결하는 헬퍼.
  const connectURLButton = (buttonId, url, statusMessage) => {
    document.getElementById(buttonId).addEventListener('click', async () => {
      ui.getQueryElement().value = url;
      resetOutputMessages();
      ui.updateStatus(statusMessage);
    }, false);
  };

  connectURLButton(
      'jena-climate-button', JENA_CLIMATE_CSV_URL,
      `예나(Jena) 기후 데이터는 일정 기간 수집된 대기 상태에 대한 기록입니다. ` +
          `이 데이터 세트에는 몇 년 동안 10분마다 14개의 서로 다른 양(기온, 기압, 습도, 풍향 등)이 기록되어 있습니다. ` +
          `이 데이터 세트의 모든 행을 세는 데 시간이 다소 걸립니다.`);
  connectURLButton(
      'boston-button', BOSTON_HOUSING_CSV_URL,
      `보스턴 주택 데이터셋은 머신러닝 입문 데이터셋으로 널리 사용됩니다.`);
  connectURLButton(
      'dresses-button', DRESSES_SALES_CSV_URL,
      `이 데이터셋은 드레스 속성과 판매에 기반한 추천을 담고 있습니다. ` +
      `OpenML 제공. 이 데이터셋에 대한 더 자세한 내용은 https://www.openml.org/d/23381을 참고하세요.`);
  connectURLButton(
      'suny-button', SUNY_CSV_URL,
      `뉴욕 주립대 캠퍼. 학부와 대학원, 일부 프로그램 정보를 제공합니다. ` +
      `https://data.ny.gov/에서 더 많은 데이터를 찾을 수 있습니다.`);

  // 버튼 연결하기
  document.getElementById('count-rows')
      .addEventListener('click', countRowsHandler, false);
  document.getElementById('get-column-names')
      .addEventListener('click', getColumnNamesHandler, false);
  document.getElementById('get-sample-row')
      .addEventListener('click', getSampleRowHandler, false);

  // 변경 사항을 가져오기 위해 샘플 인덱스 연결하기
  document.getElementById('which-sample-input')
      .addEventListener('change', getSampleRowHandler, false);
}, false);
