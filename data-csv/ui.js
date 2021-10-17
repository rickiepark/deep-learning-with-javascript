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

const statusElement = document.getElementById('status');
const rowCountOutputElement = document.getElementById('row-count-output');
const columnNamesMessageElement =
    document.getElementById('column-names-message');
const columnNamesOutputContainerElement =
    document.getElementById('column-names-output-container');
const sampleRowMessageElement = document.getElementById('sample-row-message');
const sampleRowOutputContainerElement =
    document.getElementById('sample-row-output-container');
const whichSampleInputElement = document.getElementById('which-sample-input');

/** 정보 테이블 위에 있는 메시지를 업데이트합니다. */
export function updateStatus(message) {
  statusElement.value = message;
};

/** "행 카운트" 출력 열에 메시지를 업데이트합니다. */
export function updateRowCountOutput(message) {
  rowCountOutputElement.textContent = message;
};

/** "열 이름 가져오기" 출력 열에 메시지를 업데이트합니다. */
export function updateColumnNamesMessage(message) {
  columnNamesMessageElement.textContent = message;
};


/**
 * 순서가 있는 HTML 리스트를 만들어 열 이름을 나열합니다.
 * `colNames` 매개변수는 문자열의 리스트입니다.
 */
export function updateColumnNamesOutput(colNames) {
  const container = columnNamesOutputContainerElement;
  container.align = 'left';
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  const olList = document.createElement('ol');
  for (const name of colNames) {
    const item = document.createElement('li');
    item.textContent = name;
    olList.appendChild(item);
  }
  container.appendChild(olList);
};

// "샘플 행 가져오기" 출력 열에 있는 메시지를 업데이트합니다.
export function updateSampleRowMessage(message) {
  sampleRowMessageElement.textContent = message;
};

/**
 * div 원소를 사용해 HTML 테이블을 만들어 `rawRow` 입력에 표현된 카-값 쌍을 출력합니다.
 * HTML 테이블이 "샘플 행 가져오기" 출력 열에 추가됩니다.
 * Creates an HTML table, using div elements, to display the key-value pairs
 *  represented in the input `rawRow`. This HTML table is inserted into the
 * sample row.
 */
export function updateSampleRowOutput(rawRow) {
  sampleRowOutputContainerElement.textContent = '';
  const row = rawRow;
  for (const key in row) {
    if (row.hasOwnProperty(key)) {
      const oneKeyRow = document.createElement('div');
      oneKeyRow.className = 'divTableRow';
      oneKeyRow.align = 'left';
      const keyDiv = document.createElement('div');
      const valueDiv = document.createElement('div');
      keyDiv.className = 'divTableCellKey';
      valueDiv.className = 'divTableCell';
      keyDiv.textContent = key + ': ';
      valueDiv.textContent = row[key];
      oneKeyRow.appendChild(keyDiv);
      oneKeyRow.appendChild(valueDiv);
      // updateSampleRowOutput에 div를 추가합니다.
      sampleRowOutputContainerElement.appendChild(oneKeyRow);
    }
  }
};

// 선택한 샘플 인덱스의 현재 값을 숫자로 반환합니다.
export function getSampleIndex() {
  return whichSampleInputElement.valueAsNumber;
}

// CSV 파일을 가져오기 위해 현재 지정한 URL을 반환합니다.
export const getQueryElement = () => {
  return document.getElementById('query-url');
}
