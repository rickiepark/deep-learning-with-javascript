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

// 보스턴 주택 데이터 경로:
const BASE_URL =
    'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/';

const TRAIN_FEATURES_FN = 'train-data.csv';
const TRAIN_TARGET_FN = 'train-target.csv';
const TEST_FEATURES_FN = 'test-data.csv';
const TEST_TARGET_FN = 'test-target.csv';

/**
 * 이 CSV 데이터는 숫자 배열의 배열을 반환합니다.
 *
 * @param {Array<Object>} data 다운로드된 데이터.
 *
 * @returns {Promise.Array<number[]>} 실수 값으로 파싱된 데이터.
 */
const parseCsv = async (data) => {
  return new Promise(resolve => {
    data = data.map((row) => {
      return Object.keys(row).map(key => parseFloat(row[key]));
    });
    resolve(data);
  });
};

/**
 * csv를 다운로드하여 반환합니다.
 *
 * @param {string} filename 다운로드할 파일 이름.
 *
 * @returns {Promise.Array<number[]>} 파싱된 csv 데이터.
 */
export const loadCsv = async (filename) => {
  return new Promise(resolve => {
    const url = `${BASE_URL}${filename}`;

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

/** 훈련 데이터와 테스트 데이터 적재를 위한 헬퍼 클래스 */
export class BostonHousingDataset {
  constructor() {
    // 데이터를 담을 배열
    this.trainFeatures = null;
    this.trainTarget = null;
    this.testFeatures = null;
    this.testTarget = null;
  }

  get numFeatures() {
    // 데이터를 로드하기 전에 numFetures를 참조하면 에러를 발생시킵니다.
    if (this.trainFeatures == null) {
      throw new Error('numFeatures 전에 \'loadData()\'를 호출해야 합니다.')
    }
    return this.trainFeatures[0].length;
  }

  /** 훈련 데이터와 테스트 데이터를 적재합니다. */
  async loadData() {
    [this.trainFeatures, this.trainTarget, this.testFeatures, this.testTarget] =
        await Promise.all([
          loadCsv(TRAIN_FEATURES_FN), loadCsv(TRAIN_TARGET_FN),
          loadCsv(TEST_FEATURES_FN), loadCsv(TEST_TARGET_FN)
        ]);

    shuffle(this.trainFeatures, this.trainTarget);
    shuffle(this.testFeatures, this.testTarget);
  }
}

export const featureDescriptions = [
  '범죄율', '주거용 토지 비율', '비소매업 비율', '강 인접도',
  '일산화질소 농도', '평균 방 개수', '오래된 주택 비율',
  '고용센터까지 거리', '고속도로 접근성', '세율', '학생-교사 비율',
  '고등교육 이하 비율'
];

/**
 * Fisher-Yates 알고리즘을 사용하여 데이터와 타깃의 쌍을 섞습니다.
 */
function shuffle(data, target) {
  let counter = data.length;
  let temp = 0;
  let index = 0;
  while (counter > 0) {
    index = (Math.random() * counter) | 0;
    counter--;
    // 데이터:
    temp = data[counter];
    data[counter] = data[index];
    data[index] = temp;
    // 타깃:
    temp = target[counter];
    target[counter] = target[index];
    target[index] = temp;
  }
};
