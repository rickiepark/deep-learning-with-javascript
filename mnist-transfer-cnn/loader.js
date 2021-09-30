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

import * as ui from './ui.js';
import * as util from './util.js';

/**
 * URL에서 데이터를 가져올 수 있는지 테스트합니다.
 */
export async function urlExists(url) {
  ui.status('Testing url ' + url);
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

/**
 * 원격 URL에 저장된 사전 훈련된 모델을 로드합니다.
 *
 * @return 모델 토폴로지와 가중치를 가진 `tf.Model` 인스턴스
 */
export async function loadHostedPretrainedModel(url) {
  ui.status('다음 주소에서 사전 훈련된 모델을 로드합니다: ' + url);
  try {
    const model = await tf.loadLayersModel(url);
    ui.status('사전 훈련된 모델을 로딩 완료했습니다.');
    // https://github.com/tensorflow/tfjs/issues/34 이슈 때문에
    // 모델을 두 번 로드할 수 없습니다.
    // 따라서 사용자가 혼란을 막기 위해 모델 로딩 버튼을 삭제합니다.
    ui.disableLoadModelButtons();
    return model;
  } catch (err) {
    console.error(err);
    ui.status('사전 훈련된 모델 로딩이 실패했습니다.');
  }
}

/**
 * 원격 URL에 저장된 데이터 파일을 로딩합니다.
 *
 * @return 키-값 쌍으로 메타데이터가 저장된 객체
 */
export async function loadHostedData(url, numClasses) {
  ui.status('다음 주소에서 데이터를 로딩합니다: ' + url);
  try {
    const raw = await fetch(url);
    const data = await raw.json();
    const result = util.convertDataToTensors(data, numClasses);
    result['data'] = data;
    ui.status('데이터 로딩이 완료되었습니다.');
    return result;
  } catch (err) {
    console.error(err);
    ui.status('데이터 로딩이 실패했습니다.');
  }
}
