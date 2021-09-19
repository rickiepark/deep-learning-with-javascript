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

/**
 * 주어진 URL이 유효한지 테스트합니다.
 */
export async function urlExists(url) {
  ui.status('테스트 URL: ' + url);
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
 * @return 로딩한 모델 구조와 가중치를 가진 `tf.Model` 객체
 */
export async function loadHostedPretrainedModel(url) {
  ui.status('다음 주소에서 사전 훈련된 모델을 로딩합니다: ' + url);
  try {
    const model = await tf.loadLayersModel(url);
    ui.status('사전 훈련된 모델을 로딩했습니다.');
    return model;
  } catch (err) {
    console.error(err);
    ui.status('사전 훈련된 모델을 로딩하는데 실패했습니다.');
  }
}

// 다운로드하거나 로컬에서 훈련한 모델을 클라이언트 측에 저장하기 위한 URL 형식의 구분자
const LOCAL_MODEL_URL = 'indexeddb://tfjs-iris-demo-model/v1';

export async function saveModelLocally(model) {
  const saveResult = await model.save(LOCAL_MODEL_URL);
}

export async function loadModelLocally() {
  return await tf.loadLayersModel(LOCAL_MODEL_URL);
}

export async function removeModelLocally() {
  return await tf.io.removeModel(LOCAL_MODEL_URL);
}

/**
 * 로컬에 저장된 모델의 상태를 체크합니다(예를 들어, IndexedDB에서).
 *
 * 상태에 따라 UI를 업데이트합니다.
 */
export async function updateLocalModelStatus() {
  const localModelStatus = document.getElementById('local-model-status');
  const localLoadButton = document.getElementById('load-local');
  const localRemoveButton = document.getElementById('remove-local');

  const modelsInfo = await tf.io.listModels();
  if (LOCAL_MODEL_URL in modelsInfo) {
    localModelStatus.textContent = '저장된 모델이 있습니다: ' +
        modelsInfo[LOCAL_MODEL_URL].dateSaved.toDateString();
    localLoadButton.disabled = false;
    localRemoveButton.disabled = false;
  } else {
    localModelStatus.textContent = '로컬에 저장된 모델이 없습니다.';
    localLoadButton.disabled = true;
    localRemoveButton.disabled = true;
  }
}
