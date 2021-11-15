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
 * URL에 접속 가능한지 테스트합니다.
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
 * @return 가중치와 모델 구조를 가진 `tf.Model` 객체
 */
export async function loadHostedPretrainedModel(url) {
  ui.status('사전 훈련된 모델 로딩: ' + url);
  try {
    const model = await tf.loadLayersModel(url);
    ui.status('사전 훈련된 모델 로딩 완료.');
    // 다음 이슈 때문에 모델을 두 번 로드할 수 없습니다.
    // https://github.com/tensorflow/tfjs/issues/34
    // 사용자 혼돈을 막기 위해 로드 버튼을 삭제합니다.
    ui.disableLoadModelButtons();
    return model;
  } catch (err) {
    console.error(err);
    ui.status('사전 훈련된 모델의 로딩을 실패했습니다.');
  }
}

/**
 * 원격 ULR에 저장된 메타데이터 파일을 로드합니다.
 *
 * @return 키-값 쌍으로 메타데이터를 담은 객체
 */
export async function loadHostedMetadata(url) {
  ui.status('메타 데이터 로딩: ' + url);
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    ui.status('메타데이터 로딩 완료.');
    return metadata;
  } catch (err) {
    console.error(err);
    ui.status('메타 데이터의 로딩을 실패했습니다.');
  }
}
