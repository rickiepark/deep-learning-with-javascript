#!/usr/bin/env bash
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# 이 스크립트는 양자화 예제에 있는 여러 yarn 명령을 테스트합니다.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 테스트를 위해 모든 모델을 지우고 시작합니다.
if [[ -d "models" ]]; then
  echo "에러: models/ 디렉토리가 없습니다. 디렉토리를 삭제하거나 "
  echo "       테스트 스크립트를 실행하기 전에 다른 경로로 옮기세요."
  exit 1
fi

if [[ -z "$(which pip3)" ]]; then
  echo "pip3를 찾을 수 없습니다. 설치를 시도합니다..."
  apt-get update
  apt-get install -y python3-pip
fi

yarn

# housing 모델 훈련 스크립트를 호출합니다
yarn train-housing --epochs 1

# housing 모델이 저장되었는지 확인합니다.
HOUSING_MODEL_JSON_PATH="models/housing/original/model.json"
if [[ ! -f "${HOUSING_MODEL_JSON_PATH}" ]]; then
  echo "에러: ${HOUSING_MODEL_JSON_PATH}에서 model.json 파일을 찾을 수 없습니다."
  exit 1
fi

# housing 모델 평가 스크립트 호출
yarn quantize-and-evaluate-housing

# 패션-MNIST 훈련 스크립트를 호출합니다.
# 패션-MNIST 모델 에포크는 오랜 시간이 걸리므로 0으로 설정하니다.
# 그럼에도 모델이 생성되고 디스크에 저장되어야 합니다.
yarn train-fashion-mnist --epochs 0

# 패션-MNIST 모델이 저장되었는지 확인합니다.
FASHION_MNIST_MODEL_JSON_PATH="models/fashion-mnist/original/model.json"
if [[ ! -f "${FASHION_MNIST_MODEL_JSON_PATH}" ]]; then
  echo "에러: ${FASHION_MNIST_MODEL_JSON_PATH}에서 model.json 파일을 찾을 수 없습니다."
  exit 1
fi

yarn quantize-and-evaluate-fashion-mnist

# MNIST 훈련 스크립트를 호출합니다.
# MNIST 모델 에포크는 오랜 시간이 걸리므로 0으로 설정하니다.
# 그럼에도 모델이 생성되고 디스크에 저장되어야 합니다.
yarn train-mnist --epochs 0

# 패션-MNIST 모델이 저장되었는지 확인합니다.
MNIST_MODEL_JSON_PATH="models/mnist/original/model.json"
if [[ ! -f "${MNIST_MODEL_JSON_PATH}" ]]; then
  echo "에러: ${MNIST_MODEL_JSON_PATH}에서 model.json 파일을 찾을 수 없습니다."
  exit 1
fi

yarn quantize-and-evaluate-mnist

# models/ 디렉토리를 삭제합니다.
rm -rf models/
