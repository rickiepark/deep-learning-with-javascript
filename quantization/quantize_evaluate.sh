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

# `yarn train` 명령으로 저장한 모델을 양자화하고
# 여러 양자화 방식(8비트와 16비트)에서 테스트 정확도를 평가합니다.

set -e

MODEL_NAME=$1
if [[ -z "${MODEL_NAME}" ]]; then
  echo "사용법: quantize_evaluate <MODEL_NAME>"
  exit 1
fi

# 모델 경로
MODEL_ROOT="models/${MODEL_NAME}"
MODEL_PATH="${MODEL_ROOT}/original"
MODEL_JSON_PATH="${MODEL_PATH}/model.json"

# pip3 확인
if [[ -z "$(which pip3)" ]]; then
  echo "에러: pip3를 찾을 수 없습니다."
  echo "       파이썬과 pip3를 설치하세요."
  exit 1
fi

if [[ -z "$(which virtualenv)" ]]; then
  echo "virtualenv 설치 중..."
  pip3 install virtualenv
fi

VENV_DIR="$(mktemp -d)_venv"
echo "${VENV_DIR}에 가상 환경을 만듭니다..."
virtualenv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip3 install tensorflowjs

if [[ "${MODEL_NAME}" == "MobileNetV2" ]]; then
  # 먼저 MobilNetV2 모델을 저장합니다.
  if [[ ! -f "${MODEL_JSON_PATH}" ]]; then
    python save_mobilenetv2.py
  fi
fi

if [[ ! -f "${MODEL_JSON_PATH}" ]]; then
  echo "에러: ${MODEL_JSON_PATH}에서 JSON 파일을 찾을 수 없습니다."
  echo "       먼저 yarn train 명령으로 모델을 훈련하고 저장하세요."
  rm -rf "${VENV_DIR}"
  exit 1
fi

# 16비트 양자화 수행
MODEL_PATH_16BIT="${MODEL_ROOT}/quantized-16bit"
rm -rf "${MODEL_PATH_16BIT}"
tensorflowjs_converter \
    --input_format tfjs_layers_model \
    --output_format tfjs_layers_model \
    --quantization_bytes 2 \
    "${MODEL_JSON_PATH}" "${MODEL_PATH_16BIT}"

# 8비트 양자화 수행
MODEL_PATH_8BIT="${MODEL_ROOT}/quantized-8bit"
rm -rf "${MODEL_PATH_8BIT}"
tensorflowjs_converter \
    --input_format tfjs_layers_model \
    --output_format tfjs_layers_model \
    --quantization_bytes 1 \
    "${MODEL_JSON_PATH}" "${MODEL_PATH_8BIT}"

# virtualenv 삭제
rm -rf "${VENV_DIR}"

yarn

if [[ "${MODEL_NAME}" == "MobileNetV2" ]]; then
  # MobileNetV2 평가에 필요한 데이터를 다운로드합니다.
  IMAGENET_1000_SAMPLES_DIR="imagenet-1000-samples"

  if [[ ! -d "${IMAGENET_1000_SAMPLES_DIR}" ]]; then
    curl -o imagenet-1000-samples.tar.gz \
        https://storage.googleapis.com/tfjs-examples/quantization/data/imagenet-1000-samples.tar.gz
    mkdir -p ${IMAGENET_1000_SAMPLES_DIR}
    tar xf imagenet-1000-samples.tar.gz
    rm imagenet-1000-samples.tar.gz
  fi

  # 양자화를 사용하지 않았을 때 정확도를 평가합니다(즉, 32비트 정밀도의 가중치).
  echo "=== 정확도 평가: 양자화 없음 ==="
  yarn "eval-${MODEL_NAME}" "${MODEL_JSON_PATH}" \
      "${IMAGENET_1000_SAMPLES_DIR}"

  # 16비트 양자화의 정확도를 평가합니다.
  echo "=== 정확도 평가: 16비트 양자화 ==="
  yarn "eval-${MODEL_NAME}" "${MODEL_PATH_16BIT}/model.json" \
      "${IMAGENET_1000_SAMPLES_DIR}"

  # 8비트 양자화의 정확도를 평가합니다.
  echo "=== 정확도 평가: 8비트 양자화 ==="
  yarn "eval-${MODEL_NAME}" "${MODEL_PATH_8BIT}/model.json" \
      "${IMAGENET_1000_SAMPLES_DIR}"
else
  # 양자화를 사용하지 않았을 때 정확도를 평가합니다(즉, 32비트 정밀도의 가중치).
  echo "=== 정확도 평가: 양자화 없음 ==="
  yarn "eval-${MODEL_NAME}" "${MODEL_JSON_PATH}"

  # 16비트 양자화의 정확도를 평가합니다.
  echo "=== 정확도 평가: 16비트 양자화 ==="
  yarn "eval-${MODEL_NAME}" "${MODEL_PATH_16BIT}/model.json"

  # 8비트 양자화의 정확도를 평가합니다.
  echo "=== 정확도 평가: 8비트 양자화 ==="
  yarn "eval-${MODEL_NAME}" "${MODEL_PATH_8BIT}/model.json"
fi

function calc_gzip_ratio() {
  ORIGINAL_FILES_SIZE_BYTES="$(ls -lAR ${1} | grep -v '^d' | awk '{total += $5} END {print total}')"
  TEMP_TARBALL="$(mktemp)"
  tar czf "${TEMP_TARBALL}" "${1}"
  TARBALL_SIZE="$(wc -c < ${TEMP_TARBALL})"
  ZIP_RATIO="$(awk "BEGIN { print(${ORIGINAL_FILES_SIZE_BYTES} / ${TARBALL_SIZE}) }")"
  rm "${TEMP_TARBALL}"

  echo "  전체 파일 크기: ${ORIGINAL_FILES_SIZE_BYTES} bytes"
  echo "  gzip 압축 크기: ${TARBALL_SIZE} bytes"
  echo "  gzip 비율: ${ZIP_RATIO}"
  echo
}

echo
echo "=== gzip 비율 ==="

# 원본 모델의 gzip 비율을 계산합니다.
echo "원본 모델 (양자화 없음):"
calc_gzip_ratio "${MODEL_PATH}"

# 16비트 양자화된 모델의 gzip 비율을 계산합니다.
echo "16비트 양자화 모델:"
calc_gzip_ratio "${MODEL_PATH_16BIT}"

# 8비트 양자화된 모델의 gzip 비율을 계산합니다.
echo "8비트 양자화 모델:"
calc_gzip_ratio "${MODEL_PATH_8BIT}"
