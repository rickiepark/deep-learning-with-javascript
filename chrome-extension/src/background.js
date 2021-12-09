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

import 'babel-polyfill';
import * as tf from '@tensorflow/tfjs';
import {IMAGENET_CLASSES} from './imagenet_classes';

// 로드할 모델 경로
const MOBILENET_MODEL_TFHUB_URL =
    'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2'
// mobilenet이 기대하는 이미지 크기
const IMAGE_SIZE = 224;
// 분류를 위한 최소 이미지 크기. 이 크기 아래는 확장 프로그램이 이미지 분류를 거부합니다.
const MIN_IMG_SIZE = 128;

// 선택할 예측 개수
const TOPK_PREDICTIONS = 2;
const FIVE_SECONDS_IN_MS = 5000;
/**
 * 오른쪽 클릭으로 열린 메뉴 옵션을 클릭했을 때 취할 행동.
 * 오른쪽 클릭된 이미지의 url과 현재 탭 ID를 가져와서 imageClassifier의 analyzeImage 메서드로 전달합니다.
 */
function clickMenuCallback(info, tab) {
  imageClassifier.analyzeImage(info.srcUrl, tab.id);
}

/**
 * 이미지 분류 실행을 위한 오른쪽 클릭 메뉴 옵션을 추가합니다.
 * 이 메뉴 옵션은 이미지를 오른쪽 클릭 했을 때만 나타납니다.
 */
chrome.contextMenus.create({
  title: 'Classify image with TensorFlow.js ',
  contexts: ['image'],
  onclick: clickMenuCallback
});

/**
 * 생성자에서 mobilenet을 로드합니다.
 * analyzeImage 메서드에서 이미지 분류 요청을 처리합니다.
 * 요청이 성공하면 'IMAGE_CLICK_PROCESSED' action과 함께 크롬 메시지가 게시되며
 * content.js가 이를 듣고 DOM을 조작하는데 사용할 수 있습니다.
 */
class ImageClassifier {
  constructor() {
    this.loadModel();
  }

  /**
   * URL에서 mobilenet을 로드하고 이 객체 안에 참조를 유지합니다.
   */
  async loadModel() {
    console.log('모델 로딩 중...');
    const startTime = performance.now();
    try {
      this.model =
          await tf.loadGraphModel(MOBILENET_MODEL_TFHUB_URL, {fromTFHub: true});
      // 중간 텐서 값을 만들어 GPU로 보내어 모델을 워밍업합니다.
      tf.tidy(() => {
        this.model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]));
      });
      const totalTime = Math.floor(performance.now() - startTime);
      console.log(`모델 로딩 및 초기화 시간: ${totalTime} ms...`);
    } catch {
      console.error(
          `다음 URL에서 모델을 로드할 수 없습니다: ${MOBILENET_MODEL_TFHUB_URL}`);
    }
  }

  /**
   * url에 있는 이미지에 대해 예측을 만들기 위해 모델을 호출합니다.
   * 성공적으로 예측이 수행되면 IMAGE_CLICK_PROCESSED 메시지가 게시되며
   * content.js 스크립트가 이를 듣고 예측 결과를 사용해 DOM을 업데이트합니다.
   *
   * @param {string} url 이미지 URL
   * @param {number} tabId 요청이 발생한 탭
   */
  async analyzeImage(url, tabId) {
    if (!tabId) {
      console.error('탭이 없습니다.');
      return;
    }
    if (!this.model) {
      console.log('모델 로딩 중...');
      setTimeout(() => {this.analyzeImage(url)}, FIVE_SECONDS_IN_MS);
      return;
    }
    let message;
    this.loadImage(url).then(
        async (img) => {
          if (!img) {
            console.error(
                '이미지를 로드할 수 없습니다. 너무 작거나 가져올 수 없습니다.');
            return;
          }
          const predictions = await this.predict(img);
          message = {action: 'IMAGE_CLICK_PROCESSED', url, predictions};
          chrome.tabs.sendMessage(tabId, message);
        },
        (reason) => {
          console.error(`분석 실패: ${reason}`);
        });
  }

  /**
   * dom 요소를 만들고 제공된 src에 있는 이미지를 로드합니다.
   * @param {string} src 로드할 이미지 URL
   */
  async loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = document.createElement('img');
      img.crossOrigin = 'anonymous';
      img.onerror = function(e) {
        reject(`${src}에서 이미지를 로드할 수 없습니다.`);
      };
      img.onload = function(e) {
        if ((img.height && img.height > MIN_IMG_SIZE) ||
            (img.width && img.width > MIN_IMG_SIZE)) {
          img.width = IMAGE_SIZE;
          img.height = IMAGE_SIZE;
          resolve(img);
        }
        // 이미지의 양 차원이 MIN_IMG_SIZE보다 작을 경우 실패합니다.
        reject(`이미지 크기가 너무 작습니다. [${img.height} x ${
            img.width}] vs. 최솟값 [${MIN_IMG_SIZE} x ${MIN_IMG_SIZE}]`);
      };
      img.src = src;
    });
  }

  /**
   * 점수 순으로 예측을 정렬하고 topK 예측만 취합니다.
   * @param {Tensor} logits 하나의 원소가 mobilenet의 클래스에 대응되는 텐서로 model.predict 메서드의 반환값
   * @param {number} topK 선택 개수
   */
  async getTopKClasses(logits, topK) {
    const {values, indices} = tf.topk(logits, topK, true);
    const valuesArr = await values.data();
    const indicesArr = await indices.data();
    console.log(`indicesArr ${indicesArr}`);
    const topClassesAndProbs = [];
    for (let i = 0; i < topK; i++) {
      topClassesAndProbs.push({
        className: IMAGENET_CLASSES[indicesArr[i]],
        probability: valuesArr[i]
      })
    }
    return topClassesAndProbs;
  }

  /**
   * 입력 이미지에 대해 모델을 실행하고 최상위 예측 클래스를 반환합니다.
   * @param {HTMLElement} imgElement 예측할 이미지를 담은 HTML 요소. mobilenet의 크기와 맞아야 합니다.
   */
  async predict(imgElement) {
    console.log('예측 중...');
    // 첫 번째 시간은 predict() 호출 이외에도 이미지를 HTML에서 추출하고 전처리하는 시간을 포함합니다.
    const startTime1 = performance.now();
    // 두 번째 시간은 추출과 전처리 시간을 제외하고 predict() 호출만 포함합니다.
    let startTime2;
    const logits = tf.tidy(() => {
      // Mobilenet은 -1~1 사이로 정규화된 이미지를 기대합니다.
      const img = tf.browser.fromPixels(imgElement).toFloat();
      // const offset = tf.scalar(127.5);
      // const normalized = img.sub(offset).div(offset);
      const normalized = img.div(tf.scalar(256.0));
      const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
      startTime2 = performance.now();
      const output = this.model.predict(batched);
      if (output.shape[output.shape.length - 1] === 1001) {
        // 맨 처음 로짓(백그라운드 잡음)을 삭제합니다.
        return output.slice([0, 1], [-1, 1000]);
      } else if (output.shape[output.shape.length - 1] === 1000) {
        return output;
      } else {
        throw new Error('기대하지 않는 크기입니다...');
      }
    });

    // 로짓을 확률과 클래스 이름으로 변환합니다.
    const classes = await this.getTopKClasses(logits, TOPK_PREDICTIONS);
    const totalTime1 = performance.now() - startTime1;
    const totalTime2 = performance.now() - startTime2;
    console.log(
        `걸린 시간: ${totalTime1.toFixed(1)} ms ` +
        `(전처리 제외한 시간: ${Math.floor(totalTime2)} ms)`);
    return classes;
  }
}

const imageClassifier = new ImageClassifier();
