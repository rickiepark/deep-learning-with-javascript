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

import {IMAGENET_CLASSES} from './imagenet_classes.js';

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;

let mobilenet;
const mobilenetDemo = async () => {
  status('모델 로딩 중...');

  mobilenet = await tf.loadGraphModel(MOBILENET_MODEL_PATH, {fromTFHub: true});

  // 모델을 워밍업합니다. 필수적이진 않지만 첫 번째 추론 속도를 높일 수 있습니다.
  // `dispose` 메서드를 호출하여 `predict` 메서드에서 반환한 값에 할당된 WebGL 메모리를 해제합니다.
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status('');

  // 로컬에 있는 cat.jpg 이미지를 사용해 추론을 만듭니다.
  const catElement = document.getElementById('cat');
  if (catElement.complete && catElement.naturalHeight !== 0) {
    predict(catElement);
    catElement.style.display = '';
  } else {
    catElement.onload = () => {
      predict(catElement);
      catElement.style.display = '';
    }
  }

  document.getElementById('file-container').style.display = '';
};

/**
 * 이미지 요소에서 top-k 클래스 확률을 반환하는 mobilenet을 사용해 예측을 만듭니다.
 */
async function predict(imgElement) {
  status('예측 중...');

  // 첫 번째 시작 시간에는 predict() 호출 이외에도 HTML에서 이미지 추출과 전처리 시간이 포함됩니다.
  const startTime1 = performance.now();
  // 두 번째 시작 시간에서는 추출과 전처리 시간을 제외하고 predict() 호출만 포함합니다.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels()은 이미지 요소에서 텐서를 반환합니다.
    const img = tf.cast(tf.browser.fromPixels(imgElement), 'float32');

    const offset = tf.scalar(127.5);
    // [0, 255] 사이에서 [-1, 1] 사이로 이미지를 정규화합니다.
    const normalized = img.sub(offset).div(offset);

    // predict() 메서드에 전달할 수 있도록 하나의 원소를 가진 배치로 크기를 바꿉니다.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // mobilenet으로 예측을 만듭니다.
    return mobilenet.predict(batched);
  });

  // 로짓을 확률과 클래스 이름으로 바꿉니다.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`걸린 시간: ${Math.floor(totalTime1)} ms ` +
      `(전처리 제외한 시간: ${Math.floor(totalTime2)} ms)`);

  // DOM에 클래스를 출력합니다.
  showResults(imgElement, classes);
}

/**
 * 주어진 로짓을 사용해 소프트맥스를 계산하여 확률을 얻고 소팅하여 top-k 클래스의 확률을 구합니다.
 * @param logits MobileNet에서 반환한 로짓 텐서
 * @param topK 출력할 최상위 예측 개수
 */
export async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

//
// UI
//

function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(3);
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // 썸네일을 출력하고 각 이미지에 대해 예측을 수행합니다.
  for (let i = 0, f; f = files[i]; i++) {
    // 이미지 파일만 처리합니다(이미지가 아닌 파일은 건너 뜁니다).
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    reader.onload = e => {
      // 이미지를 채우고 예측을 수행합니다.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // 데이터 URL로 이미지 파일을 읽습니다.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

mobilenetDemo();
