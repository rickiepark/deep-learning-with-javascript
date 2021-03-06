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

/**
 * 날짜 변환 예제의 추론 파트
 *
 * 파이썬 케라스 예제를 참고했습니다:
 *   https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py
 */

import {generateRandomDateTuple, INPUT_FNS, INPUT_LENGTH} from './date_format.js';
import {runSeq2SeqInference} from './model.js';

const RELATIVE_MODEL_URL = './model/model.json';
const HOSTED_MODEL_URL =
    'https://storage.googleapis.com/tfjs-examples/date-conversion-attention/dist/model/model.json';

const status = document.getElementById('status');
const inputDateString = document.getElementById('input-date-string');
const outputDateString = document.getElementById('output-date-string');
const attentionHeatmap = document.getElementById('attention-heatmap');
const randomButton = document.getElementById('random-date');

let model;

inputDateString.addEventListener('change', async () => {
  let inputStr = inputDateString.value.trim().toUpperCase();
  if (inputStr.length < 6) {
    outputDateString.value = '';
    return;
  }

  if (inputStr.length > INPUT_LENGTH) {
    inputStr = inputStr.slice(0, INPUT_LENGTH);
  }

  try {
    const getAttention = true;
    const t0 = tf.util.now();
    const {outputStr, attention} =
        await runSeq2SeqInference(model, inputStr, getAttention);
    const tElapsed = tf.util.now() - t0;
    status.textContent = `seq2seq 변환 시간: ${tElapsed.toFixed(1)} ms`;
    outputDateString.value = outputStr;

    const xTickLabels = outputStr.split('').map(
        (char, i) => `(${integerToTwoDigitString(i + 1)}) "${char}"`);
    const yTickLabels = [];
    for (let i = 0; i < INPUT_LENGTH; ++i) {
      if (i < inputStr.length) {
        yTickLabels.push(`(${integerToTwoDigitString(i + 1)}) "${inputStr[i]}"`);
      } else {
        yTickLabels.push(`(${integerToTwoDigitString(i + 1)}) ""`);
      }
    }
    await tfvis.render.heatmap(
        attentionHeatmap, {
          values: attention.squeeze([0]),
          xTickLabels,
          yTickLabels
        }, {
          width: 600,
          height: 360,
          xLabel: '출력 문자',
          yLabel: '입력 문자',
          colorMap: 'blues'
        });
  } catch (err) {
    outputDateString.value = err.message;
    console.error(err);
  }
});

randomButton.addEventListener('click', async () => {
  const inputFn = INPUT_FNS[Math.floor(Math.random() * INPUT_FNS.length)];
  inputDateString.value = inputFn(generateRandomDateTuple());
  inputDateString.dispatchEvent(new Event('change'));
});

function integerToTwoDigitString(x) {
  const str = `${x / 100}`.substr(2);
  return str.length == 1 ? str + '0' : str;
}

async function init() {
  try {
    status.textContent = `모델 로딩: ${RELATIVE_MODEL_URL} ...`;
    model = await tf.loadLayersModel(RELATIVE_MODEL_URL);
  } catch (err) {
    // 로컬 모델을 로드하는데 실패하면 원격 모델을 로드합니다.
    status.textContent = `원격 모델 로딩: ${HOSTED_MODEL_URL} ...`;
    model = await tf.loadLayersModel(HOSTED_MODEL_URL);
  }
  status.textContent = '모델 로딩 완료.';
  model.summary();

  const exampleItems = document.getElementsByClassName('input-date-example');
  for (const exampleItem of exampleItems) {
    exampleItem.addEventListener('click', (event) => {
      inputDateString.value = event.srcElement.textContent;
      inputDateString.dispatchEvent(new Event('change'));
    });
  }

  inputDateString.dispatchEvent(new Event('change'));
}

init();
