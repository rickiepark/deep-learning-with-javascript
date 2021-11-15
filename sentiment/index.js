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

import * as loader from './loader.js';
import * as ui from './ui.js';
import {OOV_INDEX, padSequences} from './sequence_utils.js';

const HOSTED_URLS = {
  model:
      'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
  metadata:
      'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
};

const LOCAL_URLS = {
  model: './resources/model.json',
  metadata: './resources/metadata.json'
};

class SentimentPredictor {
  /**
   * 감성 분석 예제 초기화
   */
  async init(urls) {
    this.urls = urls;
    this.model = await loader.loadHostedPretrainedModel(urls.model);
    await this.loadMetadata();
    return this;
  }

  async loadMetadata() {
    const sentimentMetadata =
        await loader.loadHostedMetadata(this.urls.metadata);
    ui.showMetadata(sentimentMetadata);
    this.indexFrom = sentimentMetadata['index_from'];
    this.maxLen = sentimentMetadata['max_len'];
    console.log('indexFrom = ' + this.indexFrom);
    console.log('maxLen = ' + this.maxLen);

    this.wordIndex = sentimentMetadata['word_index'];
    this.vocabularySize = sentimentMetadata['vocabulary_size'];
    console.log('vocabularySize = ', this.vocabularySize);
  }

  predict(text) {
    // 소문자로 바꾸고 구둣점을 삭제합니다.
    const inputText =
        text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
    // 단어를 단어 인덱스의 시퀀스로 바꿉니다.
    const sequence = inputText.map(word => {
      let wordIndex = this.wordIndex[word] + this.indexFrom;
      if (wordIndex > this.vocabularySize) {
        wordIndex = OOV_INDEX;
      }
      return wordIndex;
    });
    // 자르거나 패딩을 추가합니다.
    const paddedSequence = padSequences([sequence], this.maxLen);
    const input = tf.tensor2d(paddedSequence, [1, this.maxLen]);

    const beginMs = performance.now();
    const predictOut = this.model.predict(input);
    const score = predictOut.dataSync()[0];
    predictOut.dispose();
    const endMs = performance.now();

    return {score: score, elapsed: (endMs - beginMs)};
  }
};

/**
 * 사전 훈련된 모델과 메타데이터를 로드하고 예측 함수에 등록합니다.
 */
async function setupSentiment() {
  if (await loader.urlExists(HOSTED_URLS.model)) {
    ui.status('사용할 모델: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-pretrained-remote');
    button.addEventListener('click', async () => {
      const predictor = await new SentimentPredictor().init(HOSTED_URLS);
      ui.prepUI(x => predictor.predict(x));
    });
    button.style.display = 'inline-block';
  }

  if (await loader.urlExists(LOCAL_URLS.model)) {
    ui.status('사용할 모델: ' + LOCAL_URLS.model);
    const button = document.getElementById('load-pretrained-local');
    button.addEventListener('click', async () => {
      const predictor = await new SentimentPredictor().init(LOCAL_URLS);
      ui.prepUI(x => predictor.predict(x));
    });
    button.style.display = 'inline-block';
  }

  ui.status('대기중');
}

setupSentiment();
