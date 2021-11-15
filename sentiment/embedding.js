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

/**
 * 임베딩 행렬의 압축을 풀고 파일로 출력하기 위한 유틸리티
 */

import {writeFileSync} from 'fs';
import * as tf from '@tensorflow/tfjs';

/**
 * TensorFlow.js 모델에서 첫 번째 임베딩 행렬을 추출합니다.
 *
 * @param {tf.model} model tf.Model 객체. 임베딩 행렬을 포함하고 있다고 가정합니다.
 * @retuns {tf.Tensor} 모델의 모든 층을 순회하면서 만난 첫 번째 임베딩 층의 임베딩 행렬
 * @throws 모델에 임베딩 층이 없다면 에러가 발생합니다.
 */
function extractEmbeddingMatrix(model) {
  for (const layer of model.layers) {
    if (layer.getClassName() === 'Embedding') {
      const embed = layer.getWeights()[0];
      tf.util.assert(
        embed.rank === 2,
        `임베딩 행렬의 랭크는 2이여야 하지만, ` +
        `${embed.rank}을 얻었습니다.`);
      return embed;
    }
  }
  throw new Error('모델에서 임베딩층을 찾을 수 없습니다.');
}

/**
 * 모델의 첫 번째 임베딩 행렬의 값을 파일로 저장합니다.
 *
 * 단어 레이블도 저장합니다.
 * 벡터와 레이블 파일은 임베딩 프로젝터(https://projector.tensorflow.org/)에 업로드할 수 있습니다.
 *
 * @param {tf.model} model tf.Model 객체, 임베딩 층을 가지고 있다고 가정합니다.
 * @param {string} prefix 벡터와 레이블 파일 작성을 위한 프리픽스(prefix) 경로
 *   예를 들어 `prefix`가 `/tmp/embed`이면,
 *   - 벡터는 `/tmp/embed_vectors.tsv`에 저장됩니다.
 *   - 레이블은 `/tmp/embed_labels.tsv`에 저장됩니다.
 * @param {{[word: string]: number}} wordIndex 단어를 정수 인덱스에 매핑한 딕셔너리
 * @param {number} indexFrom 정수 인덱스의 베이스 값
 */
export async function writeEmbeddingMatrixAndLabels(
    model, prefix, wordIndex, indexFrom) {
  tf.util.assert(
      prefix != null && prefix.length > 0,
      `Null, undefined 또는 빈 경로입니다`);

  const embed = extractEmbeddingMatrix(model);

  const numWords = embed.shape[0];
  const embedDims = embed.shape[1];
  const embedData = await embed.data();

  // 임베딩 행렬을 파일에 저장합니다
  let vectorsStr = '';
  let index = 0;
  for (let i = 0; i < numWords; ++i) {
    for (let j = 0; j < embedDims; ++j) {
      vectorsStr += embedData[index++].toFixed(5);
      if (j < embedDims - 1) {
        vectorsStr += '\t';
      } else {
        vectorsStr += '\n';
      }
    }
  }

  const vectorsFilePath = `${prefix}_vectors.tsv`;
  writeFileSync(vectorsFilePath, vectorsStr, {encoding: 'utf-8'});
  console.log(
      `임베딩 벡터 (${numWords} * ${embedDims}) 저장: ` +
      `${vectorsFilePath}`);

  // 단어 레이블을 모아 저장합니다.
  const indexToWord = {};
  for (const word in wordIndex) {
    indexToWord[wordIndex[word]] = word;
  }

  let labelsStr = '';
  for(let i = 0; i < numWords; ++i) {
    if (i >= indexFrom) {
      labelsStr += indexToWord[i - indexFrom];
    } else {
      labelsStr += 'not-a-word';
    }
    labelsStr += '\n';
  }

  const labelsFilePath = `${prefix}_labels.tsv`;
  writeFileSync(labelsFilePath, labelsStr, {encoding: 'utf-8'});
  console.log(
      `임베딩 레이블 (${numWords}) 저장: ` +
      `${labelsFilePath}`);
}
