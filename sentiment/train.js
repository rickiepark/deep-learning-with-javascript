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

import * as tf from '@tensorflow/tfjs';
import {ArgumentParser} from 'argparse';
import * as fs from 'fs';
import * as path from 'path';
import * as shelljs from 'shelljs';

import {loadData, loadMetadataTemplate} from './data';
import {writeEmbeddingMatrixAndLabels} from './embedding';

/**
 * IMDb 감성 분석을 위한 모델을 만듭니다.
 *
 * @param {string} modelType 생성할 모델의 종류
 * @param {number} vocabularySize 입력 어휘사전 크기
 * @param {number} embeddingSize 임베딩 층에 사용할 임베딩 벡터 크기
 * @returns 컴파일되지 않은 `tf.Model` 객체
 */
export function buildModel(modelType, maxLen, vocabularySize, embeddingSize) {
  const model = tf.sequential();
  if (modelType === 'multihot') {
    // 'multihot' 모델은 시퀀스에 있는 모든 단어의 멀티-핫 인코딩을 받아
    // 렐루와 시그모이드 활성화 함수를 사용한 밀집 층을 사용해 문장을 분류합니다.
    model.add(tf.layers.dense({
      units: 16,
      activation: 'relu',
      inputShape: [vocabularySize]
    }));
    model.add(tf.layers.dense({
      units: 16,
      activation: 'relu'
    }));
  } else {
    // 다른 모델은 모두 단어 임베딩을 사용합니다.
    model.add(tf.layers.embedding({
      inputDim: vocabularySize,
      outputDim: embeddingSize,
      inputLength: maxLen
    }));
    if (modelType === 'flatten') {
      model.add(tf.layers.flatten());
    } else if (modelType === 'cnn') {
      model.add(tf.layers.dropout({rate: 0.5}));
      model.add(tf.layers.conv1d({
        filters: 250,
        kernelSize: 5,
        strides: 1,
        padding: 'valid',
        activation: 'relu'
      }));
      model.add(tf.layers.globalMaxPool1d({}));
      model.add(tf.layers.dense({units: 250, activation: 'relu'}));
    } else if (modelType === 'simpleRNN') {
      model.add(tf.layers.simpleRNN({units: 32}));
    } else if (modelType === 'lstm') {
      model.add(tf.layers.lstm({units: 32}));
    } else if (modelType === 'bidirectionalLSTM') {
      model.add(tf.layers.bidirectional(
          {layer: tf.layers.lstm({units: 32}), mergeMode: 'concat'}));
    } else {
      throw new Error(`지원하지 않는 모델 종류입니다: ${modelType}`);
    }
  }
  model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  return model;
}

function parseArguments() {
  const parser = new ArgumentParser(
      {description: 'IMDB 감성 분석을 위한 모델을 훈련합니다'});
  parser.addArgument('modelType', {
    type: 'string',
    optionStrings: [
       'multihot', 'flatten', 'cnn', 'simpleRNN', 'lstm', 'bidirectionalLSTM'],
    help: '모델 종류'
  });
  parser.addArgument('--numWords', {
    type: 'int',
    defaultValue: 10000,
    help: '어휘 사전의 단어 개수'
  });
  parser.addArgument('--maxLen', {
    type: 'int',
    defaultValue: 100,
    help: '최대 문장 길이(단어 개수). ' +
        '짧은 문장은 패딩되고 긴 문장은 잘립니다.'
  });
  parser.addArgument('--embeddingSize', {
    type: 'int',
    defaultValue: 128,
    help: '단어 임베딩 차원의 수'
  });
  parser.addArgument(
      '--gpu', {action: 'storeTrue', help: 'GPU를 사용해 훈련합니다'});
  parser.addArgument('--optimizer', {
    type: 'string',
    defaultValue: 'adam',
    help: '모델 훈련에 사용할 옵티마이저'
  });
  parser.addArgument(
      '--epochs',
      {type: 'int', defaultValue: 10, help: '훈련 에포크 횟수'});
  parser.addArgument(
      '--batchSize',
      {type: 'int', defaultValue: 128, help: '훈련 배치 크기'});
  parser.addArgument('--validationSplit', {
    type: 'float',
    defaultValue: 0.2,
    help: '검증 세트 비율'
  });
  parser.addArgument('--modelSaveDir', {
    type: 'string',
    defaultValue: 'dist/resources',
    help: '모델 저장 경로'
  });
  parser.addArgument('--embeddingFilesPrefix', {
    type: 'string',
    defaultValue: '',
    help: '임베딩 파일을 저장할 경로 프리픽스. ' +
    '이 파일을 임베딩 프로젝터(https://projector.tensorflow.org/)에 업로드할 수 있습니다. '  +
    '예를 들어 /tmp/embed로 지정하면 임베딩 벡터 파일이 ' +
    '/tmp/embed_vectors.tsv에 저장되고 레이블 파일이 /tmp/embed_label.tsv에 저장됩니다.'
  });
  parser.addArgument('--logDir', {
    type: 'string',
    help: '훈련하는 동안 손실과 정확도를 기록할 텐서보드 로그 디렉토리'
  });
  parser.addArgument('--logUpdateFreq', {
    type: 'string',
    defaultValue: 'batch',
    optionStrings: ['batch', 'epoch'],
    help: '텐서보드에 손실과 정확도를 기록할 빈도'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArguments();

  let tfn;
  if (args.gpu) {
    console.log('GPU를 사용하여 훈련합니다');
    tfn = require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('CPU를 사용하여 훈련합니다');
    tfn = require('@tensorflow/tfjs-node');
  }

  console.log('데이터 로딩 중...');
  const multihot = args.modelType === 'multihot';
  const {xTrain, yTrain, xTest, yTest} =
      await loadData(args.numWords, args.maxLen, multihot);

  console.log('모델 구축 중...');
  const model = buildModel(
      args.modelType, args.maxLen, args.numWords, args.embeddingSize);

  model.compile({
    loss: 'binaryCrossentropy',
    optimizer: args.optimizer,
    metrics: ['acc']
  });
  model.summary();

  console.log('모델 훈련 중...');
  await model.fit(xTrain, yTrain, {
    epochs: args.epochs,
    batchSize: args.batchSize,
    validationSplit: args.validationSplit,
    callbacks: args.logDir == null ? null : tfn.node.tensorBoard(args.logDir, {
      updateFreq: args.logUpdateFreq
    })
  });

  console.log('모델 평가 중...');
  const [testLoss, testAcc] =
      model.evaluate(xTest, yTest, {batchSize: args.batchSize});
  console.log(`Evaluation loss: ${(await testLoss.data())[0].toFixed(4)}`);
  console.log(`Evaluation accuracy: ${(await testAcc.data())[0].toFixed(4)}`);

  // 모델 저장
  let metadata;
  if (args.modelSaveDir != null && args.modelSaveDir.length > 0) {
    if (multihot) {
      console.warn(
          '멀티-핫 모델은 지원하지 않으므로 저장하지 않습니다.');
    } else {
      // 베이스 디렉토리를 먼저 만듭니다.
      shelljs.mkdir('-p', args.modelSaveDir);

      // 메타데이터 템플릿을 로드합니다.
      console.log('메타데이터 템플릿 로딩 중...');
      metadata = await loadMetadataTemplate();

      // 메타데이터 저장
      metadata.epochs = args.epochs;
      metadata.embedding_size = args.embeddingSize;
      metadata.max_len = args.maxLen;
      metadata.model_type = args.modelType;
      metadata.batch_size = args.batchSize;
      metadata.vocabulary_size = args.numWords;
      const metadataPath = path.join(args.modelSaveDir, 'metadata.json');
      fs.writeFileSync(metadataPath, JSON.stringify(metadata));
      console.log(`Saved metadata to ${metadataPath}`);

      // 모델 저장
      await model.save(`file://${args.modelSaveDir}`);
      console.log(`Saved model to ${args.modelSaveDir}`);
    }
  }

  if (args.embeddingFilesPrefix != null &&
      args.embeddingFilesPrefix.length > 0) {
    if (metadata == null) {
      metadata = await loadMetadataTemplate();
    }
    await writeEmbeddingMatrixAndLabels(
        model, args.embeddingFilesPrefix, metadata.word_index,
        metadata.index_from);
  }
}

if (require.main === module) {
  main();
}
