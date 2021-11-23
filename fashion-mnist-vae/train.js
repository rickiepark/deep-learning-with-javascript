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

const path = require('path');
const _ = require('lodash');
const mkdirp = require('mkdirp');
const argparse = require('argparse');

// `--gpu` 플래그가 지정되었는지에 따라 tf 모듈이 동적으로 로드됩니다.
let tf;

const {
  DATASET_PATH,
  TRAIN_IMAGES_FILE,
  IMAGE_FLAT_SIZE,
  loadImages,
  previewImage,
  batchImages,
} = require('./data');

const {encoder, decoder, vae, vaeLoss} = require('./model');

let epochs;
let batchSize;

const INTERMEDIATE_DIM = 512;
const LATENT_DIM = 2;

/**
 * 변이형 오토인코더 훈련
 *
 * @param {number[][]} images VAE 훈련을 위한 펼친 이미지
 * @param {object} vaeOpts VAE 모델 옵션. 다음 필드를 포함합니다:
 *   - originaDim {number} 펼친 입력 이미지 길이
 *   - intermediateDim {number} 중간 (은닉) 밀집 층의 유닛 개수
 *   - latentDim {number} 잠재 공간(즉, z-공간)의 차원수
 * @param {string} savePath 훈련이 끝난 뒤 VAE 모델의 디코더 부분을 저장할 경로
 * @param {string?} logDir 로그 디렉토리 경로. 이 값을 지정하면
 *   훈련 배치마다 손실 값이 이 디렉토리에 기록되므로
 *   텐서보드를 사용해 훈련 과정을 모니터링할 수 있습니다.
 */
async function train(images, vaeOpts, savePath, logDir) {
  const encoderModel = encoder(vaeOpts);
  const decoderModel = decoder(vaeOpts);
  const vaeModel = vae(encoderModel, decoderModel);

  let summaryWriter;
  if (logDir != null) {
    console.log(`손실 값을 ${logDir}에 기록합니다.`);
    console.log(
        `텐서보드 서버를 실행하려면 다음 명령을 사용하세요:`);
    console.log(`  tensorboard --logdir ${logDir}`);
    summaryWriter = tf.node.summaryFileWriter(logDir);
  }

  console.log('\n** 모델 훈련 **\n');

  // 사용자 손실 함수를 사용하기 때문에 일반적인 model.fit 대신에
  // optimizer.minimize를 사용합니다. 따사러 옵티마이저를 정의하고 배치 데이터를 직접 관리해야 합니다.

  // 옵티마이저를 만듭니다.
  const optimizer = tf.train.adam();

  // 데이터 배치를 만듭니다.
  const batches = _.chunk(images, batchSize);

  // 훈련 반복을 실행합니다.
  let step = 0;
  for (let i = 0; i < epochs; i++) {
    console.log(`\n에포크: #${i + 1} / ${epochs}\n`);
    for (let j = 0; j < batches.length; j++) {
      const currentBatchSize = batches[j].length
      const batchedImages = batchImages(batches[j]);

      const reshaped =
          batchedImages.reshape([currentBatchSize, vaeOpts.originalDim]);

      // 모델 최적화 단계입니다.
      // optimizer.minimize가 모델의 가중치를 조정할 수 있도록
      // 예측을 만들고 손실을 계산하여 반환합니다.
      optimizer.minimize(() => {
        const outputs = vaeModel.apply(reshaped);
        const loss = vaeLoss(reshaped, outputs, vaeOpts);
        process.stdout.write('.');
        if (j % 50 === 0) {
          console.log('\n손실:', loss.dataSync()[0]);
        }
        if (summaryWriter != null) {
          summaryWriter.scalar('loss', loss, step++);
        }

        return loss;
      });
      tf.dispose([batchedImages, reshaped]);
    }
    console.log('');
    // 에포크가 끝날 때마다 프리뷰를 만듭니다.
    await generate(decoderModel, vaeOpts.latentDim);
  }

  console.log('훈련 종료');
  saveDecoder(savePath, decoderModel);
}

/**
 * 이미지를 생성하고 콘솔에 출력합니다.
 *
 * @param {tf.LayersModel} decoderModel VAE의 디코더
 * @param {number} latentDimSize 잠재 공간의 차원수
 */
async function generate(decoderModel, latentDimSize) {
  const targetZ = tf.zeros([latentDimSize]).expandDims();
  const generated = (decoderModel.predict(targetZ));

  await previewImage(generated.dataSync());
  tf.dispose([targetZ, generated]);
}

async function saveDecoder(savePath, decoderModel) {
  const decoderPath = path.join(savePath, 'decoder');
  mkdirp.sync(decoderPath);
  const saveURL = `file://${decoderPath}`;
  console.log(`디코더 저장: ${saveURL}`);
  await decoderModel.save(saveURL);
}

async function run(savePath, logDir) {
  // 데이터 로드
  const dataPath = path.join(DATASET_PATH, TRAIN_IMAGES_FILE);
  const images = await loadImages(dataPath);
  console.log('데이터가 로드되었습니다', images.length);
  await previewImage(images[5]);
  await previewImage(images[50]);
  await previewImage(images[500]);
  // 훈련 시작
  const vaeOpts = {
    originalDim: IMAGE_FLAT_SIZE,
    intermediateDim: INTERMEDIATE_DIM,
    latentDim: LATENT_DIM
  };
  await train(images, vaeOpts, savePath, logDir);
}

(async function() {
  const parser = new argparse.ArgumentParser();
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'tfjs-node-gpu를 사용해 훈련합니다(CUDA와 CuDNN 필요)'
  });
  parser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 100,
    help: '모델을 훈련할 에포크 수'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 256,
    help: '모델 훈련에 사용할 배치 크기'
  });
  parser.addArgument('--logDir', {
    type: 'string',
    help: '텐서보드를 위한 로그를 기록할 디렉토리'
  });
  parser.addArgument('--savePath', {
    type: 'string',
    defaultValue: './models',
    help: '훈련이 끝난 후 VAE 모델의 디코더를 저장할 디렉토리. ' +
    '디렉토리가 없으면 새로 만듭니다.'
  });

  const args = parser.parseArgs();
  epochs = args.epochs;
  batchSize = args.batchSize;

  if (args.gpu) {
    console.log('GPU를 사용해 훈련');
    tf = require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('CPU를 사용해 훈련');
    tf = require('@tensorflow/tfjs-node');
  }

  await run(args.savePath, args.logDir);
})();
