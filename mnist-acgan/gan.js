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
 * MNIST 에서 ACGAN을 훈련합니다.
 *
 * 훈련을 시작하려면:
 *
 * ```sh
 * yarn
 * yarn train
 * ```
 *
 * CUDA GPU가 있다면 훈련 속도를 높일 수 있습니다:
 *
 * ```sh
 * yarn
 * yarn train --gpu
 * ```
 *
 * 브라우저에서 데모를 보려면 별도 터미널에서 다음 명령을 사용합니다:
 *
 * ```sh
 * yarn
 * npx http-server
 * ```
 *
 * 합성곱 연산은 tfjs-node를 사용하는 CPU보다 GPU에서 몇 배 빠르기 때문에
 * tfjs-node-gpu를 사용해 CUDA 가능 GPU에서 모델을 훈련하는 것이 좋습니다.
 *
 * ACGAN에 대한 자세한 내용은 다음을 참고하세요:
 * - Augustus Odena, Christopher Olah, Jonathon Shlens. (2017) "Conditional
 *   image synthesis with auxiliary classifier GANs"
 *   https://arxiv.org/abs/1610.09585
 *
 * 다음 구현을 참고했습니다:
 *   https://github.com/keras-team/keras/blob/master/examples/mnist_acgan.py
 */

const fs = require('fs');
const path = require('path');

const argparse = require('argparse');
const data = require('./data');

// MNIST 데이터셋에 있는 클래스 개수.
const NUM_CLASSES = 10;

// MNIST 이미지 크기
const IMAGE_SIZE = 28;

// CPU(tfjs-node)나 GPU(tfjs-node-gpu) 백엔드를 사용하는지에 따라
// tf 값은 동적으로 결정됩니다.
// 이 때문에 `const` 대신에 `let`을 사용합니다.
let tf = require('@tensorflow/tfjs');

/**
 * ACGAN의 생성자를 만듭니다.
 *
 * ACGAN의 생성자는 두 개의 입력을 받습니다:
 *
 *   1. 랜덤한 잠재 공간의 벡터 (GAN 논문에서는 종종 잠재 공간을 z-공간이라고 부릅니다)
 *   2. 원하는 이미지 카테고리 레이블 (0, 1, ..., 9)
 *
 * 하나의 출력을 만듭니다: 생성된 (즉 가짜) 이미지
 *
 * @param {number} latentSize 잠재 공간 크기
 * @returns {tf.LayersModel} 생성자 모델
 */
function buildGenerator(latentSize) {
  tf.util.assert(
      latentSize > 0 && Number.isInteger(latentSize),
      `잠재 공간 크기는 양수여야 합니다. 하지만 입력된 값은 ` +
          `${latentSize}입니다.`);

  const cnn = tf.sequential();

  // 출력의 크기르 바꾸어 이어지는 conv2dTranspose 층에 주입할 수 있도록 유닛 개수를 결정합니다.
  // 결국 마지막 텐서는 MNIST 이미지 크기([28, 28, 1])와 같아야 합니다.
  cnn.add(tf.layers.dense(
      {units: 3 * 3 * 384, inputShape: [latentSize], activation: 'relu'}));
  cnn.add(tf.layers.reshape({targetShape: [3, 3, 384]}));

  // [3, 3, ...]에서 [7, 7, ...]로 업샘플링합니다.
  cnn.add(tf.layers.conv2dTranspose({
    filters: 192,
    kernelSize: 5,
    strides: 1,
    padding: 'valid',
    activation: 'relu',
    kernelInitializer: 'glorotNormal'
  }));
  cnn.add(tf.layers.batchNormalization());

  // [14, 14, ...]로 업샘플링
  cnn.add(tf.layers.conv2dTranspose({
    filters: 96,
    kernelSize: 5,
    strides: 2,
    padding: 'same',
    activation: 'relu',
    kernelInitializer: 'glorotNormal'
  }));
  cnn.add(tf.layers.batchNormalization());

  // [28, 28, ...]로 업샘플링
  cnn.add(tf.layers.conv2dTranspose({
    filters: 1,
    kernelSize: 5,
    strides: 2,
    padding: 'same',
    activation: 'tanh',
    kernelInitializer: 'glorotNormal'
  }));

  // 대부분의 TensorFlow.js 모델과 달리 ACGAN의 생성자는 두 개의 입력을 받습니다:
  //  1. 가짜 이미지 생성의 시드로 사용할 잠재 벡터
  //  2. 생성될 가짜 이미지의 숫자 클래스를 제어하기 위한 클래스 레이블

  // GAN 논문에서 종종 z-공간이라고 부릅니다.
  const latent = tf.input({shape: [latentSize]});

  // 생성할 이미지의 레이블. [0, NUM_CLASSES] 사이의 정수
  const imageClass = tf.input({shape: [1]});

  // 레이블은 임베딩 룩업을 통해 `latentSize` 길이의 벡터로 변환됩니다.
  const classEmbedding = tf.layers.embedding({
    inputDim: NUM_CLASSES,
    outputDim: latentSize,
    embeddingsInitializer: 'glorotNormal'
  }).apply(imageClass);

  // z-공간과 클래스 조건 임베딩 사이의 아다마르 곱(Hadamard product)
  const h = tf.layers.multiply().apply([latent, classEmbedding]);

  const fakeImage = cnn.apply(h);
  return tf.model({inputs: [latent, imageClass], outputs: fakeImage});
}

/**
 * ACGAN의 판별자를 만듭니다.
 *
 * ACGAN의 판별자는 [batchSize, 28, 28, 1] 크기의 이미지를 받습니다.
 *
 * 두 개의 출력을 만듭니다:
 *   1. 판별자가 입력 이미지를 진짜(1에 가까움) 혹은 가짜(0에 가까움)로
 *      판단하는지를 나타내는 0~1 사이의 시그모이드 확률 점수.
 *   2. 10개 MNIST 숫자 카테고리에 대한 소프트맥스 확률 점수.
 *      판별자가 입력 이미지를 10개 클래스에 대해 분류한 결과입니다.
 *
 * @returns {tf.LayersModel} 판별자 모델
 */
function buildDiscriminator() {
  const cnn = tf.sequential();

  cnn.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    padding: 'same',    strides: 2,
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1]
  }));
  cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
  cnn.add(tf.layers.dropout({rate: 0.3}));

  cnn.add(tf.layers.conv2d(
      {filters: 64, kernelSize: 3, padding: 'same', strides: 1}));
  cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
  cnn.add(tf.layers.dropout({rate: 0.3}));

  cnn.add(tf.layers.conv2d(
      {filters: 128, kernelSize: 3, padding: 'same', strides: 2}));
  cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
  cnn.add(tf.layers.dropout({rate: 0.3}));

  cnn.add(tf.layers.conv2d(
      {filters: 256, kernelSize: 3, padding: 'same', strides: 1}));
  cnn.add(tf.layers.leakyReLU({alpha: 0.2}));
  cnn.add(tf.layers.dropout({rate: 0.3}));

  cnn.add(tf.layers.flatten());

  const image = tf.input({shape: [IMAGE_SIZE, IMAGE_SIZE, 1]});
  const features = cnn.apply(image);

  // 대부분의 TensorFlow.js 모델과 달리 판별자는 두 개의 출력을 만듭니다.
  // 첫 번째 출력은 입력 샘플이 (생성자가 만든 가짜가 아니라) 진짜 MNIST 이미지와 얼마나 비슷한지
  // 판별자가 할당한 확률 점수입니다.
  const realnessScore =
      tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(features);
  // 두 번째 출력은 10개 MNIST 숫자 클래스(0~9)에 대해 판별자가 할당한 소프트맥스 확률입니다.
  // (ACGAN의 이름에 있는) "auxiliary"를 의미하는 "aux"는
  // (단순히 진짜/가짜 이진 분류만 수행하는) 표준 GAN과 달리
  // ACGAN의 판별자는 다중 분류도 수행한다는 사실을 나타냅니다.
  const aux = tf.layers.dense({units: NUM_CLASSES, activation: 'softmax'})
                  .apply(features);

  return tf.model({inputs: image, outputs: [realnessScore, aux]});
}

/**
 * 연결된 ACGAN 모델 만들기.
 *
 * @param {number} latentSize 잠재 벡터 크기
 * @param {tf.SymbolicTensor} imageClass 원하는 이미지 클래스를 위한 심볼릭 텐서
 *   생성자의 입력 중 하나입니다.
 * @param {tf.LayersModel} generator 생성자
 * @param {tf.LayersModel} discriminator 판별자
 * @param {tf.Optimizer} optimizer 연결된 모델을 훈련하기 위해 사용하는 옵티마이저
 * @returns {tf.LayersModel} 컴파일된 ACGAN 모델
 */
function buildCombinedModel(latentSize, generator, discriminator, optimizer) {
  // 잠재 벡터. 생성자의 첫 번째 입력입니다.
  const latent = tf.input({shape: [latentSize]});
  // 원하는 이미지 클래스. 생성자의 두 번째 입력입니다.
  const imageClass = tf.input({shape: [1]});
  // 생성자가 만든 가짜 이미지의 심볼릭 텐서를 얻습니다.
  let fake = generator.apply([latent, imageClass]);
  let aux;

  // 연결된 모델에서는 생성자만 훈련합니다.
  discriminator.trainable = false;
  [fake, aux] = discriminator.apply(fake);
  const combined =
      tf.model({inputs: [latent, imageClass], outputs: [fake, aux]});
  combined.compile({
    optimizer,
    loss: ['binaryCrossentropy', 'sparseCategoricalCrossentropy']
  });
  combined.summary();
  return combined;
}

// 연결한 ACGAN 모델을 훈련하는데 사용할 "소프트" 1.
// GAN 훈련 트릭 중 하나입니다.
const SOFT_ONE = 0.95;

/**
 * 판별자를 한 스텝 훈련합니다.
 *
 * 이 단계에서는 판별자의 가중치만 업데이트됩니다. 생성자는 훈련되지 않습니다.
 *
 * 다음 단계로 진행합니다:
 *   - 진짜 데이터의 배치를 준비합니다.
 *   - 랜덤한 잠재 벡터와 레이블 벡터를 생성합니다.
 *   - 랜덤한 잠재 벡터와 레이블 벡터를 생성자에게 주입하고 생성된 (즉 가짜) 이미지 배치를 만듭니다.
 *   - 진짜 데이터와 가짜 데이터를 연결합니다; 연결된 데이터에서 판별자를 한 스텝 훈련합니다.
 *   - 손실을 계산하여 반환합니다.
 *
 * @param {tf.Tensor} xTrain 모든 훈련 샘플의 특성을 담은 텐서
 * @param {tf.Tensor} yTrain 모든 훈련 샘플의 레이블을 담은 텐서
 * @param {number} batchStart 배치 시작 인덱스
 * @param {number} batchSize `xTrain`과 `yTrain`에서 뽑을 배치 크기
 * @param {number} latentSize 잠재 공간(z-공간) 크기
 * @param {tf.LayersModel} generator ACGAN의 생성자
 * @param {tf.LayersModel} discriminator ACGAN의 판별자
 * @returns {number[]} 한 스텝 훈련으로 계산한 손실 값
 */
async function trainDiscriminatorOneStep(
    xTrain, yTrain, batchStart, batchSize, latentSize, generator,
    discriminator) {
  const [x, y, auxY] = tf.tidy(() => {
    const imageBatch = xTrain.slice(batchStart, batchSize);
    const labelBatch = yTrain.slice(batchStart, batchSize).asType('float32');

    // 잠재 벡터
    let zVectors = tf.randomUniform([batchSize, latentSize], -1, 1);
    let sampledLabels =
        tf.randomUniform([batchSize, 1], 0, NUM_CLASSES, 'int32')
            .asType('float32');

    const generatedImages =
        generator.predict([zVectors, sampledLabels], {batchSize: batchSize});

    const x = tf.concat([imageBatch, generatedImages], 0);

    const y = tf.tidy(
        () => tf.concat(
            [tf.ones([batchSize, 1]).mul(SOFT_ONE), tf.zeros([batchSize, 1])]));

    const auxY = tf.concat([labelBatch, sampledLabels], 0);
    return [x, y, auxY];
  });

  const losses = await discriminator.trainOnBatch(x, [y, auxY]);
  tf.dispose([x, y, auxY]);
  return losses;
}

/**
 * 연결한 ACGAN을 한 스텝 훈련합니다.
 *
 * 이 단계에서는 생성자의 가중치만 업데이트됩니다.
 *
 * @param {number} batchSize 생성할 가짜 이미지 배치 크기
 * @param {number} latentSize 잠재 공간 (z-공간) 크기
 * @param {tf.LayersModel} combined 생성자와 판별자를 연결한 tf.LayersModel 객체
 * @returns {number[]} 연결된 모델의 손실 값
 */
async function trainCombinedModelOneStep(batchSize, latentSize, combined) {
  const [noise, sampledLabels, trick] = tf.tidy(() => {
    // 새로운 잠재 벡터 만들기
    const zVectors = tf.randomUniform([batchSize, latentSize], -1, 1);
    const sampledLabels =
        tf.randomUniform([batchSize, 1], 0, NUM_CLASSES, 'int32')
            .asType('float32');

    // 판별자를 속이기 위해 생성자를 훈련합니다.
    // 가짜와 진짜 레이블을 모두 진짜로 나타냅니다.
    const trick = tf.tidy(() => tf.ones([batchSize, 1]).mul(SOFT_ONE));
    return [zVectors, sampledLabels, trick];
  });

  const losses = await combined.trainOnBatch(
      [noise, sampledLabels], [trick, sampledLabels]);
  tf.dispose([noise, sampledLabels, trick]);
  return losses;
}

function parseArguments() {
  const parser = new argparse.ArgumentParser({
    description: 'TensorFlowj.js: MNIST ACGAN 훈련 예제',
    addHelp: true
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'tfjs-node-gpu를 사용해 훈련합니다(CUDA GPU 필요)'
  });
  parser.addArgument(
      '--epochs',
      {type: 'int', defaultValue: 100, help: '훈련 에포크 횟수'});
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 100,
    help: '훈련에 사요할 배치 크기'
  });
  parser.addArgument('--latentSize', {
    type: 'int',
    defaultValue: 100,
    help: '잠재 공간 (z-공간) 크기'
  });
  parser.addArgument(
      '--learningRate',
      {type: 'float', defaultValue: 0.0002, help: '학습률'});
  parser.addArgument('--adamBeta1', {
    type: 'float',
    defaultValue: 0.5,
    help: 'ADAM 옵티마이저의 Beta1 파라미터'
  });
  parser.addArgument('--generatorSavePath', {
    type: 'string',
    defaultValue: './dist/generator',
    help: '에포크가 끝날 때마다 생성자 모델을 저장할 경로'
  });
  parser.addArgument('--logDir', {
    type: 'string',
    help: '손실 값을 저장할 로그 디렉토리'
  });
  return parser.parseArgs();
}

function makeMetadata(totalEpochs, currentEpoch, completed) {
  return {
    totalEpochs,
    currentEpoch,
    completed,
    lastUpdated: new Date().getTime()
  }
}

async function run() {
  const args = parseArguments();
  if (args.gpu) {
    console.log('GPU 사용');
    tf = require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('CPU 사용');
    tf = require('@tensorflow/tfjs-node');
  }

  if (!fs.existsSync(path.dirname(args.generatorSavePath))) {
    fs.mkdirSync(path.dirname(args.generatorSavePath));
  }
  const saveURL = `file://${args.generatorSavePath}`;
  const metadataPath = path.join(args.generatorSavePath, 'acgan-metadata.json');

  // 판별자를 만듭니다.
  const discriminator = buildDiscriminator();
  discriminator.compile({
    optimizer: tf.train.adam(args.learningRate, args.adamBeta1),
    loss: ['binaryCrossentropy', 'sparseCategoricalCrossentropy']
  });
  discriminator.summary();

  // 생성자를 만듭니다.
  const generator = buildGenerator(args.latentSize);
  generator.summary();

  const optimizer = tf.train.adam(args.learningRate, args.adamBeta1);
  const combined = buildCombinedModel(
      args.latentSize, generator, discriminator, optimizer);

  await data.loadData();
  let {images: xTrain, labels: yTrain} = data.getTrainData();
  yTrain = tf.expandDims(yTrain.argMax(-1), -1);

  // 훈련을 시작하기 전에 생성자 모델을 저장합니다.
  await generator.save(saveURL);

  let numTensors;
  let logWriter;
  if (args.logDir) {
    console.log(`텐서보드에 로그를 기록합니다: ${args.logDir}`);
    logWriter = tf.node.summaryFileWriter(args.logDir);
  }

  let step = 0;
  for (let epoch = 0; epoch < args.epochs; ++epoch) {
    // 에포크를 시작할 때마다 메타데이터를 디스크에 저장합니다.
    fs.writeFileSync(
        metadataPath,
        JSON.stringify(makeMetadata(args.epochs, epoch, false)));

    const tBatchBegin = tf.util.now();

    const numBatches = Math.ceil(xTrain.shape[0] / args.batchSize);

    for (let batch = 0; batch < numBatches; ++batch) {
      const actualBatchSize = (batch + 1) * args.batchSize >= xTrain.shape[0] ?
          (xTrain.shape[0] - batch * args.batchSize) :
          args.batchSize;

      const dLoss = await trainDiscriminatorOneStep(
          xTrain, yTrain, batch * args.batchSize, actualBatchSize,
          args.latentSize, generator, discriminator);

      // 생성자 옵티마이저가 판별자와 동일한 이미지 개수를 처리하도록
      // 2 * actualBatchSize 크기를 사용합니다.
      const gLoss = await trainCombinedModelOneStep(
          2 * actualBatchSize, args.latentSize, combined);

      console.log(
          `epoch ${epoch + 1}/${args.epochs} batch ${batch + 1}/${
              numBatches}: ` +
          `dLoss = ${dLoss[0].toFixed(6)}, gLoss = ${gLoss[0].toFixed(6)}`);
      if (logWriter != null) {
        logWriter.scalar('dLoss', dLoss[0], step);
        logWriter.scalar('gLoss', gLoss[0], step);
        step++;
      }

      // 메모리 누수 확인
      if (numTensors == null) {
        numTensors = tf.memory().numTensors;
      } else {
        tf.util.assert(
            tf.memory().numTensors === numTensors,
            `Leaked ${tf.memory().numTensors - numTensors} tensors`);
      }
    }

    await generator.save(saveURL);
    console.log(
        `에포크 ${epoch + 1}의 소요 시간: ` +
        `${((tf.util.now() - tBatchBegin) / 1e3).toFixed(1)} s`);
    console.log(`생성자 저장: ${saveURL}\n`);
  }

  // 훈련이 끝났음을 표시하기 위해 메타데이터를 디스크에 기록합니다.
  fs.writeFileSync(
      metadataPath,
      JSON.stringify(makeMetadata(args.epochs, args.epochs, true)));
}

if (require.main === module) {
  run();
}

module.exports = {
  buildCombinedModel,
  buildDiscriminator,
  buildGenerator,
  trainCombinedModelOneStep,
  trainDiscriminatorOneStep
};
