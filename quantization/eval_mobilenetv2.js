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

import * as argparse from 'argparse';
import * as fs from 'fs';
import * as jimp from 'jimp';
import * as path from 'path';
const ProgressBar = require('progress');

import {IMAGENET_CLASSES} from './imagenet_classes';

// `--gpu` 플래그 지정에 따라 동적으로 tf를 임포트합니다.
let tf;

function parseArgs() {
  const parser = new argparse.ArgumentParser({
    description:
        'TensorFlow.js 양자화 예제: MobileNetV2 모델 평가하기',
    addHelp: true
  });
  parser.addArgument('modelSavePath', {
    type: 'string',
    help: '평가할 모델을 저장할 경로'
  });
  parser.addArgument('imageDir', {
    type: 'string',
    help: '테스트 이미지가 저장된 경로'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'tfjs-node-gpu를 사용해 평가합니다(CUDA 가능 GPU, 지원 드라이버와 라이브러리가 필요).'
  });
  return parser.parseArgs();
}

async function readImageTensorFromFile(filePath, height, width) {
  return new Promise((resolve, reject) => {
    jimp.read(filePath, (err, image) => {
      if (err) {
        reject(err);
      } else {
        const h = image.bitmap.height;
        const w = image.bitmap.width;
        const buffer = tf.buffer([1, h, w, 3], 'float32');
        image.scan(0, 0, w, h, function(x, y, index) {
        buffer.set(image.bitmap.data[index], 0, y, x, 0);
        buffer.set(image.bitmap.data[index + 1], 0, y, x, 1);
        buffer.set(image.bitmap.data[index + 2], 0, y, x, 2);
      });
      resolve(tf.tidy(() => tf.image.resizeBilinear(
          buffer.toTensor(), [height, width]).div(255)));
      }
    });
  });
}

async function main() {
  const args = parseArgs();
  if (args.gpu) {
    tf = require('@tensorflow/tfjs-node-gpu');
  } else {
    tf = require('@tensorflow/tfjs-node');
  }

  console.log(`${args.modelSavePath}에서 모델 로딩 중...`);
  const model = await tf.loadLayersModel(`file://${args.modelSavePath}`);

  const imageH = model.inputs[0].shape[2];
  const imageW = model.inputs[0].shape[2];

  // 이미지를 텐서로 로딩합니다.
  const dirContent = fs.readdirSync(args.imageDir);
  dirContent.sort();
  const numImages = dirContent.length;
  console.log(`${numImages} 개의 이미지 읽는 중...`);
  const progressBar = new ProgressBar('[:bar]', {
    total: numImages,
    width: 80,
    head: '>'
  });
  const imageTensors = [];
  const truthLabels = [];
  for (const fileName of dirContent) {
    const truthLabel = fileName.split('.')[0].split('_')[2];
    truthLabels.push(truthLabel);
    const imageFilePath = path.join(args.imageDir, fileName);
    const imageTensor =
        await readImageTensorFromFile(imageFilePath, imageH, imageW);
    imageTensors.push(imageTensor);
    progressBar.tick();
  }

  const stackedImageTensor = tf.concat(imageTensors, 0);
  console.log('model.predict() 호출 중...');
  const t0 = new Date().getTime();
  const {top1Indices, top5Indices} = tf.tidy(() => {
    const probs = model.predict(stackedImageTensor, {batchSize: 64});
    return {
      top1Indices: probs.argMax(-1).arraySync(),
      top5Indices: probs.topk(5).indices.arraySync()
    };
  });
  console.log(`model.predict() 걸린 시간: ${(new Date().getTime() - t0).toFixed(2)} ms`);

  let numCorrectTop1 = 0;
  let numCorrectTop5 = 0;
  top1Indices.forEach((top1Index, i) => {
    const truthLabel = truthLabels[i];
    if (IMAGENET_CLASSES[top1Index].indexOf(truthLabel) !== -1) {
      numCorrectTop1++;
    }
    for (let k = 0; k < 5; ++k) {
      if (IMAGENET_CLASSES[top5Indices[i][k]].indexOf(truthLabel) !== -1) {
        numCorrectTop5++;
        break;
      }
    }
  });
  console.log(
      `총 개수 = ${numImages}; top-1 정답 개수 = ${numCorrectTop1}; ` +
      `top-1 정확도 = ${(numCorrectTop1 / numImages).toFixed(3)}; ` +
      `top-5 정답 개수 = ${numCorrectTop5}; ` +
      `top-5 정확도 = ${(numCorrectTop5 / numImages).toFixed(3)}\n`);
  tf.dispose([imageTensors, stackedImageTensor]);
}

if (require.main === module) {
  main();
}
