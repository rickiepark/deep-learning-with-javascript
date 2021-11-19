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
 * 여러가지 날짜 포맷을 ISO 날짜 포맷으로 바꾸기 위해 어텐션 LSTM 시퀀스-투-시퀀스 디코더를 훈련합니다.
 *
 * 다음 예를 참고했습니다.
 * https://github.com/wanasit/katakana/blob/master/notebooks/Attention-based%20Sequence-to-Sequence%20in%20Keras.ipynb
 */

import * as fs from 'fs';
import * as shelljs from 'shelljs';
import * as argparse from 'argparse';
import * as tf from '@tensorflow/tfjs';
import * as dateFormat from './date_format';
import {createModel, runSeq2SeqInference} from './model';

/**
 * 훈련용 날짜 데이터 생성
 *
 * @param {number} trainSplit 훈련 세트 비율. 0보다 크고 1보다 작아야 합니다.
 * @param {number} valSplit 검증 세트 비율. 0보다 크고 1보다 작아야 합니다.
 * @return 다음으로 구성된 `Object`
 *   - trainEncoderInput, `[numTrainExapmles, inputLength]` 크기의  `tf.Tensor`
 *   - trainDecoderInput, `[numTrainExapmles, outputLength]` 크기의 `tf.Tensor`.
 *     모든 샘플의 첫 번째 원소는 START_CODE(시작 토큰)로 지정됩니다.
 *   - trainDecoderOuptut, `[numTrainExamples, outputLength, outputVocabSize]` 크기의
 *     원-핫 인코딩된 `tf.Tensor`
 *   - valEncoderInput, trainEncoderInput와 동일하지만 검증 세트용
 *   - valDecoderInput, trainDecoderInput와 동일하지만 검증 세트용
 *   - valDecoderOutput, trainDecoderOuptut와 동일하지만 검증 세트용
 *   - testDateTuples, 테스트 세트 날짜 튜플 ([year, month, day])
 */
export function generateDataForTraining(trainSplit = 0.25, valSplit = 0.15) {
  tf.util.assert(
      trainSplit > 0 && valSplit > 0 && trainSplit + valSplit <= 1,
      `잘못된 trainSplit (${trainSplit})와 valSplit (${valSplit})`);

  const dateTuples = [];
  const MIN_YEAR = 1950;
  const MAX_YEAR = 2050;
  for (let date = new Date(MIN_YEAR,0,1);
       date.getFullYear() < MAX_YEAR;
       date.setDate(date.getDate() + 1)) {
    dateTuples.push([date.getFullYear(), date.getMonth() + 1, date.getDate()]);
  }
  tf.util.shuffle(dateTuples);

  const numTrain = Math.floor(dateTuples.length * trainSplit);
  const numVal = Math.floor(dateTuples.length * valSplit);
  console.log(`훈련에 사용할 날짜 개수: ${numTrain}`);
  console.log(`검증에 사용할 날짜 개수: ${numVal}`);
  console.log(
      `테스트에 사용할 날짜 개수: ` +
      `${dateTuples.length - numTrain - numVal}`);

  function dateTuplesToTensor(dateTuples) {
    return tf.tidy(() => {
      const inputs =
          dateFormat.INPUT_FNS.map(fn => dateTuples.map(tuple => fn(tuple)));
      const inputStrings = [];
      inputs.forEach(inputs => inputStrings.push(...inputs));
      const encoderInput =
          dateFormat.encodeInputDateStrings(inputStrings);
      const trainTargetStrings = dateTuples.map(
          tuple => dateFormat.dateTupleToYYYYDashMMDashDD(tuple));
      let decoderInput =
          dateFormat.encodeOutputDateStrings(trainTargetStrings)
          .asType('float32');
      // 한 스텝 시간 이동: 디코더 입력은 인코더 입력에 대비하여 한 타임 스텝 왼쪽으로 이동합니다.
      // 이는 추론시에 일어나는 스텝 단위 디코딩을 해결합니다.
      decoderInput = tf.concat([
        tf.ones([decoderInput.shape[0], 1]).mul(dateFormat.START_CODE),
        decoderInput.slice(
            [0, 0], [decoderInput.shape[0], decoderInput.shape[1] - 1])
      ], 1).tile([dateFormat.INPUT_FNS.length, 1]);
      const decoderOutput = tf.oneHot(
          dateFormat.encodeOutputDateStrings(trainTargetStrings),
          dateFormat.OUTPUT_VOCAB.length).tile(
              [dateFormat.INPUT_FNS.length, 1, 1]);
      return {encoderInput, decoderInput, decoderOutput};
    });
  }

  const {
    encoderInput: trainEncoderInput,
    decoderInput: trainDecoderInput,
    decoderOutput: trainDecoderOutput
  } = dateTuplesToTensor(dateTuples.slice(0, numTrain));
  const {
    encoderInput: valEncoderInput,
    decoderInput: valDecoderInput,
    decoderOutput: valDecoderOutput
  } = dateTuplesToTensor(dateTuples.slice(numTrain, numTrain + numVal));
  const testDateTuples =
      dateTuples.slice(numTrain + numVal, dateTuples.length);
  return {
    trainEncoderInput,
    trainDecoderInput,
    trainDecoderOutput,
    valEncoderInput,
    valDecoderInput,
    valDecoderOutput,
    testDateTuples
  };
}

function parseArguments() {
  const argParser = new argparse.ArgumentParser({
    description:
        'TensorFlow.js로 어텐션 기반 날짜 변환 모델을 훈련합니다.'
  });
  argParser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'tfjs-node-gpu를 사용해 모델을 훈련합니다. CUDA/CuDNN 필요.'
  });
  argParser.addArgument('--epochs', {
    type: 'int',
    defaultValue: 2,
    help: '모델을 훈련할 에포크 횟수'
  });
  argParser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 128,
    help: '모델 훈련에 사용할 배치 크기'
  });
  argParser.addArgument('--trainSplit ', {
    type: 'float',
    defaultValue: 0.25,
    help: '훈련에 사용할 날짜 비율. ' +
    '0보다 크고 1보다 작아야 합니다. valSplit와 더해서 1보다 작아야 합니다.'
  });
  argParser.addArgument('--valSplit', {
    type: 'float',
    defaultValue: 0.15,
    help: '검증에 사용할 날짜 비율. ' +
    '0보다 크고 1보다 작아야 합니다. trainSplit와 더해서 1보다 작아야 합니다.'
  });
  argParser.addArgument('--savePath', {
    type: 'string',
    defaultValue: './dist/model',
  });
  argParser.addArgument('--logDir', {
    type: 'string',
    help: '훈련하는 동안 손실과 정확도를 기록할 텐서보드 로그 디렉토리.'
  });
  argParser.addArgument('--logUpdateFreq', {
    type: 'string',
    defaultValue: 'batch',
    optionStrings: ['batch', 'epoch'],
    help: '텐서보드에 손실과 정확도를 기록할 빈도'
  });
  return argParser.parseArgs();
}

async function run() {
  const args = parseArguments();
  let tfn;
  if (args.gpu) {
    console.log('Using GPU');
    tfn = require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU');
    tfn = require('@tensorflow/tfjs-node');
  }

  const model = createModel(
      dateFormat.INPUT_VOCAB.length, dateFormat.OUTPUT_VOCAB.length,
      dateFormat.INPUT_LENGTH, dateFormat.OUTPUT_LENGTH);
  model.summary();

  const {
    trainEncoderInput,
    trainDecoderInput,
    trainDecoderOutput,
    valEncoderInput,
    valDecoderInput,
    valDecoderOutput,
    testDateTuples
  } = generateDataForTraining(args.trainSplit, args.valSplit);

  await model.fit(
      [trainEncoderInput, trainDecoderInput], trainDecoderOutput, {
        epochs: args.epochs,
        batchSize: args.batchSize,
        shuffle: true,
        validationData: [[valEncoderInput, valDecoderInput], valDecoderOutput],
        callbacks: args.logDir == null ? null :
            tfn.node.tensorBoard(args.logDir, {updateFreq: args.logUpdateFreq})
      });

  // 모델 저장
  if (args.savePath != null && args.savePath.length) {
    if (!fs.existsSync(args.savePath)) {
      shelljs.mkdir('-p', args.savePath);
    }
    const saveURL = `file://${args.savePath}`
    await model.save(saveURL);
    console.log(`모데 저장: ${saveURL}`);
  }

  // seq2seq 추론을 실행하고 결과를 콘솔에 출력합니다.
  const numTests = 10;
  for (let n = 0; n < numTests; ++n) {
    for (const testInputFn of dateFormat.INPUT_FNS) {
      const inputStr = testInputFn(testDateTuples[n]);
      console.log('\n-----------------------');
      console.log(`입력 문자열: ${inputStr}`);
      const correctAnswer =
          dateFormat.dateTupleToYYYYDashMMDashDD(testDateTuples[n]);
      console.log(`정확한 답: ${correctAnswer}`);

      const {outputStr} = await runSeq2SeqInference(model, inputStr);
      const isCorrect = outputStr === correctAnswer;
      console.log(
          `모델 출력: ${outputStr} (${isCorrect ? 'OK' : 'WRONG'})` );
    }
  }
}

if (require.main === module) {
  run();
}
