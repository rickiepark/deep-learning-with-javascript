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

import {WebsitePhishingDataset} from './data.js';
import * as ui from './ui.js';
import * as utils from './utils.js';

function falsePositives(yTrue, yPred) {
  return tf.tidy(() => {
    const one = tf.scalar(1);
    const zero = tf.scalar(0);
    return tf.logicalAnd(yTrue.equal(zero), yPred.equal(one))
        .sum()
        .cast('float32');
  });
}

function trueNegatives(yTrue, yPred) {
  return tf.tidy(() => {
    const zero = tf.scalar(0);
    return tf.logicalAnd(yTrue.equal(zero), yPred.equal(zero))
        .sum()
        .cast('float32');
  });
}

function falsePositiveRate(yTrue, yPred) {
  return tf.tidy(() => {
    const fp = falsePositives(yTrue, yPred);
    const tn = trueNegatives(yTrue, yPred);
    return fp.div(fp.add(tn));
  });
}

/**
 * ROC 곡선을 그립니다.
 *
 * @param {tf.Tensor} targets 정답 타깃 레이블로 0과 1로만 구성된 1D 텐서 객체
 * @param {tf.Tensor} probs 모델이 출력한 확률로 `targets`와 동일한 크기의 1D 텐서
 *   이 값은 0보다 크거나 같고 1보다 작거나 같아야 합니다.
 * @param {number} epoch `probs` 값의 에포크 횟수
 * @returns {number} AUC
 */
function drawROC(targets, probs, epoch) {
  return tf.tidy(() => {
    const thresholds = [
      0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,  0.45, 0.5,  0.55,
      0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0
    ];
    const tprs = [];  // 진짜 양성 비율
    const fprs = [];  // 거짓 양성 비율
    let area = 0;
    for (let i = 0; i < thresholds.length; ++i) {
      const threshold = thresholds[i];

      const threshPredictions = utils.binarize(probs, threshold).as1D();
      const fpr = falsePositiveRate(targets, threshPredictions).dataSync()[0];
      const tpr = tf.metrics.recall(targets, threshPredictions).dataSync()[0];
      fprs.push(fpr);
      tprs.push(tpr);

      // AUC 계산을 위해 면적을 누적합니다.
      if (i > 0) {
        area += (tprs[i] + tprs[i - 1]) * (fprs[i - 1] - fprs[i]) / 2;
      }
    }
    ui.plotROC(fprs, tprs, epoch);
    return area;
  });
}

// 모델 훈련을 위한 하이퍼파라미터
const epochs = 400;
const batchSize = 350;

const data = new WebsitePhishingDataset();
data.loadData().then(async () => {
  await ui.updateStatus('훈련 데이터와 테스트 데이터를 준비합니다...');
  const trainData = data.getTrainData();
  const testData = data.getTestData();

  await ui.updateStatus('모델을 만듭니다...');
  const model = tf.sequential();
  model.add(tf.layers.dense(
      {inputShape: [data.numFeatures], units: 100, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 100, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  model.compile(
      {optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy']});

  const trainLogs = [];
  let auc;

  await ui.updateStatus('훈련을 시작합니다...');
  await model.fit(trainData.data, trainData.target, {
    batchSize: batchSize,
    epochs: epochs,
    validationSplit: 0.2,
    callbacks: {
      onEpochBegin: async (epoch) => {
        // 몇 번의 에포크마다 ROC 곡선을 그립니다.
        if ((epoch + 1) % 100 === 0 || epoch === 0 || epoch === 2 ||
            epoch === 4) {
          const probs = model.predict(testData.data);
          auc = drawROC(testData.target, probs, epoch);
        }
      },
      onEpochEnd: async (epoch, logs) => {
        await ui.updateStatus(`총 에포크: ${epochs}, 완료 에포크: ${epoch + 1}`);
        trainLogs.push(logs);
        ui.plotLosses(trainLogs);
        ui.plotAccuracies(trainLogs);
      }
    }
  });

  await ui.updateStatus('테스트 데이터에서 실행합니다...');
  tf.tidy(() => {
    const result =
        model.evaluate(testData.data, testData.target, {batchSize: batchSize});

    const lastTrainLog = trainLogs[trainLogs.length - 1];
    const testLoss = result[0].dataSync()[0];
    const testAcc = result[1].dataSync()[0];

    const probs = model.predict(testData.data);
    const predictions = utils.binarize(probs).as1D();

    const precision =
        tf.metrics.precision(testData.target, predictions).dataSync()[0];
    const recall =
        tf.metrics.recall(testData.target, predictions).dataSync()[0];
    const fpr = falsePositiveRate(testData.target, predictions).dataSync()[0];
    ui.updateStatus(
        `최종 테스트 세트 손실: ${lastTrainLog.loss.toFixed(4)} 정확도: ${
            lastTrainLog.acc.toFixed(4)}\n` +
        `최종 검증 세트 손실: ${
            lastTrainLog.val_loss.toFixed(
                4)} 정확도: ${lastTrainLog.val_acc.toFixed(4)}\n` +
        `테스트 세트 손실: ${testLoss.toFixed(4)} 정확도: ${
            testAcc.toFixed(4)}\n` +
        `정밀도: ${precision.toFixed(4)}\n` +
        `재현율: ${recall.toFixed(4)}\n` +
        `거짓 양성 비율 (FPR): ${fpr.toFixed(4)}\n` +
        `AUC: ${auc.toFixed(4)}`);
  });
});
