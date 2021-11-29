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

import * as fs from 'fs';

import * as argparse from 'argparse';
import {mkdir} from 'shelljs';

// (TensorFlow.js Node 모듈인) tf 값은 --gpu 플래그에 따라 동적으로 설정됩니다.
let tf;

import {SnakeGameAgent} from './agent';
import {copyWeights} from './dqn';
import {SnakeGame} from './snake_game';

class MovingAverager {
  constructor(bufferLength) {
    this.buffer = [];
    for (let i = 0; i < bufferLength; ++i) {
      this.buffer.push(null);
    }
  }

  append(x) {
    this.buffer.shift();
    this.buffer.push(x);
  }

  average() {
    return this.buffer.reduce((x, prev) => x + prev) / this.buffer.length;
  }
}

/**
 * 스네이크 게임을 플레이하기 위한 에이전트를 훈련합니다.
 *
 * @param {SnakeGameAgent} agent 훈련할 에이전트.
 * @param {number} batchSize 훈련을 위한 배치 크기.
 * @param {number} gamma 보상 할인 계수. 0보다 크거나 같고 1보다 작거나 같아야 합니다.
 * @param {number} learnigRate 학습률
 * @param {number} cumulativeRewardThreshold 한 게임에서 얻는 누적 보상의 이동 평균 임곗값.
 *   임곗값에 도달하면 바로 훈련이 중지됩니다.
 * @param {number} maxNumFrames 훈련할 최대 프레임 수
 * @param {number} syncEveryFrames 에이전트의 온라인 DQN에서 타깃 DQN으로 가중치를 복사하는 빈도(프레임 수)
 * @param {string} savePath 훈련이 끝난 후 에이전트의 온라인 DQN을 저장할 경로
 * @param {string} logDir 훈련하는 동안 텐서보드 로그를 기록할 디렉토리
 */
export async function train(
    agent, batchSize, gamma, learningRate, cumulativeRewardThreshold,
    maxNumFrames, syncEveryFrames, savePath, logDir) {
  let summaryWriter;
  if (logDir != null) {
    summaryWriter = tf.node.summaryFileWriter(logDir);
  }

  for (let i = 0; i < agent.replayBufferSize; ++i) {
    agent.playStep();
  }

  // 이동 평균: 가장 최근의 100개 에피소드에 걸쳐 보상을 누적합니다.
  const rewardAverager100 = new MovingAverager(100);
  // 이동 평균: 가장 최근의 100개 에피소드에 걸쳐 먹은 과일을 누적합니다.
  const eatenAverager100 = new MovingAverager(100);

  const optimizer = tf.train.adam(learningRate);
  let tPrev = new Date().getTime();
  let frameCountPrev = agent.frameCount;
  let averageReward100Best = -Infinity;
  while (true) {
    agent.trainOnReplayBatch(batchSize, gamma, optimizer);
    const {cumulativeReward, done, fruitsEaten} = agent.playStep();
    if (done) {
      const t = new Date().getTime();
      const framesPerSecond =
          (agent.frameCount - frameCountPrev) / (t - tPrev) * 1e3;
      tPrev = t;
      frameCountPrev = agent.frameCount;

      rewardAverager100.append(cumulativeReward);
      eatenAverager100.append(fruitsEaten);
      const averageReward100 = rewardAverager100.average();
      const averageEaten100 = eatenAverager100.average();

      console.log(
          `Frame #${agent.frameCount}: ` +
          `cumulativeReward100=${averageReward100.toFixed(1)}; ` +
          `eaten100=${averageEaten100.toFixed(2)} ` +
          `(epsilon=${agent.epsilon.toFixed(3)}) ` +
          `(${framesPerSecond.toFixed(1)} frames/s)`);
      if (summaryWriter != null) {
        summaryWriter.scalar(
            'cumulativeReward100', averageReward100, agent.frameCount);
        summaryWriter.scalar('eaten100', averageEaten100, agent.frameCount);
        summaryWriter.scalar('epsilon', agent.epsilon, agent.frameCount);
        summaryWriter.scalar(
            'framesPerSecond', framesPerSecond, agent.frameCount);
      }
      if (averageReward100 >= cumulativeRewardThreshold ||
          agent.frameCount >= maxNumFrames) {
        break;
      }
      if (averageReward100 > averageReward100Best) {
        averageReward100Best = averageReward100;
        if (savePath != null) {
          if (!fs.existsSync(savePath)) {
            mkdir('-p', savePath);
          }
          await agent.onlineNetwork.save(`file://${savePath}`);
          console.log(`${savePath}에 DQN을 저장했습니다.`);
        }
      }
    }
    if (agent.frameCount % syncEveryFrames === 0) {
      copyWeights(agent.targetNetwork, agent.onlineNetwork);
      console.log('온라인 네트워크에서 타깃 네트워크로 가중치를 동기화했습니다');
    }
  }
}

export function parseArguments() {
  const parser = new argparse.ArgumentParser({
    description: '스네이크 게임을 플레이할 DQN을 훈련하는 스크립트'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: '훈련에 tfjs-node-gpu을 사용합니다' +
    '(CUDA GPU, 드라이버, 라이브러리가 필요합니다).'
  });
  parser.addArgument('--height', {
    type: 'int',
    defaultValue: 9,
    help: '게임 보드의 높이'
  });
  parser.addArgument('--width', {
    type: 'int',
    defaultValue: 9,
    help: '게임 보드의 너비'
  });
  parser.addArgument('--numFruits', {
    type: 'int',
    defaultValue: 1,
    help: '어떤 순간에 보드에 나타날 과일 개수'
  });
  parser.addArgument('--initLen', {
    type: 'int',
    defaultValue: 2,
    help: '초기 스네이크 길이(사각형 개수)'
  });
  parser.addArgument('--cumulativeRewardThreshold', {
    type: 'float',
    defaultValue: 100,
    help: '최근 100개 게임에 대한 누적 보상의 임곗값(이동 평균). ' +
    '임곗값에 도달하면 (또는 --maxNumFrames에 도달하면) 바로 훈련이 중지됩니다.'
  });
  parser.addArgument('--maxNumFrames', {
    type: 'float',
    defaultValue: 1e6,
    help: '훈련하는 동안 실행할 수 있는 최대 프레임 수. ' +
    '이 프레임 수에 도달하면 바로 훈련이 중지됩니다.'
  });
  parser.addArgument('--replayBufferSize', {
    type: 'int',
    defaultValue: 1e4,
    help: '재생 메모리 버퍼의 길이.'
  });
  parser.addArgument('--epsilonInit', {
    type: 'float',
    defaultValue: 0.5,
    help: '입실론 그리디 알고리즘에서 사용할 초기 입실론 값.'
  });
  parser.addArgument('--epsilonFinal', {
    type: 'float',
    defaultValue: 0.01,
    help: '입실론 그리디 알고리즘에서 사용할 마지막 입실론 값.'
  });
  parser.addArgument('--epsilonDecayFrames', {
    type: 'int',
    defaultValue: 1e5,
    help: 'epsilonInit에서 epsilonFinal까지 입실론 값을 감소시키기 위한 프레임 수'
  });
  parser.addArgument('--batchSize', {
    type: 'int',
    defaultValue: 64,
    help: 'DQN 훈련의 배치 크기'
  });
  parser.addArgument('--gamma', {
    type: 'float',
    defaultValue: 0.99,
    help: '보상 할인 계수'
  });
  parser.addArgument('--learningRate', {
    type: 'float',
    defaultValue: 1e-3,
    help: 'DQN 훈련의 학습률'
  });
  parser.addArgument('--syncEveryFrames', {
    type: 'int',
    defaultValue: 1e3,
    help: '온라인 네트워크에서 타깃 네트워크로 가중치를 복사할 주기'
  });
  parser.addArgument('--savePath', {
    type: 'string',
    defaultValue: './models/dqn',
    help: '훈련이 끝난 후 온라인 DQN을 저장할 파일 경로.'
  });
  parser.addArgument('--logDir', {
    type: 'string',
    defaultValue: null,
    help: '텐서보드 로그를 기록할 디렉토리 경로'
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArguments();
  if (args.gpu) {
    tf = require('@tensorflow/tfjs-node-gpu');
  } else {
    tf = require('@tensorflow/tfjs-node');
  }
  console.log(`args: ${JSON.stringify(args, null, 2)}`);

  const game = new SnakeGame({
    height: args.height,
    width: args.width,
    numFruits: args.numFruits,
    initLen: args.initLen
  });
  const agent = new SnakeGameAgent(game, {
    replayBufferSize: args.replayBufferSize,
    epsilonInit: args.epsilonInit,
    epsilonFinal: args.epsilonFinal,
    epsilonDecayFrames: args.epsilonDecayFrames,
    learningRate: args.learningRate
  });

  await train(
      agent, args.batchSize, args.gamma, args.learningRate,
      args.cumulativeRewardThreshold, args.maxNumFrames,
      args.syncEveryFrames, args.savePath, args.logDir);
}

if (require.main === module) {
  main();
}
