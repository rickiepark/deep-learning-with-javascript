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
 * TensorFlow.js 강화 학습 예제: Cart-Pole 시스템 균형 잡기.
 *
 * 시뮬레이션, 훈련, 테스트, 시각화 부분이 모두 자바스크립트로 작성되었으며
 * WebGL 가속을 사용해 웹 브라우저에서 실행할 수 있습니다.
 *
 * 이 강화 학습 문제는 다음 문헌에서 소개되었습니다:
 *
 * - Barto, Sutton, and Anderson, "Neuronlike Adaptive Elements That Can Solve
 *   Difficult Learning Control Problems," IEEE Trans. Syst., Man, Cybern.,
 *   Vol. SMC-13, pp. 834--846, Sept.--Oct. 1983
 * - Sutton, "Temporal Aspects of Credit Assignment in Reinforcement Learning",
 *   Ph.D. Dissertation, Department of Computer and Information Science,
 *   University of Massachusetts, Amherst, 1984.
 *
 * 나중에 OpenAI의 짐(gym) 환경 중의 하나가 되었습니다:
 *   https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
 */

import {maybeRenderDuringTraining, onGameEnd, setUpUI} from './ui.js';

/**
 * cart-pole 시스템을 제어하기 위한 정책 네트워크
 *
 * 정책 네트워크의 역할은 시스템의 관측된 상태를 기반으로 행동을 선택하는 것입니다.
 * 이 경우 행동은 왼쪽으로 미는 힘이나 오른쪽으로 미는 힘입니다.
 * 관측된 시스템 상태는 4차원 벡터로 카트 위치, 카트 속도, 막대 각도, 막대 각속도로 구성됩니다.
 *
 */
class PolicyNetwork {
  /**
   * PolicyNetwork 생성자
   *
   * @param {number | number[] | tf.LayersModel} hiddenLayerSizes
   *   다음 중 하나가 가능합니다:
   *   - (하나의 은닉층을 위해) 하나의 숫자로 된 은닉 층의 크기
   *   - (여러 개의 은닉층을 위한) 숫자 배열
   *   - tf.LayersModel 객체
   */
  constructor(hiddenLayerSizesOrModel) {
    if (hiddenLayerSizesOrModel instanceof tf.LayersModel) {
      this.policyNet = hiddenLayerSizesOrModel;
    } else {
      this.createPolicyNetwork(hiddenLayerSizesOrModel);
    }
  }

  /**
   * 정책 네트워크의 모델을 만듭니다.
   *
   * @param {number | number[]} hiddenLayerSizes (하나의 은닉층을 위해) 하나의 숫자로 된 은닉 층의 크기
   *   또는 (여러 개의 은닉층을 위한) 숫자 배열
   */
  createPolicyNetwork(hiddenLayerSizes) {
    if (!Array.isArray(hiddenLayerSizes)) {
      hiddenLayerSizes = [hiddenLayerSizes];
    }
    this.policyNet = tf.sequential();
    hiddenLayerSizes.forEach((hiddenLayerSize, i) => {
      this.policyNet.add(tf.layers.dense({
        units: hiddenLayerSize,
        activation: 'elu',
        // `inputShape`은 첫 번째 층에만 필요합니다.
        inputShape: i === 0 ? [4] : undefined
      }));
    });
    // 마지막 층은 하나의 유닛만 가집니다. 이 출력 숫자는 왼쪽으로 미는 행동을 선택하는 확률로 바뀝니다.
    this.policyNet.add(tf.layers.dense({units: 1}));
  }

  /**
   * 정책 네트워크 모델 훈련하기
   *
   * @param {CartPole} cartPoleSystem 훈련에 사용할 cart-pole 시스템 객체
   * @param {tf.train.Optimizer} optimizer 훈련에 사용할 TensorFlow.js 옵티마이저 객체
   * @param {number} discountRate 보상 할인 계수: 0~1 사이의 숫자
   * @param {number} numGames 모델 파라미터를 업데이트하기 위해 플레이할 게임 횟수
   * @param {number} maxStepsPerGame 한 게임에서 수행할 최대 스텝 횟수.
   *   이 숫자에 도달하면 게임은 즉시 종료됩니다.
   * @returns {number[]} `numGames` 게임에서 완료된 스텝 횟수
   */
  async train(
      cartPoleSystem, optimizer, discountRate, numGames, maxStepsPerGame) {
    const allGradients = [];
    const allRewards = [];
    const gameSteps = [];
    onGameEnd(0, numGames);
    for (let i = 0; i < numGames; ++i) {
      // 매 게임 시작마다 cart-pole 시스템의 상태를 랜덤하게 초기화합니다.
      cartPoleSystem.setRandomState();
      const gameRewards = [];
      const gameGradients = [];
      for (let j = 0; j < maxStepsPerGame; ++j) {
        // 게임 스텝 마다 보상으로 이어지는 행동 선택의 확률에 대한 정책 네트워크 가중치의 그레이디언트를 저장합니다.
        const gradients = tf.tidy(() => {
          const inputTensor = cartPoleSystem.getStateTensor();
          return this.getGradientsAndSaveActions(inputTensor).grads;
        });

        this.pushGradients(gameGradients, gradients);
        const action = this.currentActions_[0];
        const isDone = cartPoleSystem.update(action);

        await maybeRenderDuringTraining(cartPoleSystem);

        if (isDone) {
          // 최대 스텝에 도달하기 전에 게임이 종료되면 보상은 0입니다.
          gameRewards.push(0);
          break;
        } else {
          // 게임이 끝나지 않는 한 매 스텝마다 보상 1을 받습니다.
          // 이런 보상 값은 나중에 할인되고 오래 지속된 게임에서 높은 보상 값을 얻습니다.
          gameRewards.push(1);
        }
      }
      onGameEnd(i + 1, numGames);
      gameSteps.push(gameRewards.length);
      this.pushGradients(allGradients, gameGradients);
      allRewards.push(gameRewards);
      await tf.nextFrame();
    }

    tf.tidy(() => {
      // 다음 코드는 세 가지 작업을 합니다:
      // 1. 보상을 할인합니다. 즉, 최근 보상이 과거의 보상보다 우선됩니다. 이로 인해 많은 스텝을 진행한 게임의
      //    보상이 적은 스텝을 진행한 게임의 보상보다 높아지게 됩니다.
      // 2. 보상을 정규화합니다. 즉, 보상의 전체 평균 값을 빼고 전체 표준 편차로 나눕니다. 단계 1과
      //    합치면 오래 지속된 게임의 보상을 양수로 만들고 짧게 진행된 게임의 보상을 음수로 만듭니다.
      // 3. 정규화된 보상 값으로 그레이디언트의 스케일을 조정합니다.
      const normalizedRewards =
          discountAndNormalizeRewards(allRewards, discountRate);
      // 스케일 조정된 그레이디언트를 정책 네트워크의 가중치에 적용합니다. 이 단계는 정책 네트워크가
      // 게임을 오래 진행시킬 행동을 선택할 가능성을 높입니다(즉, 이 강화 학습 알고리즘의 핵심입니다).
      optimizer.applyGradients(
          scaleAndAverageGradients(allGradients, normalizedRewards));
    });
    tf.dispose(allGradients);
    return gameSteps;
  }

  getGradientsAndSaveActions(inputTensor) {
    const f = () => tf.tidy(() => {
      const [logits, actions] = this.getLogitsAndActions(inputTensor);
      this.currentActions_ = actions.dataSync();
      const labels =
          tf.sub(1, tf.tensor2d(this.currentActions_, actions.shape));
      return tf.losses.sigmoidCrossEntropy(labels, logits).asScalar();
    });
    return tf.variableGrads(f);
  }

  getCurrentActions() {
    return this.currentActions_;
  }

  /**
   * 상태 텐서 입력 기반으로 정책 네트워크의 로짓과 행동을 얻습니다.
   *
   * @param {tf.Tensor} inputs `[batchSize, 4]` 크기의 tf.Tensor 객체
   * @returns {[tf.Tensor, tf.Tensor]}
   *   1. `[batchSize, 1]` 크기의 로짓 텐서
   *   2. `[batchSize, 1]` 크기의 행동 텐서
   */
  getLogitsAndActions(inputs) {
    return tf.tidy(() => {
      const logits = this.policyNet.predict(inputs);

      // 왼쪽으로 미는 힘의 확률을 얻습니다.
      const leftProb = tf.sigmoid(logits);
      // 왼쪽과 오른쪽 행동의 확률
      const leftRightProbs = tf.concat([leftProb, tf.sub(1, leftProb)], 1);
      const actions = tf.multinomial(leftRightProbs, 1, null, true);
      return [logits, actions];
    });
  }

  /**
   * 상태-텐서 입력을 기반으로 행동을 선택합니다.
   *
   * @param {tf.Tensor} inputs `[batchSize, 4]` 크기의 tf.Tensor
   * @param {Float32Array} inputs 입력에 대한 행동. `batchSize` 길이
   */
  getActions(inputs) {
    return this.getLogitsAndActions(inputs)[1].dataSync();
  }

  /**
   * 새로운 그레이디언트의 객체를 레코드에 추가합니다.
   *
   * @param {{[varName: string]: tf.Tensor[]}} record 그레이디언트 변수 레코드:
   *   변수 이름을 해당 변수의 그레이디언트 배열에 매핑한 객체
   * @param {{[varName: string]: tf.Tensor}} gradients `record`에 추가할 새로운 그레이디언트:
   *   변수 이름을 그레이디언트 텐서에 매핑한 객체
   */
  pushGradients(record, gradients) {
    for (const key in gradients) {
      if (key in record) {
        record[key].push(gradients[key]);
      } else {
        record[key] = [gradients[key]];
      }
    }
  }
}

// 정책 네트워크 모델이 저장될 IndexedDB 경로
const MODEL_SAVE_PATH_ = 'indexeddb://cart-pole-v1';

/**
 * 모델 저장과 로딩을 지원하기 위한 PolicyNetwork의 서브클래스
 */
export class SaveablePolicyNetwork extends PolicyNetwork {
  /**
   * SaveablePolicyNetwork의 생성자
   *
   * @param {number | number[]} hiddenLayerSizesOrModel
   */
  constructor(hiddenLayerSizesOrModel) {
    super(hiddenLayerSizesOrModel);
  }

  /**
   * IndexedDB에 모델을 저장합니다.
   */
  async saveModel() {
    return await this.policyNet.save(MODEL_SAVE_PATH_);
  }

  /**
   * IndexedDB에서 모델을 로드합니다.
   *
   * @returns {SaveablePolicyNetwork} 로드된 `SaveablePolicyNetwork`의 객체
   * @throws {Error} 만약 IndexedDB에서 모델을 찾을 수 없다면 오류가 발생합니다.
   */
  static async loadModel() {
    const modelsInfo = await tf.io.listModels();
    if (MODEL_SAVE_PATH_ in modelsInfo) {
      console.log(`기존 모델을 로딩 중...`);
      const model = await tf.loadLayersModel(MODEL_SAVE_PATH_);
      console.log(`${MODEL_SAVE_PATH_}에서 모델을 로드했습니다.`);
      return new SaveablePolicyNetwork(model);
    } else {
      throw new Error(`${MODEL_SAVE_PATH_}에서 모델을 찾을 수 없습니다.`);
    }
  }

  /**
   * 로컬에 저장된 모델의 상태를 확인합니다.
   *
   * @returns 로컬에 저장된 모델이 있다면, 이 모델의 정보를 JSON 객체로 반환하고 그렇지 않으면
   *   `undefined`을 반환합니다.
   */
  static async checkStoredModelStatus() {
    const modelsInfo = await tf.io.listModels();
    return modelsInfo[MODEL_SAVE_PATH_];
  }

  /**
   * IndexedDB에서 로컬에 저장된 모델을 삭제합니다.
   */
  async removeModel() {
    return await tf.io.removeModel(MODEL_SAVE_PATH_);
  }

  /**
   * 은닉층의 크기를 가져옵니다.
   *
   * @returns {number | number[]} 모델이 하나의 은닉층을 가지고 있으면
   *   하나의 숫자로 은닉층의 크기를 반환합니다. 모델이 여러 개의 은닉층을 가지고 있으면
   *   숫자 배열로 크기를 반환합니다.
   */
  hiddenLayerSizes() {
    const sizes = [];
    for (let i = 0; i < this.policyNet.layers.length - 1; ++i) {
      sizes.push(this.policyNet.layers[i].units);
    }
    return sizes.length === 1 ? sizes[0] : sizes;
  }
}

/**
 * 보상을 할인합니다.
 *
 * @param {number[]} rewards 할인할 보상 값
 * @param {number} discountRate 할인 계수: 0~1 사이의 숫자, 예를 들면 0.95
 * @returns {tf.Tensor} 할인된 보상 값을 나타내는 1D tf.Tensor
 */
function discountRewards(rewards, discountRate) {
  const discountedBuffer = tf.buffer([rewards.length]);
  let prev = 0;
  for (let i = rewards.length - 1; i >= 0; --i) {
    const current = discountRate * prev + rewards[i];
    discountedBuffer.set(current, i);
    prev = current;
  }
  return discountedBuffer.toTensor();
}

/**
 * 보상 값을 할인하고 정규화합니다.
 *
 * 이 함순느 두 단계를 수행합니다:
 *
 * 1. `discountRate`을 사용해 보상 값을 할인합니다.
 * 2. 전체 보상 평균과 표준 편차를 사용해 보상 값을 정규화합니다.
 *
 * @param {number[][]} rewardSequences 보상 값의 시퀀스
 * @param {number} discountRate 할인 계수: 0~1 사이의 숫자. 예를 들면 0.95
 * @returns {tf.Tensor[]} 할인되고 정규화된 보상 값을 담은 tf.Tensor 배열
 */
function discountAndNormalizeRewards(rewardSequences, discountRate) {
  return tf.tidy(() => {
    const discounted = [];
    for (const sequence of rewardSequences) {
      discounted.push(discountRewards(sequence, discountRate))
    }
    // 전체 평균과 표준 편차를 계산합니다.
    const concatenated = tf.concat(discounted);
    const mean = tf.mean(concatenated);
    const std = tf.sqrt(tf.mean(tf.square(concatenated.sub(mean))));
    // 평균과 표준 편차를 사용해 보상을 정규화합니다.
    const normalized = discounted.map(rs => rs.sub(mean).div(std));
    return normalized;
  });
}

/**
 * 정규화된 보상 값을 사용해 그레이디언트의 스케일을 조정하고 평균을 계산합니다.
 *
 * 그레이디언트 값을 정규화된 보상 값으로 스케일을 조정합니다.
 * 그다음 모든 게임과 모든 스텝에 걸쳐 평균합니다.
 *
 * @param {{[varName: string]: tf.Tensor[][]}} allGradients 모든 게임과 모든 스텝에 걸쳐
 *   변수 이름과 그레이디언트 값을 매핑한 객체
 * @param {tf.Tensor[]} normalizedRewards 모든 게임에 대한 정규화된 보상 값의 배열
 *   배열의 각 원소는 게임의 스텝 횟수와 같은 길이를 가진 1D tf.Tensor입니다.me.
 * @returns {{[varName: string]: tf.Tensor}} 변수에 대해 스케일을 조정하고 평균한 그레이디언트
 */
function scaleAndAverageGradients(allGradients, normalizedRewards) {
  return tf.tidy(() => {
    const gradients = {};
    for (const varName in allGradients) {
      gradients[varName] = tf.tidy(() => {
        // 그레이디언트를 쌓습니다.
        const varGradients = allGradients[varName].map(
            varGameGradients => tf.stack(varGameGradients));
        // 브로드캐스팅으로 곱셈을 수행하기 위해 보상 텐서의 차원을 확장합니다.
        const expandedDims = [];
        for (let i = 0; i < varGradients[0].rank - 1; ++i) {
          expandedDims.push(1);
        }
        const reshapedNormalizedRewards = normalizedRewards.map(
            rs => rs.reshape(rs.shape.concat(expandedDims)));
        for (let g = 0; g < varGradients.length; ++g) {
          // 이 mul() 호출은 브로드캐스팅을 사용합니다.
          varGradients[g] = varGradients[g].mul(reshapedNormalizedRewards[g]);
        }
        // 그레이디언트를 연결한 다음 모든 게임의 모든 스텝에 대해 평균합니다.
        return tf.mean(tf.concat(varGradients, 0), 0);
      });
    }
    return gradients;
  });
}

setUpUI();
