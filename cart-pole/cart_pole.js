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
 * 다음을 기반으로 구현했습니다: http://incompleteideas.net/book/code/pole.c
 */

/**
 * Cart-pole 시스템 시뮬레이터
 *
 * 제어 이론 입장에서 이 시스템은 네 가지 상태 변수가 있습니다:
 *
 *   - x: 카트의 1D 위치
 *   - xDot: 카트의 속도
 *   - theta: 막대의 각도 (라디안). 0이 수직 위치에 해당합니다.
 *   - thetaDot: 막대의 각속도
 *
 * 이 시스템은 하나의 행동으로 제어됩니다:
 *
 *   - 왼쪽으로 미는 힘이나 오른쪽으로 미는 힘
 */
export class CartPole {
  /**
   * CartPole의 생성자
   */
  constructor() {
    // 시스템의 상수
    this.gravity = 9.8;
    this.massCart = 1.0;
    this.massPole = 0.1;
    this.totalMass = this.massCart + this.massPole;
    this.cartWidth = 0.2;
    this.cartHeight = 0.1;
    this.length = 0.5;
    this.poleMoment = this.massPole * this.length;
    this.forceMag = 10.0;
    this.tau = 0.02;  // 상태 업데이트 시간 간격

    // 시뮬레이션을 실패로 표시하기 위한 임곗값
    this.xThreshold = 2.4;
    this.thetaThreshold = 12 / 360 * 2 * Math.PI;

    this.setRandomState();
  }

  /**
   * cart-pole 시스템의 상태를 랜덤하게 설정합니다.
   */
  setRandomState() {
    // cart-pole 시스템의 제어 이론 상태 변수
    // 카트 위치 (미터)
    this.x = Math.random() - 0.5;
    // 카트 속도
    this.xDot = (Math.random() - 0.5) * 1;
    // 막대 각도 (라디안)
    this.theta = (Math.random() - 0.5) * 2 * (6 / 360 * 2 * Math.PI);
    // 막대 각속도
    this.thetaDot =  (Math.random() - 0.5) * 0.5;
  }

  /**
   * [1, 4] 크기의 tf.Tensor로 현재 상태를 반환합니다.
   */
  getStateTensor() {
    return tf.tensor2d([[this.x, this.xDot, this.theta, this.thetaDot]]);
  }

  /**
   * 행동을 사용해 cart-pole 시스템을 업데이트합니다.
   * @param {number} action `action`의 부호만 중요합니다.
   *   0보다 크면 고정된 크기의 오른쪽으로 미는 힘입니다.
   *   0보다 작거나 같으면 고정된 크기의 왼쪽으로 미는 힘입니다.
   */
  update(action) {
    const force = action > 0 ? this.forceMag : -this.forceMag;

    const cosTheta = Math.cos(this.theta);
    const sinTheta = Math.sin(this.theta);

    const temp =
        (force + this.poleMoment * this.thetaDot * this.thetaDot * sinTheta) /
        this.totalMass;
    const thetaAcc = (this.gravity * sinTheta - cosTheta * temp) /
        (this.length *
         (4 / 3 - this.massPole * cosTheta * cosTheta / this.totalMass));
    const xAcc = temp - this.poleMoment * thetaAcc * cosTheta / this.totalMass;

    // 오일러 방법을 사용해 네 개의 상태 변수를 업데이트합니다.
    this.x += this.tau * this.xDot;
    this.xDot += this.tau * xAcc;
    this.theta += this.tau * this.thetaDot;
    this.thetaDot += this.tau * thetaAcc;

    return this.isDone();
  }

  /**
   * 시뮬레이션이 끝났는지 결정합니다.
   *
   * `x`(카트의 위치)가 경계를 넘어가거나
   * `theta`(막대의 각도)가 임계점을 넘어가면 시뮬레이션이 종료됩니다.
   *
   * @returns {bool} Whether the simulation is done.
   */
  isDone() {
    return this.x < -this.xThreshold || this.x > this.xThreshold ||
        this.theta < -this.thetaThreshold || this.theta > this.thetaThreshold;
  }
}
