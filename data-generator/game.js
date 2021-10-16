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
 * 이 파일은 단순화된 포커 게임과 비슷한 2인용 카드 게임을 구현합니다.
 * 카드는 1에서 GAME_STATE.max_card_value까지 값을 가집니다.
 * 각 플레이어는 [1, GAME_STATE.max_card_value] 범위에서 균등하게 샘플링한 세 개의 카드를 받습니다.
 * 더 좋은 카드를 가진 플레이어가 이깁니다.
 *
 *    트리플 카드(같은 카드 세 개)가 더블 카드를 이깁니다.
 *    더블 카드(같은 카드 두 개)가 싱글 카드를 이깁니다.
 *    두 플레이어가 모두 트리플이라면 높은 숫자의 트리플이 이깁니다.
 *    두 플레이어가 모두 더블이라면 높은 숫자의 더블이 이깁니다.
 *    두 플레이어가 모두 더블이 아니라면 높은 숫자의 카드를 가진 플레이어가 이깁니다.
 *    두 플레이어의 카드 숫자가 동일하면 랜덤하게 50:50 확률로 승자를 결정합니다.
 */

// 게임 규칙과 지금까지 플레이된 게임 횟수를 추적하기 위한 상태.
// 개별 항목은 다른 모듈에서 수정될 수 있습니다.
export const GAME_STATE = {
  // 게임 플레이 횟수.
  // 모델을 훈련하기 위해 준비한 시뮬레이션 횟수를 표현하는데 유용합니다.
  num_simulations_so_far: 0,
  // 카드 값의 범위를 정하는 상수
  min_card_value: 1,
  max_card_value: 13,
  // 카드 갯수. 화면에서 수정할 수 있습니다.
  num_cards_per_hand: 3
};

/**
 * [GAME_STATE.min_card_value, GAME_STATE.max_card_value] 범위에서 랜덤한 정수를 반환합니다.
 */
export function getRandomDigit() {
  return Math.floor(
      GAME_STATE.min_card_value + Math.random() * GAME_STATE.max_card_value);
}

/**
 * GAME_STATE.num_cards_per_hand 개의 원소로 구성된 랜덤한 배열을 반환합니다.
 * 각 원소는 [GAME_STATE.min_card_value, GAME_STATE.max_card_value] 범위에서 랜덤하게 선택한 정수입니다.
 * 카드는 정렬하여 반환합니다.
 */
export function randomHand() {
  const hand = [];
  for (let i = 0; i < GAME_STATE.num_cards_per_hand; i++) {
    hand.push(getRandomDigit());
  }
  return hand.sort((a, b) => a - b);
}

/**
 * 카드 그룹(단일 카드, 페어, 트리플 등) 별로 가장 큰 값을 나타내는 배열을 만듭니다.
 * 0은 해당 그룹 크기가 없다는 것을 의미합니다.
 *
 * 예를 들어, 이 함수가 반환한 값이 [9, 3, 0, 0, 0, 0]라면 다음과 같은 의미입니다.
 *   - 가장 큰 단일 카드는 9입니다.
 *   - 가장 큰 더블 카드는 3입니다.
 *   - 트리플 카드는 없습니다.
 *
 * 손에 든 카드가 [1, 2, 2, 3, 3, 9]일 때 이런 출력이 만들어 집니다.
 *
 * @param {number[]} hand GAME_STATE.num_cards_per_hand 크기의 정렬된 정수 배열
 * @returns {number[]} 카드 그룹 별로 가장 큰 값을 나타내는 배열.
 *     0은 해당 그룹 크기가 없다는 것을 의미합니다.
 */
export function handScoringHelper(hand) {
  // 모두 0인 배열을 만듭니다.
  const faceValOfEachGroup = [];
  for (let i = 0; i < GAME_STATE.num_cards_per_hand; i++) {
    faceValOfEachGroup.push(0);
  }
  let runLength = 0;
  let prevVal = 0;
  for (let i = 0; i < GAME_STATE.num_cards_per_hand; i++) {
    const card = hand[i];
    if (card == prevVal) {
      runLength += 1;
    } else {
      prevVal = card;
      runLength = 1;
    }
    faceValOfEachGroup[runLength - 1] = card;
  }
  return faceValOfEachGroup;
}

/**
 * hand1이 hand2를 이기면 1을 반환합니다.
 * hand2가 hand1을 이기면 0을 반환합니다.
 * 동률이면 절반의 확률로 0이나 1을 랜덤하게 반환합니다.
 * @param {number[]} hand1 플레이어 1의 정렬된 숫자
 * @param {number[]} hand2 플레이어 2의 정렬된 숫자
 * @returns {number} 플레이어 1이 이겼는지 졌는지를 나타내는 1 또는 0
 */
export function compareHands(hand1, hand2) {
  const handScore1 = handScoringHelper(hand1);
  const handScore2 = handScoringHelper(hand2);
  // 그룹 크기의 역순으로 어떤 플레이어가 더 나은지 결정합니다.
  for (let group = GAME_STATE.num_cards_per_hand - 1; group >= 0; group--) {
    if (handScore1[group] > handScore2[group]) {
      return 1;
    }
    if (handScore1[group] < handScore2[group]) {
      return 0;
    }
  }
  // 동률이면 랜덤하게 승자를 결정합니다.
  if (Math.random() > 0.5) {
    return 1;
  }
  return 0;
}

/**
 * 한 번의 게임 플레이를 표현하는 객체를 반환합니다.
 * 두 플레이어의 카드를 만들고 두 카드를 비교합니다.
 * [hand1, hand2, whetherHand1Wins]을 반환합니다.
 */
export function generateOnePlay() {
  const player1Hand = randomHand();
  const player2Hand = randomHand();
  const player1Win = compareHands(player1Hand, player2Hand);
  GAME_STATE.num_simulations_so_far++;
  return {player1Hand, player2Hand, player1Win};
}
