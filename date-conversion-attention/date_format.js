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
 * 날짜 포맷과 변환 유틸리티 함수
 *
 * 이 파일은 날짜 변환 모델의 훈련과 훈련된 모델을 기반으로한 날짜 변환에 사용됩니다.
 *
 * 랜덤한 날짜를 생성하고 2019-01-20나 20JAN19와 같이 여러 포맷으로 표현하는 함수를 포함합니다.
 * 또한 날짜의 텍스트 표현을 원-핫 `tf.Tensor` 표현으로 바꾸는 함수를 포함합니다.
 */

const MONTH_NAMES_FULL = [
  'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
  'September', 'October', 'November', 'December'
];
const MONTH_NAMES_3LETTER =
    MONTH_NAMES_FULL.map(name => name.slice(0, 3).toUpperCase());

const MIN_DATE = new Date('1950-01-01').getTime();
const MAX_DATE = new Date('2050-01-01').getTime();

export const INPUT_LENGTH = 12   // 모든 입력 포맷의 최대 길이
export const OUTPUT_LENGTH = 10  // 'YYYY-MM-DD'의 길이

// 입력과 출력의 패딩을 위해 "\n"을 사용합니다.
// 케라스 모델에서 `mask_zero=True`를 사용할 수 있도록 초기에 패딩되어야 합니다.
export const INPUT_VOCAB = '\n0123456789/-., ' +
    MONTH_NAMES_3LETTER.join('')
        .split('')
        .filter(function(item, i, ar) {
          return ar.indexOf(item) === i;
        })
        .join('');

// OUTPUT_VOCAB는 '\n'으로 표현된 시퀀스 시작 토큰을 포함합니다.
// 날짜 문자열은 단어나 다른 것이 아니라 날짜를 구성하는 문자로 표현됩니다.
export const OUTPUT_VOCAB = '\n\t0123456789-';

export const START_CODE = 1;

/**
 * 랜덤한 날짜를 생성합니다.
 *
 * @return {[number, number, number]} 정수 값의 연도, 1과 12 사이의 정수인 월, 1보다 큰 정수인 일
 */
export function generateRandomDateTuple() {
  const date = new Date(Math.random() * (MAX_DATE - MIN_DATE) + MIN_DATE);
  return [date.getFullYear(), date.getMonth() + 1, date.getDate()];
}

function toTwoDigitString(num) {
  return num < 10 ? `0${num}` : `${num}`;
}

/** 01202019 포맷 */
export function dateTupleToDDMMMYYYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dayStr}${monthStr}${dateTuple[0]}`;
}

/** 01/20/2019 포맷 */
export function dateTupleToMMSlashDDSlashYYYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${monthStr}/${dayStr}/${dateTuple[0]}`;
}

/** 1/20/2019 포맷 */
export function dateTupleToMSlashDSlashYYYY(dateTuple) {
  return `${dateTuple[1]}/${dateTuple[2]}/${dateTuple[0]}`;
}

/** 01/20/19 포맷 */
export function dateTupleToMMSlashDDSlashYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr}/${dayStr}/${yearStr}`;
}

/** 1/20/19 포맷 */
export function dateTupleToMSlashDSlashYY(dateTuple) {
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${dateTuple[1]}/${dateTuple[2]}/${yearStr}`;
}

/** 012019 포맷 */
export function dateTupleToMMDDYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr}${dayStr}${yearStr}`;
}

/** JAN 20 19 포맷 */
export function dateTupleToMMMSpaceDDSpaceYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr} ${dayStr} ${yearStr}`;
}

/** JAN 20 2019 포맷 */
export function dateTupleToMMMSpaceDDSpaceYYYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${monthStr} ${dayStr} ${dateTuple[0]}`;
}

/** JAN 20, 19 포맷 */
export function dateTupleToMMMSpaceDDCommaSpaceYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  const yearStr = `${dateTuple[0]}`.slice(2);
  return `${monthStr} ${dayStr}, ${yearStr}`;
}

/** JAN 20, 2019 포맷 */
export function dateTupleToMMMSpaceDDCommaSpaceYYYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${monthStr} ${dayStr}, ${dateTuple[0]}`;
}

/** 20-01-2019 포맷 */
export function dateTupleToDDDashMMDashYYYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dayStr}-${monthStr}-${dateTuple[0]}`;
}

/** 20-1-2019 포맷 */
export function dateTupleToDDashMDashYYYY(dateTuple) {
  return `${dateTuple[2]}-${dateTuple[1]}-${dateTuple[0]}`;
}

/** 20.01.2019 포맷 */
export function dateTupleToDDDotMMDotYYYY(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dayStr}.${monthStr}.${dateTuple[0]}`;
}

/** 20.1.2019 포맷 */
export function dateTupleToDDotMDotYYYY(dateTuple) {
  return `${dateTuple[2]}.${dateTuple[1]}.${dateTuple[0]}`;
}

/** 2019.01.20 포맷 */
export function dateTupleToYYYYDotMMDotDD(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dateTuple[0]}.${monthStr}.${dayStr}`;
}

/** 2019.1.20 포맷 */
export function dateTupleToYYYYDotMDotD(dateTuple) {
  return `${dateTuple[0]}.${dateTuple[1]}.${dateTuple[2]}`;
}

/** 20190120 포맷 */
export function dateTupleToYYYYMMDD(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dateTuple[0]}${monthStr}${dayStr}`;
}

/** 2019-1-20 포맷 */
export function dateTupleToYYYYDashMDashD(dateTuple) {
  return `${dateTuple[0]}-${dateTuple[1]}-${dateTuple[2]}`;
}

/** 20 JAN 2019 포맷 */
export function dateTupleToDSpaceMMMSpaceYYYY(dateTuple) {
  const monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1];
  return `${dateTuple[2]} ${monthStr} ${dateTuple[0]}`;
}

/**
 * 2019-01-20 포맷
 * (즉, ISO 포맷과 타깃).
 * */
export function dateTupleToYYYYDashMMDashDD(dateTuple) {
  const monthStr = toTwoDigitString(dateTuple[1]);
  const dayStr = toTwoDigitString(dateTuple[2]);
  return `${dateTuple[0]}-${monthStr}-${dayStr}`;
}

export const INPUT_FNS = [
  dateTupleToDDMMMYYYY,
  dateTupleToMMDDYY,
  dateTupleToMMSlashDDSlashYY,
  dateTupleToMMSlashDDSlashYYYY,
  dateTupleToMSlashDSlashYYYY,
  dateTupleToDDDashMMDashYYYY,
  dateTupleToDDashMDashYYYY,
  dateTupleToMMMSpaceDDSpaceYY,
  dateTupleToMSlashDSlashYY,
  dateTupleToMMMSpaceDDSpaceYYYY,
  dateTupleToMMMSpaceDDCommaSpaceYY,
  dateTupleToMMMSpaceDDCommaSpaceYYYY,
  dateTupleToDDDotMMDotYYYY,
  dateTupleToDDotMDotYYYY,
  dateTupleToYYYYDotMMDotDD,
  dateTupleToYYYYDotMDotD,
  dateTupleToYYYYMMDD,
  dateTupleToYYYYDashMDashD,
  dateTupleToDSpaceMMMSpaceYYYY,
  dateTupleToYYYYDashMMDashDD
];

/**
 * 여러 개의 입력 날짜 문자열을 `tf.Tensor`로 인코딩합니다.
 *
 * 인코딩은 원-핫 벡터의 시퀀스입니다.
 * 이 시퀀스는 유효한 입력 날짜 문자열의 최대 길이가 되도록 끝에 패딩됩니다.
 * 패딩 값은 0입니다.
 *
 * @param {string[]} dateStrings 입력 날짜 문자열. 이 배열의 각 원소는 위에 나열된 포맷 중 하나여야 합니다.
 *   배열에 여러 개의 포맷이 혼합되어 있어도 괜찮습니다.
 * @returns {tf.Tensor} 원-핫 인코딩된 문자로 `[numExamples, maxInputLength]` 크기의 `float32` `tf.Tensor`입니다.
 *   여기에서 `maxInputLength`는 유효한 입력 날짜 문자열 포맷의 최대 입력 길이입니다.
 */
export function encodeInputDateStrings(dateStrings) {
  const n = dateStrings.length;
  const x = tf.buffer([n, INPUT_LENGTH], 'float32');
  for (let i = 0; i < n; ++i) {
    for (let j = 0; j < INPUT_LENGTH; ++j) {
      if (j < dateStrings[i].length) {
        const char = dateStrings[i][j];
        const index = INPUT_VOCAB.indexOf(char);
        if (index === -1) {
          throw new Error(`알 수 없는 문자: ${char}`);
        }
        x.set(index, i, j);
      }
    }
  }
  return x.toTensor();
}

/**
 * `tf.Tensor`로 여러 개의 출력 날짜 문자열을 인코딩합니다.
 *
 * 이 인코딩은 정수 인덱스의 시퀀스입니다.
 *
 * @param {string[]} dateStrings 출력 날짜 문자열의 배열. ISO 날짜 포맷(YYYY-MM-DD).
 * @returns {tf.Tensor} 문자의 정수 인덱스.
 *   `[numExamples, outputLength]` 크기의 `int32` `tf.Tensor`.
 *   여기에서 `outputLength`는 표준 출력 포맷의 길이(즉, 10)입니다.
 */
export function encodeOutputDateStrings(dateStrings, oneHot = false) {
  const n = dateStrings.length;
  const x = tf.buffer([n, OUTPUT_LENGTH], 'int32');
  for (let i = 0; i < n; ++i) {
    tf.util.assert(
        dateStrings[i].length === OUTPUT_LENGTH,
        `날짜 문자열이 ISO 포맷이 아닙니다: "${dateStrings[i]}"`);
    for (let j = 0; j < OUTPUT_LENGTH; ++j) {
      const char = dateStrings[i][j];
      const index = OUTPUT_VOCAB.indexOf(char);
      if (index === -1) {
        throw new Error(`알 수 없는 문자: ${char}`);
      }
      x.set(index, i, j);
    }
  }
  return x.toTensor();
}
