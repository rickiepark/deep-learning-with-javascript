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

if (typeof tf === 'undefined') {
  global.tf = require('@tensorflow/tfjs');
}

export const TEXT_DATA_URLS = {
  'nietzsche': {
    url:
        'https://storage.googleapis.com/tfjs-examples/lstm-text-generation/data/nietzsche.txt',
    needle: 'Nietzsche'
  },
  'julesverne': {
    url:
        'https://storage.googleapis.com/tfjs-examples/lstm-text-generation/data/t1.verne.txt',
    needle: 'Jules Verne'
  },
  'shakespeare': {
    url:
        'https://storage.googleapis.com/tfjs-examples/lstm-text-generation/data/t8.shakespeare.txt',
    needle: 'Shakespeare'
  },
  'tfjs-code': {
    url: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.11.7/dist/tf.js',
    needle: 'TensorFlow.js Code (Compiled, 0.11.7)'
  }
}

/**
 * 텍스트 데이터를 위한 클래스
 *
 * 이 클래스는 다음과 같은 작업을 수행합니다:
 *
 * - (문자열) 훈련 데이터를 원-핫 인코딩 벡터로 변환합니다.
 * - 훈련 데이터에서 랜덤한 슬라이스를 뽑습니다.
 *   모델 훈련과 텍스트 생성시 시드 텍스트에 사용합니다.
 */
export class TextData {
  /**
   * TextData 생성자
   *
   * @param {string} dataIdentifier TextData 객체를 위한 식별자
   * @param {string} textString 훈련 텍스트 데이터
   * @param {number} sampleLen 훈련 샘플의 길이, 즉 LSTM 모델이 기대하는 입력 시퀀스 길이
   * @param {number} sampleStep 훈련 데이터(`textString`)에 있는 한 샘플에서 다음 샘플로 이동할 때
   *   건너 뛸 문자 개수
   */
  constructor(dataIdentifier, textString, sampleLen, sampleStep) {
    tf.util.assert(
        sampleLen > 0,
        `sampleLen은 양의 정수여야 합니다: ${sampleLen}`);
    tf.util.assert(
        sampleStep > 0,
        `sampleStep은 양의 정수여야 합니다: ${sampleStep}`);

    if (!dataIdentifier) {
      throw new Error('모델 식별자를 입력해 주세요.');
    }

    this.dataIdentifier_ = dataIdentifier;

    this.textString_ = textString;
    this.textLen_ = textString.length;
    this.sampleLen_ = sampleLen;
    this.sampleStep_ = sampleStep;

    this.getCharSet_();
    this.convertAllTextToIndices_();
  }

  /**
   * 데이터 식별자 얻기
   *
   * @returns {string} 데이터 식별자
   */
  dataIdentifier() {
    return this.dataIdentifier_;
  }

  /**
   * 훈련 텍스트 데이터 길이 얻기
   *
   * @returns {number} 훈련 텍스트 데이터의 길이
   */
  textLen() {
    return this.textLen_;
  }

  /**
   * 각 훈련 샘플의 길이 얻기
   */
  sampleLen() {
    return this.sampleLen_;
  }

  /**
   * 문자 집합의 크기 얻기
   *
   * @returns {number} 문자 집합의 크기. 즉, 훈련 텍스트 데이터에 있는 고유한 문자 개수.
   */
  charSetSize() {
    return this.charSetSize_;
  }

  /**
   * 모델 훈련을 위한 다음 에포크 데이터를 생성하기
   *
   * @param {number} numExamples 생성할 샘플 개수
   * @returns {[tf.Tensor, tf.Tensor]} `xs`와 `ys` 텐서.
   *   `xs`는 `[numExamples, this.sampleLen, this.charSetSize]` 크기 입니다..
   *   `ys`는 `[numExamples, this.charSetSize]` 크기 입니다.
   */
  nextDataEpoch(numExamples) {
    this.generateExampleBeginIndices_();

    if (numExamples == null) {
      numExamples = this.exampleBeginIndices_.length;
    }

    const xsBuffer = new tf.TensorBuffer([
        numExamples, this.sampleLen_, this.charSetSize_]);
    const ysBuffer  = new tf.TensorBuffer([numExamples, this.charSetSize_]);
    for (let i = 0; i < numExamples; ++i) {
      const beginIndex = this.exampleBeginIndices_[
          this.examplePosition_ % this.exampleBeginIndices_.length];
      for (let j = 0; j < this.sampleLen_; ++j) {
        xsBuffer.set(1, i, j, this.indices_[beginIndex + j]);
      }
      ysBuffer.set(1, i, this.indices_[beginIndex + this.sampleLen_]);
      this.examplePosition_++;
    }
    return [xsBuffer.toTensor(), ysBuffer.toTensor()];
  }

  /**
   * 문자 집합에서 주어진 인덱스의 문자 얻기
   *
   * @param {number} index
   * @returns {string} 문자 집합의 `index` 위치에 있는 문자
   */
  getFromCharSet(index) {
    return this.charSet_[index];
  }

  /**
   * 텍스트 문자열을 정수 인덱스로 변환합니다.
   *
   * @param {string} text 입력 텍스트
   * @returns {number[]} `text`에 있는 문자의 인덱스
   */
  textToIndices(text) {
    const indices = [];
    for (let i = 0; i < text.length; ++i) {
      indices.push(this.charSet_.indexOf(text[i]));
    }
    return indices;
  }

  /**
   * 텍스트 데이터에서 랜덤한 슬라이스를 가져옵니다.
   *
   * @returns {[string, number[]} 슬라이스의 문자열과 인덱스
   */
  getRandomSlice() {
    const startIndex =
        Math.round(Math.random() * (this.textLen_ - this.sampleLen_ - 1));
    const textSlice = this.slice_(startIndex, startIndex + this.sampleLen_);
    return [textSlice, this.textToIndices(textSlice)];
  }

  /**
   * 훈련 텍스트 데이터에서 슬라이스를 얻습니다.
   *
   * @param {number} startIndex
   * @param {number} endIndex
   * @param {bool} useIndices 문자열 대신 인덱스를 반환할지 여부
   * @returns {string | Uint16Array} 슬라이싱 결과
   */
  slice_(startIndex, endIndex) {
    return this.textString_.slice(startIndex, endIndex);
  }

  /**
   * 텍스트에서 고유한 문자 집합을 만듭니다.
   */
  getCharSet_() {
    this.charSet_ = [];
    for (let i = 0; i < this.textLen_; ++i) {
      if (this.charSet_.indexOf(this.textString_[i]) === -1) {
        this.charSet_.push(this.textString_[i]);
      }
    }
    this.charSetSize_ = this.charSet_.length;
  }

  /**
   * 모든 훈련 텍스트를 정수 인덱스로 바꿉니다.
   */
  convertAllTextToIndices_() {
    this.indices_ = new Uint16Array(this.textToIndices(this.textString_));
  }

  /**
   * 샘플의 시작 인덱스를 생성합니다. 그다음 랜덤하게 섞습니다.
   */
  generateExampleBeginIndices_() {
    // 샘플의 시작 인덱스를 준비합니다.
    this.exampleBeginIndices_ = [];
    for (let i = 0;
        i < this.textLen_ - this.sampleLen_ - 1;
        i += this.sampleStep_) {
      this.exampleBeginIndices_.push(i);
    }

    // 시작 인덱스를 랜덤하게 섞습니다.
    tf.util.shuffle(this.exampleBeginIndices_);
    this.examplePosition_ = 0;
  }
}
