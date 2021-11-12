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
 * 예나 날씨 데이터셋을 위한 데이터 객체
 *
 * 이 데모에 사용한 데이터는
 * [예나 날씨 데이터셋](https://www.kaggle.com/pankrzysiu/weather-archive-jena)입니다.
 */

const LOCAL_JENA_WEATHER_CSV_PATH = './jena_climate_2009_2016.csv';
const REMOTE_JENA_WEATHER_CSV_PATH =
    'https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv';

/**
 * 예나 날씨 데이터를 가져와 처리하는 클래스
 *
 * 훈련과 검증 데이터의 배치를 순회하는 함수를 만드는 메서드도 제공합니다.
 */
export class JenaWeatherData {
  constructor() {}

  /**
   * 데이터 로딩과 전처리
   *
   * 이 메서드는 먼저 (상대 경로인) `LOCAL_JENA_WEATHER_CSV_PATH`에서 데이터를 로드하고 실패하면
   * 원격 URL(`JENA_WEATHER_CSV_PATH`)에서 데이터를 로드합니다.
   */
  async load() {
    let response;
    try {
      response = await fetch(LOCAL_JENA_WEATHER_CSV_PATH);
    } catch (err) {}

    if (response != null &&
        (response.statusCode === 200 || response.statusCode === 304)) {
      console.log('Loading data from local path');
    } else {
      response = await fetch(REMOTE_JENA_WEATHER_CSV_PATH);
      console.log(
          `Loading data from remote path: ${REMOTE_JENA_WEATHER_CSV_PATH}`);
    }
    const csvData = await response.text();

    // CSV 파일 파싱
    const csvLines = csvData.split('\n');

    // 헤더 파싱
    const columnNames = csvLines[0].split(',');
    for (let i = 0; i < columnNames.length; ++i) {
      // 열 이름 앞뒤의 따옴표를 제거합니다.
      columnNames[i] = columnNames[i].slice(1, columnNames[i].length - 1);
    }

    this.dateTimeCol = columnNames.indexOf('Date Time');
    tf.util.assert(this.dateTimeCol === 0, `Unexpected date-time column index`);

    this.dataColumnNames = columnNames.slice(1);
    this.tempCol = this.dataColumnNames.indexOf('T (degC)');
    tf.util.assert(this.tempCol >= 1, `Unexpected T (degC) column index`);

    this.dateTime = [];
    this.data = [];  // 정규화되지 않은 데이터
    // 0~1 사이로 정규화된 일자 데이터
    this.normalizedDayOfYear = [];
    // 0~1 사이로 정규화된 시간 데이터
    this.normalizedTimeOfDay = [];
    for (let i = 1; i < csvLines.length; ++i) {
      const line = csvLines[i].trim();
      if (line.length === 0) {
        continue;
      }
      const items = line.split(',');
      const parsed = this.parseDateTime_(items[0]);
      const newDateTime = parsed.date;
      if (this.dateTime.length > 0 &&
          newDateTime.getTime() <=
              this.dateTime[this.dateTime.length - 1].getTime()) {
      }

      this.dateTime.push(newDateTime);
      this.data.push(items.slice(1).map(x => +x));
      this.normalizedDayOfYear.push(parsed.normalizedDayOfYear);
      this.normalizedTimeOfDay.push(parsed.normalizedTimeOfDay);
    }
    this.numRows = this.data.length;
    this.numColumns = this.data[0].length;
    this.numColumnsExcludingTarget = this.data[0].length - 1;
    console.log(
        `this.numColumnsExcludingTarget = ${this.numColumnsExcludingTarget}`);

    await this.calculateMeansAndStddevs_();
  }

  /**
   * 예나 날씨 CSV 파일에서 날짜-시간 문자열을 파싱합니다.
   *
   * @param {*} str "17.01.2009 22:10:00"와 같은 포맷의 날짜 시간 문자열
   * @returns date: 자바스크립트 Date 객체
   *          normalizedDayOfYear: 0~1 사이로 정규화된 일자
   *          normalizedTimeOfDay: 0~1 사이로 정규화된 시간
   */
  parseDateTime_(str) {
    const items = str.split(' ');
    const dateStr = items[0];
    const dateStrItems = dateStr.split('.');
    const day = +dateStrItems[0];
    const month = +dateStrItems[1] - 1;  // 자바스크립트 `Date` 클래스에서 월은 0부터 시작합니다.
    const year = +dateStrItems[2];

    const timeStrItems = items[1].split(':');
    const hours = +timeStrItems[0];
    const minutes = +timeStrItems[1];
    const seconds = +timeStrItems[2];

    const date = new Date(Date.UTC(year, month, day, hours, minutes, seconds));
    const yearOnset = new Date(year, 0, 1);
    const normalizedDayOfYear =
        (date - yearOnset) / (366 * 1000 * 60 * 60 * 24);
    const dayOnset = new Date(year, month, day);
    const normalizedTimeOfDay = (date - dayOnset) / (1000 * 60 * 60 * 24)
    return {date, normalizedDayOfYear, normalizedTimeOfDay};
  }


  /**
   * 열의 평균과 표준편차를 계산합니다.
   *
   * 가속을 위해 TensorFlow.js를 사용합니다.
   */
  async calculateMeansAndStddevs_() {
    tf.tidy(() => {
      // 일부 컴퓨터에서 WebGL OOM를 피하기 위해 한 번에 전체 열을 계산하지 않고
      // 한 열씩 처리합니다.
      this.means = [];
      this.stddevs = [];
      for (const columnName of this.dataColumnNames) {
        const data =
            tf.tensor1d(this.getColumnData(columnName).slice(0, 6 * 24 * 365));
        const moments = tf.moments(data);
        this.means.push(moments.mean.dataSync());
        this.stddevs.push(Math.sqrt(moments.variance.dataSync()));
      }
      console.log('means:', this.means);
      console.log('stddevs:', this.stddevs);
    });

    // 정규화된 값을 캐싱합니다.
    this.normalizedData = [];
    for (let i = 0; i < this.numRows; ++i) {
      const row = [];
      for (let j = 0; j < this.numColumns; ++j) {
        row.push((this.data[i][j] - this.means[j]) / this.stddevs[j]);
      }
      this.normalizedData.push(row);
    }
  }

  getDataColumnNames() {
    return this.dataColumnNames;
  }

  getTime(index) {
    return this.dateTime[index];
  }

  /** 데이터 열의 평균과 표준 편차를 반환합니다. */
  getMeanAndStddev(dataColumnName) {
    if (this.means == null || this.stddevs == null) {
      throw new Error('평균과 표준 편차를 아직 계산하지 않았습니다.');
    }

    const index = this.getDataColumnNames().indexOf(dataColumnName);
    if (index === -1) {
      throw new Error(`잘못된 열 이름: ${dataColumnName}`);
    }
    return {
      mean: this.means[index], stddev: this.stddevs[index]
    }
  }

  getColumnData(
      columnName, includeTime, normalize, beginIndex, length, stride) {
    const columnIndex = this.dataColumnNames.indexOf(columnName);
    tf.util.assert(columnIndex >= 0, `잘못된 열 이름: ${columnName}`);

    if (beginIndex == null) {
      beginIndex = 0;
    }
    if (length == null) {
      length = this.numRows - beginIndex;
    }
    if (stride == null) {
      stride = 1;
    }
    const out = [];
    for (let i = beginIndex; i < beginIndex + length && i < this.numRows;
         i += stride) {
      let value = normalize ? this.normalizedData[i][columnIndex] :
                              this.data[i][columnIndex];
      if (includeTime) {
        value = {x: this.dateTime[i].getTime(), y: value};
      }
      out.push(value);
    }
    return out;
  }

  /**
   * 데이터 반복 함수를 반환합니다.
   *
   * @param {boolean} shuffle 데이터를 섞을지 여부.
   *   `false`로 지정하면 반환된 반복자 함수를 호출하여 생성된 샘플은 순서대로
   *   `minIndex`와 `maxIndex`로 지정된 범위를 (지정하지 않으면 CSV 파일 전체 범위를) 스캔합니다.
   *   `true`로 지정하면 반환된 반복자 함수가 생성된 샘플은 랜덤한 행에서 시작합니다.
   * @param {number} lookBack 룩백(look-back) 스텝 횟수.
   *   예측을 만들 때 사용할 이전 데이터 스텝. 일반적인 값: 10일(즉, 6 * 24 * 10 = 1440)
   * @param {number} delay 입력 특성의 마지막 시간부터 예측 시간까지 타임 스텝 수.
   *   일반적인 값: 1일 (즉, 6 * 24 = 144).
   * @param {number} batchSize 배치 크기.
   * @param {number} step 입력 특성에서 연속된 포인트 간의 스텝 수.
   *   입력 특성의 다운샘플링 인자입니다. 일반적인 값: 1시간 (즉, 6).
   * @param {number} minIndex (옵션 매개변수) 원본 데이터셋에서 추출하기 위한 최소 인덱스.
   *   `maxIndex`와 함께 검증이나 평가를 위해 원본 데이터의 일부를 예약하는데 사용할 수 있습니다.
   * @param {number} maxIndex (옵션 매개변수) 원본 데이터셋에서 추출하기 위한 최대 인덱스.
   *   `minIndex`와 함께 검증이나 평가를 위해 원본 데이터의 일부를 예약하는데 사용할 수 있습니다.
   * @param {boolean} normalize 반복 함수가 정규화된 데이터를 반환할지 여부.
   * @param {boolean} includeDateTime 정규화된 일자와 시간과 함께 날짜-시간 특성을 포함할지 여부.
   * @return {Function} 특성과 타깃의 배치를 반환하는 반복 함수.
   *   특성과 타깃은 순서대로 길이 2인 배열로 구성됩니다.
   *   특성은 `[batchSize, Math.floor(lookBack / step), featureLength]`
   *     크기의 float32 타입의 `tf.Tensor`로 표현됩니다.
   *   타깃은 `[batchSize, 1]` 크기의 float32 타입의 `tf.Tensor`로 표현됩니다.
   */
  getNextBatchFunction(
      shuffle, lookBack, delay, batchSize, step, minIndex, maxIndex, normalize,
      includeDateTime) {
    let startIndex = minIndex + lookBack;
    const lookBackSlices = Math.floor(lookBack / step);

    return {
      next: () => {
        const rowIndices = [];
        let done = false;  // 데이터셋이 끝났는지 나타냅니다.
        if (shuffle) {
          // `shuffle`이 `true`이면 랜덤하게 선택한 행부터 시작합니다.
          const range = maxIndex - (minIndex + lookBack);
          for (let i = 0; i < batchSize; ++i) {
            const row = minIndex + lookBack + Math.floor(Math.random() * range);
            rowIndices.push(row);
          }
        } else {
          // `shuffle`이 `false`이면 minIndex부터 순서대로 시작합니다.
          let r = startIndex;
          for (; r < startIndex + batchSize && r < maxIndex; ++r) {
            rowIndices.push(r);
          }
          if (r >= maxIndex) {
            done = true;
          }
        }

        const numExamples = rowIndices.length;
        startIndex += numExamples;

        const featureLength =
            includeDateTime ? this.numColumns + 2 : this.numColumns;
        const samples = tf.buffer([numExamples, lookBackSlices, featureLength]);
        const targets = tf.buffer([numExamples, 1]);
        // 샘플을 순회합니다. 한 샘플은 여러 개의 행을 담고 있습니다.
        for (let j = 0; j < numExamples; ++j) {
          const rowIndex = rowIndices[j];
          let exampleRow = 0;
          // 샘플에 있는 행을 순회합니다.
          for (let r = rowIndex - lookBack; r < rowIndex; r += step) {
            let exampleCol = 0;
            // 행에 있는 특성을 순회합니다.
            for (let n = 0; n < featureLength; ++n) {
              let value;
              if (n < this.numColumns) {
                value = normalize ? this.normalizedData[r][n] : this.data[r][n];
              } else if (n === this.numColumns) {
                // 정규화된 일자 특성
                value = this.normalizedDayOfYear[r];
              } else {
                // 정규화된 시간 특성
                value = this.normalizedTimeOfDay[r];
              }
              samples.set(value, j, exampleRow, exampleCol++);
            }

            const value = normalize ?
                this.normalizedData[r + delay][this.tempCol] :
                this.data[r + delay][this.tempCol];
            targets.set(value, j, 0);
            exampleRow++;
          }
        }
        return {
          value: {xs: samples.toTensor(), ys: targets.toTensor()},
          done
        };
      }
    };
  }
}
