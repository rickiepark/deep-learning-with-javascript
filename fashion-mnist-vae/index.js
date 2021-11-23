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
 * 이 파일은 노드에서 훈련한 VAE를 위한 브라우저 기반의 뷰어를 제공합니다.
 */

// 다음 위치에 모델이 있는지 확인하세요.
// 이 주소를 브라우저에 붙여 넣고 json 파일을 출력하는지 확인하세요.
const decoderUrl = './models/decoder/model.json';

let decoder;

const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_CHANNELS = 1;

const LATENT_DIMS = 2;

async function loadModel(modelUrl) {
  const decoder = await tf.loadLayersModel(modelUrl);

  const queryString = window.location.search.substring(1);
  if (queryString.match('debug')) {
    tfvis.show.modelSummary({name: 'decoder'}, decoder);
    tfvis.show.layer({name: 'dense2'}, decoder.getLayer('dense_Dense2'));
    tfvis.show.layer({name: 'dense3'}, decoder.getLayer('dense_Dense3'));
  }
  return decoder;
}

/**
 * 잠재 공간을 위한 표현을 생성합니다.
 *
 * 각 차원을 표현하는 텐서 배열을 반환합니다.
 * 각 차원에 있는 포인트 간의 간격은 동일합니다.
 *
 * @param {number} dimensions 차원 수
 * @param {number} pointsPerDim 각 차원의 포인트 수
 * @param {number} start 시작값
 * @param {number} end 종료값
 * @returns {Tensor1d[]}
 */
function generateLatentSpace(dimensions, pointsPerDim, start, end) {
  const result = [];
  for (let i = 0; i < dimensions; i++) {
    const values = tf.linspace(start, end, pointsPerDim);
    result.push(values);
  }

  return result;
}

/**
 * z 벡터(의 배치)를 이미지 텐서로 디코딩합니다.
 * Z는 이미지를 생성하려는 잠재 공간의 벡터입니다.
 *
 * [batch, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS] 크기의 이미지 텐서를 반환합니다.
 *
 * @param {Tensor2D} inputTensor [batch, LATENT_DIMS] 크기의 이미지 텐서
 */
function decodeZ(inputTensor) {
  return tf.tidy(() => {
    const res = decoder.predict(inputTensor).mul(255).cast('int32');
    const reshaped = res.reshape(
        [inputTensor.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]);
    return reshaped;
  });
}

/**
 * VAE를 통해 z 벡터로 잠재 공간을 렌더링하고 결과를 출력합니다.
 *
 * 2D 잠재 공간만 다룹니다.
 */
async function renderLatentSpace(latentSpace) {
  document.getElementById('plot-area').innerText = '';
  const [xAxis, yAxis] = latentSpace;

  // 캔바스를 만듭니다.
  const xPlaceholder = Array(xAxis.shape[0]).fill(0);
  const yPlaceholder = Array(yAxis.shape[0]).fill(0);

  const rows = d3.select('.plot-area').selectAll('div.row').data(xPlaceholder);
  const rEnter = rows.enter().append('div').attr('class', 'row');
  rows.exit().remove();

  const cols = rEnter.selectAll('div.col').data(yPlaceholder);
  cols.enter()
      .append('div')
      .attr('class', 'col')
      .append('canvas')
      .attr('width', 50)
      .attr('height', 50);

  // 이미지를 생성하고 캔바스 원소에 그립니다.
  rows.merge(rEnter).each(async function(rowZ, rowIndex) {
    // 각 행에 대한 z 벡터의 배치를 생성합니다.
    const zX = xAxis.slice(rowIndex, 1).tile(yAxis.shape);
    const zBatch = zX.stack(yAxis).transpose();
    const batchImageTensor = decodeZ(zBatch);
    const imageTensors = batchImageTensor.unstack();

    tf.dispose([zX, zBatch, batchImageTensor]);

    const cols = d3.select(this).selectAll('.col');
    cols.each(async function(colZ, colIndex) {
      const canvas = d3.select(this).select('canvas').node();
      const imageTensor = imageTensors[colIndex];

      // 결과를 캔바스에 그립니다.
      tf.browser.toPixels(imageTensor, canvas).then(() => {
        tf.dispose([imageTensor]);
      });
    });
  });
}

function getParams() {
  const ppd = document.getElementById('pointsPerDim');
  const start = document.getElementById('start');
  const end = document.getElementById('end');

  return {
    pointsPerDim: parseInt(ppd.value), start: parseFloat(start.value),
        end: parseFloat(end.value),
  }
}

/**
 * 일정한 간격으로 떨어진 2d 잠재 공간을 생성합니다.
 */
function draw() {
  const params = getParams();
  console.log('params', params);
  const latentSpace = generateLatentSpace(
      LATENT_DIMS, params.pointsPerDim, params.start, params.end);

  renderLatentSpace(latentSpace);
  tf.dispose(latentSpace);
}

function setupListeners() {
  document.getElementById('update').addEventListener('click', () => {
    draw();
  })
}

// VAE로 만든 이미지를 그립니다.
(async function run() {
  setupListeners();
  decoder = await loadModel(decoderUrl);
  draw();
})();
