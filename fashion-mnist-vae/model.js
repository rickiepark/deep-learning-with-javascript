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
 * 이 파일은 변이형 오토인코더 기반의 다층 퍼셉트론을 구현합니다.
 * 다음 코드를 참고했습니다.
 * https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
 *
 * 오토인코더의 동작 방식에 대한 설명은 다음 튜토리얼을 참고하세요.
 * https://blog.keras.io/building-autoencoders-in-keras.html
 */

const tf = require('@tensorflow/tfjs');

/**
 * VAE 모델의 인코더
 *
 * @param {object} opts 다음 필드를 포함한 인코더 설정
 *   - originaDim {number} 펼친 입력 이미지의 길이
 *   - intermediateDim {number} 중간 (은닉) 밀집 층의 유닛 개수
 *   - latentDim {number} 잠재 공간의 차원 (즉, z-공간)
 * @returns {tf.LayersModel} 인코더 모델
 */
function encoder(opts) {
  const {originalDim, intermediateDim, latentDim} = opts;

  const inputs = tf.input({shape: [originalDim], name: 'encoder_input'});
  const x = tf.layers.dense({units: intermediateDim, activation: 'relu'})
                .apply(inputs);
  const zMean = tf.layers.dense({units: latentDim, name: 'z_mean'}).apply(x);
  const zLogVar =
      tf.layers.dense({units: latentDim, name: 'z_log_var'}).apply(x);

  const z =
      new ZLayer({name: 'z', outputShape: [latentDim]}).apply([zMean, zLogVar]);

  const enc = tf.model({
    inputs: inputs,
    outputs: [zMean, zLogVar, z],
    name: 'encoder',
  });

  // console.log('Encoder Summary');
  // enc.summary();
  return enc;
}

/**
 * 이 층은 다음 페이지에서 언급된 재파라미터화 트릭을 구현합니다.
 * https://blog.keras.io/building-autoencoders-in-keras.html.
 *
 * call 메서드에 구현되어 있습니다.
 * Q(z|X)에서 샘플링하는 대신:
 *    sample epsilon = N(0,I)
 *    z = z_mean + sqrt(var) * epsilon
 */
class ZLayer extends tf.layers.Layer {
  constructor(config) {
    super(config);
  }

  computeOutputShape(inputShape) {
    tf.util.assert(inputShape.length === 2 && Array.isArray(inputShape[0]),
        () => `입력 크기는 정확히 2여야 합니다. 현재 입력 크기: ${inputShape}`);
    return inputShape[0];
  }

  /**
   * ZLayer 객체가 수행하는 실제 연산.
   *
   * @param {Tensor[]} inputs 이 층은 z_mean와 z_log_var 두 개 텐서를 받습니다.
   * @return z_mean, z_log_var와 같은 크기의 텐서로
   *     z_mean + sqrt(exp(z_log_var)) * epsilon를 계산한 값입니다.
   *     여기서 epsilon은 표준 정규 분포(N(0, I))를 따르는 벡터입니다.
   */
  call(inputs, kwargs) {
    const [zMean, zLogVar] = inputs;
    const batch = zMean.shape[0];
    const dim = zMean.shape[1];

    const mean = 0;
    const std = 1.0;
    // epsilon = N(0, I) 샘플링
    const epsilon = tf.randomNormal([batch, dim], mean, std);

    // z = z_mean + sqrt(var) * epsilon
    return zMean.add(zLogVar.mul(0.5).exp().mul(epsilon));
  }

  static get className() {
    return 'ZLayer';
  }
}
tf.serialization.registerClass(ZLayer);

/**
 * VAE 모델의 디코더
 *
 * @param {*} opts 디코더 설정
 * @param {number} opts.originalDim 원본 데이터의 차원 수
 * @param {number} opts.intermediateDim 중간 층의 유닛 개수
 * @param {number} opts.latentDim 잠재 공간의 차원 수
 */
function decoder(opts) {
  const {originalDim, intermediateDim, latentDim} = opts;

  // 디코더 모델은 선형적인 구조라 `tf.sequential()`으로 만들 수 있습니다.
  // 하지만 인코더 모델(`encoder()` 참고)과 통일성을 위해 함수형 API(즉, `tf.model()`)을 사용합니다.
  const input = tf.input({shape: [latentDim]});
  let y = tf.layers.dense({
    units: intermediateDim,
    activation: 'relu'
  }).apply(input);
  y = tf.layers.dense({
    units: originalDim,
    activation: 'sigmoid'
  }).apply(y);
  const dec = tf.model({inputs: input, outputs: y});

  // console.log('Decoder Summary');
  // dec.summary();
  return dec;
}

/**
 * 인코더-디코더 파이프라인
 *
 * @param {tf.Model} encoder
 * @param {tf.Model} decoder
 *
 * @returns {tf.Model} VAE 모델
 */
function vae(encoder, decoder) {
  const inputs = encoder.inputs;
  const encoderOutputs = encoder.apply(inputs);
  const encoded = encoderOutputs[2];
  const decoderOutput = decoder.apply(encoded);
  const v = tf.model({
    inputs: inputs,
    outputs: [decoderOutput, ...encoderOutputs],
    name: 'vae_mlp',
  })

  // console.log('VAE Summary');
  // v.summary();
  return v;
}

/**
 * VAE를 위한 사용자 손실 함수
 *
 * @param {tf.tensor} inputs 인코더에 입력할 배치 이미지 텐서
 * @param {[tf.tensor]} outputs VAE 출력, [decoderOutput, ...encoderOutputs]
 */
function vaeLoss(inputs, outputs) {
  return tf.tidy(() => {
    const originalDim = inputs.shape[1];
    const decoderOutput = outputs[0];
    const zMean = outputs[1];
    const zLogVar = outputs[2];

    // 먼저 재구성 손실 항을 계산합니다.
    // 이 항을 최소화하는 목적은 모델이 입력 데이터와 같은 출력을 만들게 하는 것입니다.
    const reconstructionLoss =
        tf.losses.meanSquaredError(inputs, decoderOutput).mul(originalDim);

    // 이 대신 binaryCrossEntropy를 사용할 수 있습니다.
    // const reconstructionLoss =
    //  tf.metrics.binaryCrossentropy(inputs, decoderOutput).mul(originalDim);

    // 그다음 zLogVar와 zMean 사이의 KL-발산을 계산합니다.
    // 이 항을 최소화하는 목적은 잠재 변수가 잠재 공간의 중심에서 더 정규 분포를 띠도록 만드는 것입니다.
    let klLoss = zLogVar.add(1).sub(zMean.square()).sub(zLogVar.exp());
    klLoss = klLoss.sum(-1).mul(-0.5);

    return reconstructionLoss.add(klLoss).mean();
  });
}

module.exports = {
  vae,
  encoder,
  decoder,
  vaeLoss,
}
