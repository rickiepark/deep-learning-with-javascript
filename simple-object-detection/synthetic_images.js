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
 * simple-object-detection 모델을 훈련하고 테스트하는데 사용하기 위한 이미지 합성 모듈
 *
 * 이 모듈은 Node.js 기반 훈련 파이프라인(train.js)과 브라우저 기반의 테스트 환경(index.js)에서
 * 모두 사용할 수 있도록 만들었습니다.
 */

let tf;  // 브라우저/노드 호환성을 위해 전달된 tensorflowjs 모듈

/**
 * 캔바스 스트로크와 색 채우기를 위한 랜덤한 컬러 스타일을 생성합니다.
 *
 * @returns {string} 'rgb(100,200,250)' 형태의 스타일 문자열
 */
function generateRandomColorStyle() {
  const colorR = Math.round(Math.random() * 255);
  const colorG = Math.round(Math.random() * 255);
  const colorB = Math.round(Math.random() * 255);
  return `rgb(${colorR},${colorG},${colorB})`;
}

/**
 * 간단한 객체 인식을 위한 이미지를 합성합니다.
 *
 * 합성 이미지는 다음과 같은 요소로 구성됩니다.
 * - 흰 배경
 * - 랜덤한 크기와 색깔의 원으로 개수는 설정 가능합니다.
 * - 시작과 끝 위치가 랜덤하고 색깔이 랜덤한 선분으로 개수는 설정 가능합니다.
 * - 타깃 객체: 지정된 확률에 따라 직사각형 또는 삼각형
 *   - 직사각형이면 변 길이와 색깔이 랜덤합니다.
 *   - 삼각형이면 항상 정삼각형입니다. 변 길이와 색깔이 랜덤하고 랜덤한 각도로 회전되어 있습니다.
 */
export class ObjectDetectionImageSynthesizer {
  /**
   * ObjectDetectionImageSynthesizer의 생성자
   *
   * @param {} canvas HTML 캔버스 객체 또는 node 캔버스 객체
   * @param {*} tensorFlow 전달된 텐서플로 모듈. 이는 브라우저와 Node.js 간의 호환성을 위한 것입니다.
   */
  constructor(canvas, tensorFlow) {
    this.canvas = canvas;
    tf = tensorFlow;

    // 원 반지름의 최소, 최댓값
    this.CIRCLE_RADIUS_MIN = 5;
    this.CIRCLE_RADIUS_MAX = 20;
    // 직사각형의 변 길이의 최소, 최댓값
    this.RECTANGLE_SIDE_MIN = 40;
    this.RECTANGLE_SIDE_MAX = 100;
    // 삼각형 변 길이의 최소, 최댓값
    this.TRIANGLE_SIDE_MIN = 50;
    this.TRIANGLE_SIDE_MAX = 100;

    // 캔버스 크기
    this.w = this.canvas.width;
    this.h = this.canvas.height;
  }

  /**
   * 하나의 이미지 샘플을 생성합니다.
   *
   * @param {number} numCircles 포함시킬 (배경 객체인) 원 개수
   * @param {number} numLines 포함시킬 (배경 객체인) 선분 개수
   * @param {number} triangleProbability 타깃 개체가 (직사각형이 아니라) 삼각형이 될 확률.
   *   이 숫자는 0보다 크거나 같고 1보다 작거나 같아야 합니다. 기본값은 0.5입니다.
   * @returns {Object} 다음 항목을 가진 객체.
   *   - image: 이미지의 픽셀 내용을 담은 [w, h, 3] 크기의 텐서.
   *     w와 h는 캔버스의 너비와 높이입니다.
   *   - target: [5] 크기의 텐서.
   *     첫 번째 원소는 타깃이 삼각형(0)인지 직사각형(1)인지 나타냅니다.
   *     남은 네 개의 원소는 도형의 바운딩 박스입니다. 픽셀 단위로 [왼쪽, 오른쪽, 위, 아래] 크기입니다.
   */
  async generateExample(numCircles, numLines, triangleProbability = 0.5) {
    if (triangleProbability == null) {
      triangleProbability = 0.5;
    }
    tf.util.assert(
        triangleProbability >= 0 && triangleProbability <= 1,
        `triangleProbability는 0보다 크거나 같고 1보다 작거나 같아야 합니다. 하지만 ` +
            `${triangleProbability}를 입력했습니다.`);

    const ctx = this.canvas.getContext('2d');
    ctx.clearRect(0, 0, this.w, this.h);  // 캔버스 클리어

    // 원 크리기 (절반)
    for (let i = 0; i < numCircles / 2; ++i) {
      this.drawCircle(ctx);
    }

    // 선분 그리기 (절반)
    for (let i = 0; i < numLines / 2; ++i) {
      this.drawLineSegment(ctx);
    }

    // 타깃 객체 그리기: 직사각형 또는 정삼각형
    // 타깃이 직사각형인지 삼각형인지 결정합니다.
    const isRectangle = Math.random() > triangleProbability;

    let boundingBox;
    ctx.fillStyle = generateRandomColorStyle();
    ctx.beginPath();
    if (isRectangle) {
      // 직사각형을 그립니다.
      // 직사각형의 두 변의 길이는 랜덤하며 서로 독립적입니다.
      const rectangleW =
          Math.random() * (this.RECTANGLE_SIDE_MAX - this.RECTANGLE_SIDE_MIN) +
          this.RECTANGLE_SIDE_MIN;
      const rectangleH =
          Math.random() * (this.RECTANGLE_SIDE_MAX - this.RECTANGLE_SIDE_MIN) +
          this.RECTANGLE_SIDE_MIN;
      const centerX = (this.w - rectangleW) * Math.random() + (rectangleW / 2);
      const centerY = (this.h - rectangleH) * Math.random() + (rectangleH / 2);
      boundingBox =
          this.drawRectangle(ctx, centerX, centerY, rectangleH, rectangleW);
    } else {
      // 랜덤한 각도로 회전한 정삼각형을 그립니다.
      // 정삼각형의 중심에서 세 개의 꼭지점까지 거리.
      const side = this.TRIANGLE_SIDE_MIN +
          (this.TRIANGLE_SIDE_MAX - this.TRIANGLE_SIDE_MIN) * Math.random();
      const centerX = (this.w - side) * Math.random() + (side / 2);
      const centerY = (this.h - side) * Math.random() + (side / 2);
      // 0~120도 사이의 균등 분포에서 선택한 랜덤한 각도로 정삼각형을 회전합니다.
      const angle = Math.PI / 3 * 2 * Math.random();  // 0 - 120도.
      boundingBox = this.drawTriangle(ctx, centerX, centerY, side, angle);
    }
    ctx.fill();

    // 원 그리기 (나머지 절반)
    for (let i = numCircles / 2; i < numCircles; ++i) {
      this.drawCircle(ctx);
    }

    // 선분 그리기 (나머지 절반)
    for (let i = numLines / 2; i < numLines; ++i) {
      this.drawLineSegment(ctx);
    }

    return tf.tidy(() => {
      const imageTensor = tf.browser.fromPixels(this.canvas);
      const shapeClassIndicator = isRectangle ? 1 : 0;
      const targetTensor =
          tf.tensor1d([shapeClassIndicator].concat(boundingBox));
      return {image: imageTensor, target: targetTensor};
    });
  }

  drawCircle(ctx, centerX, centerY, radius) {
    centerX = centerX == null ? this.w * Math.random() : centerX;
    centerY = centerY == null ? this.h * Math.random() : centerY;
    radius = radius == null ? this.CIRCLE_RADIUS_MIN +
            (this.CIRCLE_RADIUS_MAX - this.CIRCLE_RADIUS_MIN) * Math.random() :
                              radius;

    ctx.fillStyle = generateRandomColorStyle();
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  drawLineSegment(ctx, x0, y0, x1, y1) {
    x0 = x0 == null ? Math.random() * this.w : x0;
    y0 = y0 == null ? Math.random() * this.h : y0;
    x1 = x1 == null ? Math.random() * this.w : x1;
    y1 = y1 == null ? Math.random() * this.h : y1;

    ctx.strokeStyle = generateRandomColorStyle();
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }

  /**
   * 직사각형을 그립니다.
   *
   * 간단한 객체 탐지 예제에서 사각형은 타깃 객체이므로 바운딩 박스를 반환합니다.
   *
   * @param {} ctx  캔바스 컨택스트
   * @param {number} centerX 직사각형 중심의 x 좌표
   * @param {number} centerY 직사각형 중심의 y 좌표
   * @param {number} w 직사각형의 너비
   * @param {number} h 직사각형의 높이
   * @returns {[number, number, number, number]} 직사각형의 바운딩 박스:
   *   [왼쪽, 오른쪽, 위, 아래]
   */
  drawRectangle(ctx, centerX, centerY, w, h) {
    ctx.moveTo(centerX - w / 2, centerY - h / 2);
    ctx.lineTo(centerX + w / 2, centerY - h / 2);
    ctx.lineTo(centerX + w / 2, centerY + h / 2);
    ctx.lineTo(centerX - w / 2, centerY + h / 2);

    return [centerX - w / 2, centerX + w / 2, centerY - h / 2, centerY + h / 2];
  }

  /**
   * 삼각형을 그립니다.
   *
   * 간단한 객체 탐지 예제에서 삼각형은 타깃 객체이므로 바운딩 박스를 반환합니다.
   *
   * @param {} ctx  캔바스 컨택스트
   * @param {number} centerX 삼각형 중심의 x 좌표
   * @param {number} centerY 삼각형 중심의 x 좌표
   * @param {number} side 변의 길이
   * @param {number} angle 삼각형 회전 각도(라디안)
   * @returns {[number, number, number, number]} 회전을 고려한 삼각형의 바운딩 박스:
   *   [왼쪽, 오른쪽, 위, 아래].
   */
  drawTriangle(ctx, centerX, centerY, side, angle) {
    const ctrToVertex = side / 2 / Math.cos(30 / 180 * Math.PI);
    ctx.fillStyle = generateRandomColorStyle();
    ctx.beginPath();

    const alpha1 = angle + Math.PI / 2;
    const x1 = centerX + Math.cos(alpha1) * ctrToVertex;
    const y1 = centerY + Math.sin(alpha1) * ctrToVertex;
    const alpha2 = alpha1 + Math.PI / 3 * 2;
    const x2 = centerX + Math.cos(alpha2) * ctrToVertex;
    const y2 = centerY + Math.sin(alpha2) * ctrToVertex;
    const alpha3 = alpha2 + Math.PI / 3 * 2;
    const x3 = centerX + Math.cos(alpha3) * ctrToVertex;
    const y3 = centerY + Math.sin(alpha3) * ctrToVertex;

    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x3, y3);

    const xs = [x1, x2, x3];
    const ys = [y1, y2, y3];
    return [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  }

  /**
   * 배치 샘플을 생성합니다.
   *
   * @param {number} batchSize 배치에 있는 샘플 이미지의 개수
   * @param {number} numCircles 포함할 원 개수 (배경 객체)
   * @param {number} numLines 포함할 선분 객체 (배경 객체)
   * @returns {Object} 다음 조건을 가진 객체An object with the following fields:
   *   - image: 이미지의 픽셀 컨텐츠를 담은 [batchSize, w, h, 3] 크기의 텐서.
   *     w와 h는 캔바스의 너비와 높이입니다.
   *   - target: [batchSize, 5] 크기의 텐서. 첫 번째 열은 타깃이 삼각형(0) 또는 직사각형(1)인지
   *     나타내는 0-1 표시자입니다. 남은 네 열은 도형의 바운딩 박스입니다(픽셀 단위):
   *     [왼쪽, 오른쪽, 위, 아래]
   */
  async generateExampleBatch(
      batchSize, numCircles, numLines, triangleProbability) {
    if (triangleProbability == null) {
      triangleProbability = 0.5;
    }
    const imageTensors = [];
    const targetTensors = [];
    for (let i = 0; i < batchSize; ++i) {
      const {image, target} =
          await this.generateExample(numCircles, numLines, triangleProbability);
      imageTensors.push(image);
      targetTensors.push(target);
    }
    const images = tf.stack(imageTensors);
    const targets = tf.stack(targetTensors);
    tf.dispose([imageTensors, targetTensors]);
    return {images, targets};
  }
}
