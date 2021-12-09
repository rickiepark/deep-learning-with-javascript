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

// 이 스크립트가 추가할 모든 텍스트 노드의 이름
const TEXT_DIV_CLASSNAME = 'tfjs_mobilenet_extension_text';
// 어떤 메시지를 출력할지 제어하는 LOW_CONFIDENCE_THRESHOLD와 HIGH_CONFIDENCE_THRESHOLD를 위한 임곗값
const HIGH_CONFIDENCE_THRESHOLD = 0.5;
const LOW_CONFIDENCE_THRESHOLD = 0.1;

/**
 * 예측을 요약하는 짧은 문자열을 만듭니다.
 * 입력은 {className: string, prediction: float} 객체의 리스트입니다.
 * @param {[{className: string, predictions: number}]} predictions 예측 클래스와 점수로 구성된 객체의 정렬된 리스트
 */
function textContentFromPrediction(predictions) {
  if (!predictions || predictions.length < 1) {
    return `예측이 없습니다 🙁`;
  }
  // 확신함
  if (predictions[0].probability >= HIGH_CONFIDENCE_THRESHOLD) {
    return `😄 ${predictions[0].className}!`;
  }
  // 확신 없음
  if (predictions[0].probability >= LOW_CONFIDENCE_THRESHOLD &&
      predictions[0].probability < HIGH_CONFIDENCE_THRESHOLD) {
    return `${predictions[0].className}?...\n 아마도 ${
        predictions[1].className}?`;
  }
  // 거의 확신하지 못함
  if (predictions[0].probability < LOW_CONFIDENCE_THRESHOLD) {
    return `😕  ${predictions[0].className}????...\n 아마도 ${
        predictions[1].className}????`;
  }
}

/**
 * src URL에 지정된 모든 DOM 이미지 요소를 반환합니다.
 * @param {string} srcUrl 'http(s)://'를 포함하여 탐색할 url
 * @returns {HTMLElement[]} srcUrl에서 지정한 모든 img 요소
 */
function getImageElementsWithSrcUrl(srcUrl) {
  const imgElArr = Array.from(document.getElementsByTagName('img'));
  const filtImgElArr = imgElArr.filter(x => x.src === srcUrl);
  return filtImgElArr;
}

/**
 * 이 확장 프로그램이 추가한 모든 텍스트 예측을 찾아 삭제합니다. 그리고 DOM에서 제거합니다.
 */
function removeTextElements() {
  const textDivs = document.getElementsByClassName(TEXT_DIV_CLASSNAME);
  for (const div of textDivs) {
    div.parentNode.removeChild(div);
  }
}

/**
 * imgNode를 콘테이너 div 안으로 이동하고 텍스트 div를 추가합니다.
 * 이미지 위에 텍스트를 쓰기 위해 콘테이너 div와 텍스트 div 스타일을 조정합니다.
 * @param {HTMLElement} imgNode 콘텐츠를 쓸 이미지 노드
 * @param {string} textContent 이미지에 쓸 텍스트
 */
function addTextElementToImageNode(imgNode, textContent) {
  const originalParent = imgNode.parentElement;
  const container = document.createElement('div');
  container.style.position = 'relative';
  container.style.textAlign = 'center';
  container.style.color = 'white';
  const text = document.createElement('div');
  text.className = 'tfjs_mobilenet_extension_text';
  text.style.position = 'absolute';
  text.style.top = '50%';
  text.style.left = '50%';
  text.style.transform = 'translate(-50%, -50%)';
  text.style.fontSize = '34px';
  text.style.fontFamily = 'Google Sans,sans-serif';
  text.style.fontWeight = '700';
  text.style.color = 'white';
  text.style.lineHeight = '1em';
  text.style['-webkit-text-fill-color'] = 'white';
  text.style['-webkit-text-stroke-width'] = '1px';
  text.style['-webkit-text-stroke-color'] = 'black';
  // 이미지 바로 옆에 콘테이너 노드를 추가합니다.
  originalParent.insertBefore(container, imgNode);
  // 이미지 노드를 콘테이너 노드 안으로 옮깁니다.
  container.appendChild(imgNode);
  // 이미지 노드 다음에 텍스트 노드를 추가합니다.
  container.appendChild(text);
  text.textContent = textContent;
}

// 이미지가 처리될 때 content.js 페이지에서 듣기 위한 리스너를 추가합니다.
// 메시지는 action, url, prediction(분류기 출력)을 포함해야 합니다.
//
// message: {action, url, predictions}
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message && message.action === 'IMAGE_CLICK_PROCESSED' && message.url &&
      message.predictions) {
    // url에 해당하는 이미지 목록을 가져옵니다.
    const imgElements = getImageElementsWithSrcUrl(message.url);
    for (const imgNode of imgElements) {
      const textContent = textContentFromPrediction(message.predictions);
      addTextElementToImageNode(imgNode, textContent);
    }
  }
});

// 사용자가 왼쪽 마우스 버튼을 클릭하면 모든 주석을 제거하기 위한 리스너를 준비합니다.
// 그렇지 않으면 쉽게 윈도우가 복잡해 집니다.
window.addEventListener('click', clickHandler, false);
/**
 * 왼쪽 클릭할 때 DOM에서 텍스트 요소를 제거합니다.
 */
function clickHandler(mouseEvent) {
  if (mouseEvent.button == 0) {
    removeTextElements();
  }
}
