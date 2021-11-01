# TensorFlow.js 예제: 합성곱 신경망 시각화

## 설명

이 TensorFlow.js 예제는 합성곱 신경망의 내부 동작을 시각화하는 몇 가지 기법을 시연합니다:

- 훈련한 합성곱 층의 필터가 민감한 입력 찾기: 입력 공간에 경사 상승법을 적용하여 합성곱 필터를
  가장 크게 활성화하는 입력 이미지를 찾습니다.
- TensorFlow.js의 함수형 API를 사용해 합성곱 신경망의 내부 활성화 출력을 구합니다.
- 그레이디언트 기반의 클래스 활성화 맵(class activation map, CAM) 방식을 사용해 입력 이미지의 어느 부분이
  합성곱 신경망(여기에서는 VGG16)의 분류 결정에 가장 크게 관련되어 있는지 찾습니다.

## 데모 사용 방법

다음 명령을 실행합니다:
```sh
yarn visualize
```

다음 작업이 자동으로 수행됩니다.

1. 필요한 파이썬 라이브러리를 설치합니다. 필요한 파이썬 패키지(keras, tensorflow, tensorflowjs)가
   이미 설치되어 있다면 아무런 작업을 수행하지 않습니다. 하지만 이 단계가 전역 파이썬 환경을 바꾸지 않게 하려면
   이 데모를 [virtualenv](https://virtualenv.pypa.io/en/latest/)나
   [pipenv](https://pipenv.readthedocs.io/en/latest/)에서 실행하세요.
2. VGG16 모델을 다운로드하여 TensorFlow.js 포맷으로 변환합니다.
3. Node.js 스크립트를 구동하여 변환된 모델을 로드하고 경사 상승법을 사용하여 입력 공간에서
   합성곱 신경망의 필터를 최대로 활성화하는 입력 이미지를 계산하고 `dist/filters` 디렉토리에
   이미지 파일로 저장합니다.
4. Node.js 스크립트를 구동하여 합성곱 층의 내부 활성화와 그레디디언트 기반의 클래스 활성화 맵을 계산하고
   `dist/activation` 디렉토리 아래에 이미지 파일로 저장합니다.
5. parcel을 사용해 웹 페이지를 컴파일하고 페이지를 띄웁니다.

단계 3과 4는 (특히 단계 3) 비교적 계산량이 많기 때문에 tfjs-node 대신에 tfjs-node-gpu를
사용하는 것이 좋습니다. CUDA 가능 GPU가 필요하고 관련된 드라이버와 라이브러리를 컴퓨터에 설치해야 합니다.

이런 조건을 만족하면 다음처럼 실행할 수 있습니다:

```sh
yarn visualize --gpu
```

합성곱 층마다 시각화할 필터 개수를 기본값 8에서 더 크게(예를 들어 32) 늘릴 수 있습니다:

```sh
yarn visualize --gpu --filters 32
```

내부 활성화와 CAM 시각화에 사용하는 이미지는 기본적으로 'cat.jpg'입니다. "--image" 플래그를
사용해 다른 이미지로 바꿀 수 있습니다.

```sh
yarn visualize --image dog.jpg
```
