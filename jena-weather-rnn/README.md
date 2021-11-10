# TensorFlow.js 예제: 예나 날씨 데이터 시각화와 예측

이 데모에 포함된 내용은 다음과 같습니다.
- [tfjs-vis](https://www.npmjs.com/package/@tensorflow/tfjs-vis) 라이브러리를 사용하여 시계열 데이터 시각화하기
- 여러 종류의 모델을 사용해 순차 입력 데이터를 기반으로 미래의 값을 예측하기
  - 선형 회귀
  - 다층 퍼셉트론 (MLP)
  - 순환 신경망 (RNN)
- 과소적합, 과대적합, 과대적합을 줄이기 위한 여러가지 기법
  - L2 규제
  - 드롭아웃
  - 순환 드롭아웃

이 데모에 사용된 데이터는 [예나 날씨 데이터셋](https://www.kaggle.com/pankrzysiu/weather-archive-jena)입니다.

이 예제는 TensorFlow.js에서 다음 API를 사용합니다.

- `tf.data.generator()`: 제너레이터 함수에서 `tf.data.Dataset` 객체를 생성합니다.
- `tf.Model.fitDataset()`: `tf.data.Dataset` 객체를 사용해 `tf.Model`을 훈련하고
  또 다른 `tf.data.Dataset` 객체를 사용해 훈련 에포크가 끝날 때마다 모델의 검증 성능을 평가합니다.
- `tfvis.show.fitCallbacks()`: 모델 훈련에서 배치와 에포크가 끝날 때마다
  훈련 세트와 검증 세트의 손실을 그래프로 그립니다.

## Node.js에서 RNN 훈련하기

이 예제는 선형 회귀, 다층 퍼셉트론, 순환 신경망 등을 사용해 온도를 예측하는 방법을 보여줍니다.
처음 두 개의 모델은 브라우저에서 훈련하지만 RNN 훈련은 계산 부하가 크고 훈련 시간이 오래 걸리기 때문에
Node.js에서 수행합니다.

예를 들어 GRU 모델을 훈련하려면 다음과 같은 명령을 사용합니다:

```sh
yarn
yarn train-rnn
```

기본적으로 tfjs-node의 Eigen 연산을 사용해 CPU에서 훈련됩니다. CUDA 가능 GPU를 가지고 있고
필요한 드라이버와 라이브러리(CUDA와 CuDNN)가 설치되어 있다면 tfjs-node-gpu에서 CUDA/CuDNN을 사용해
모델을 훈련할 수 있습니다. 이렇게 하려면 `--gpu` 플래그를 추가하면 됩니다:

```sh
yarn
yarn train-rnn --gpu
```

또한 머신러닝을 사용하지 않는 방법을 기반으로 예측 오류(평균 절댓값 오차)를 계산할 수 있습니다.
입력 특성의 마지막 온도 데이터를 사용하는 방법입니다. 이 값을 계산하려면 `--modelType` 플래그를
`baseline`으로 지정합니다.

```sh
yarn
yarn train-rnn --modelType baseline
```

### 텐서보드로 Node.js 훈련 모니터링하기

Node.js 기반 훈련 스크립트는 모델의 손실 값을 텐서보드로 기록할 수 있습니다. 기본적으로 훈련 스크립트는
콘솔에 손실 값을 출력합니다. 이에 비해 텐서보드에 로깅을 하면 다음과 같은 잇점이 있습니다:

1. 손실 값이 저장되기 때문에 훈련 도중 어떤 이유로 시스템에 장애가 있더라도 훈련 기록을 유지할 수 있습니다.
   반면 이런 경우 콘솔 로그는 사라집니다.
2. 손실 값을 그래프로 시각화하면 트렌드를 쉽게 파악할 수 있습니다.
3. 텐서보드 HTTP 서버를 통해 원격 컴퓨터에서 훈련을 모니터링할 수 있습니다.

이 예제에서 텐서보드를 사용하려면 `--logDir` 플래그에 로그를 기록할 디렉토리를 지정합니다.

```sh
yarn train-rnn --gpu --logDir /tmp/jena-weather-logs-1
```

그다음 텐서보드를 설치하고 이 디렉토리를 지정하여 실행합니다:

```sh
# 텐서보드가 이미 설치되어 있다면 이 단계를 건너 뛰세요.
pip install tensorboard

tensorboard --logdir /tmp/jena-weather-logs-1
```

텐서보드는 HTTP URL을 터미널에 출력합니다. 브라우저를 열고 이 URL에 접속하면 텐서보드의
스칼라 대시보드에서 손실 곡선을 볼 수 있습니다.
