# TensorFlow.js 예제: 간단한 객체 탐지

이 예제는 TensorFlow.js로 간단한 객체 탐지를 수행하는 모델을 어떻게 훈련하는지 보여줍니다.
전체 워크플로는 다음과 같습니다.

- 훈련과 테스트를 위해 합성 이미지와 레이블 생성
- 객체 탐지 작업을 위해 사전 훈련된 컴퓨터 비전 모델(MobileNet) 기반의 모델 구축
- [tfjs-node](https://github.com/tensorflow/tfjs-node)를 사용해 Node.js에서 모델 훈련
- Node.js 환경에서 훈련한 모델을 저장하고 브라우저에서 로드하여 변환하기
- 브라우저에서 로드된 모델로 추론 수행하고 결과를 시각화하기

## 예제 사용 방법

먼저, Node.js를 사용해 모델을 훈련합니다.

```sh
yarn
yarn train
```

그 다음 브라우저에서 모델을 실행합니다.

```sh
npx http-server
```

지금 바로 [온라인 데모](http://ml-ko.kr/tfjs/simple-object-detection/)를 확인할 수도 있습니다!

## "yarn train" 커맨드 라인의 훈련 옵션

`yarn train` 명령은 모든 훈련 샘플을 메모리에 저장합니다.
따라서 메모리 부족 현상이 일어날 수 있고 너무 많은 훈련 샘플이 생성되면 시스템이 중지될 수 있습니다.
하지만 많은 개수의 훈련 샘플을 사용하면 모델의 정확도에 도움이 됩니다.
기본 샘플 개수는 2000입니다. `yarn train` 명령의 `--numExamples` 옵션을 사용해 샘플 개수를 바꿀 수 있습니다.
예를 들어 호스팅된 모델은 다음과 같은 명령을 사용해 20000개의 샘플로 훈련되었습니다.

```sh
yarn train \
    --numExamples 20000 \
    --initialTransferEpochs 100 \
    --fineTuningEpochs 200
```

다른 옵션에 대해서는 `train.js`를 참고하세요.

### CUDA GPU를 사용해 훈련하기

기본적으로 tfjs-node의 CPU 버전을 사용해 모델을 훈련합니다.
컴퓨터에 CUDA GPU가 있다면 tfjs-node-gpu를 사용해 훈련 시간을 크게 단축시킬 수 있습니다.
구체적으로 다음 명령처럼 `--gpu` 옵션을 추가합니다.

```sh
yarn train --gpu \
    --numExamples 20000 \
    --initialTransferEpochs 100 \
    --fineTuningEpochs 200
```

### 텐서보드로 모델 훈련 모니터링하기

Node.js 기반 훈련 스크립트로 손실 값을 [텐서보드](https://www.tensorflow.org/guide/summaries_and_tensorboard)(TensorBoard)에 기록할 수 있습니다. 훈련 스크립트가 기본적으로 콘솔에 손실 값을 출력하는 것과 비교하면 텐서보드 로깅은 다음과 같은 장점이 있습니다.

1. 손실 값을 저장하기 때문에 훈련 도중에 어떤 이유로 시스템이 망가지더라도 훈련 기록을 남길 수 있습니다.
하지만 콘솔 로그는 수명이 짧습니다.
2. 손길 값을 그래프로 시각화하면 트렌드를 더 쉽게 확인할 수 있습니다.

이 예제에 적용하려면 `yarn train` 명령에 `--logDir` 옵션을 추가하고 로그를 저장할 디렉토리를 지정합니다.

 ```sh
yarn train
    --numExamples 20000 \
    --initialTransferEpochs 100 \
    --fineTuningEpochs 200 \
    --logDir /tmp/simple-object-detection-logs
```

그다음 텐서보드를 설치하고 이 로그 디렉토리를 지정하여 시작합니다.

 ```sh
# Skip this step if you have already installed tensorboard.
pip install tensorboard
tensorboard --logdir /tmp/simple-object-detection-logs
```

텐서보드가 터미널에 HTTP URL을 출력합니다.
브라우저를 열고 이 URL에 접속하여 텐서보드의 스칼라(Scalar) 대시보드에서 손실 곡선을 확인할 수 있습니다.
