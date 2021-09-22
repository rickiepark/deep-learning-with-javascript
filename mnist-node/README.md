# TensorFlow.js 예제: Node.js로 MNIST 모델 훈련하기

이 예제는 Node.js로 (layers API를 사용한) MNIST 모델을 훈련하는 방법을 보여줍니다.

이 모델은 훈련 데이터셋(60,000개 샘플)을 한 번 처리한 후 에포크마다 정확도를 계산하기 위해 테스트 데이터셋 중에서 50개의 샘플을 평가합니다.

node 환경을 준비합니다:
```sh
$ npm install
# 또는
$ yarn
```

훈련 스크립트를 실행합니다:
```sh
$ node main.js
```

[CUDA 호환](https://www.tensorflow.org/install/install_linux) 리눅스에서 실행한다면,
GPU 패키지를 설치하고 require 문을 다음처럼 교체하세요:

```sh
$ npm install @tensorflow/tfjs-node-gpu
# 또는
$ yarn add @tensorflow/tfjs-node-gpu
```

패키지를 설치한 후에 main.js 파일에서 `require('@tensorflow/tfjs-node')`를 `require('@tensorflow/tfjs-node-gpu');`로 교체하세요.
