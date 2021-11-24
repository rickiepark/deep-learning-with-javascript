# TensorFlow.js 예제: 생성적 적대 신경망(GAN)

## 예제 소개

이 예제는 MNIST 데이터셋에서 ACGAN을 훈련합니다.

ACGAN에 대해서는 다음을 참고하세요:
 - Augustus Odena, Christopher Olah, Jonathon Shlens. (2017) "Conditional
   image synthesis with auxiliary classifier GANs"
   https://arxiv.org/abs/1610.09585

이 예제의 훈련 스크립트([gan.js](./gan.js))는 케라스 예제를 기반으로 합니다:
  - https://github.com/keras-team/keras/blob/master/examples/mnist_acgan.py

이 TensorFlow.js 예제는 두 환경에서 실행할 수 있습니다:
 - Node.js 환경에서 훈련. 장기간 실행되는 훈련 과정에서 에포크가 끝날 때마다 생성자를 디스크에 저장합니다.
 - 브라우저에서 생성 과정 시연. 데모 웹페이지는 훈련 과정에서 저장된 생성자를 로드하고 이를 사용해 브라우저에서
   가짜 MNIST 이미지를 생성합니다.

## 예제 사용 방법

두 가지 방법으로 사용할 수 있습니다:

1. 로컬 컴퓨터에서 훈련과 생성 데모를 모두 수행하거나
2. 웹에서 원격에 있는 생성 모델을 로딩하여 생성 데모만 수행합니다.

1번의 경우 다음과 같이 훈련을 실행할 수 있습니다:

```sh
yarn
yarn train
```

컴퓨터에 CUDA 지원 GPU가 있다면 `--gpu` 플래그를 추가하여 GPU에서 모델을 훈련할 수 있습니다.
이렇게 하면 훈련 속도를 크게 높일 수 있습니다:

```sh
yarn
yarn train --gpu
```

훈련은 오랜 시간이 걸리며 (@tensorflow/tfjs-node-gpu를 사용해) GPU에서 완료하는데 몇 시간이 걸립니다.
(@tensorflow/tfjs-node를 사용하는) CPU엣j는 더 오래 걸립니다.
훈련을 시작할 때와 훈련 에포크가 끝날 때마다 ACGAN의 생성자를 `./dist/generator` 폴더에 저장합니다.
모델과 함께 일부 메타데이터도 저장합니다.

### TensorBoard를 사용하여 GAN 훈련 모니터링하기

Node.js 기반 훈련 스크립트를 사용하면 생성자와 판별자의 손실 값을
[텐서보드](https://www.tensorflow.org/guide/summaries_and_tensorboard)에 기록할 수 있습니다.
훈련 스크립트의 기본값인 콘솔에 손실 값을 출력하는 것에 비해
텐서보드에 로그를 기록하면 다음과 같은 장점이 있습니다:

1. 손실 값을 영구 기록하면 훈련 도중 어떤 이유로 시스템에 장애가 생기더라도 훈련 이력을 유지할 수 있습니다.
   콘솔에 로그를 기록하면 시스템 장애시 모두 사라집니다.
2. 손실 값을 그래프로 시각화하면 트렌드를 쉽게 파악할 수 있습니다(아래 스크린샷 참고).

![MNIST ACGAN Training: TensorBoard Example](./mnist-acgan-tensorboard-example.png)

자세한 손실 그래프는
[TensorBoard.dev](https://tensorboard.dev/experiment/iBcGONlbQbmVyNd8H6unJg/#scalars)에서
볼 수 있습니다.

이렇게 예제를 실행하려면 `yarn train` 명령에 `--logDir` 플래그를 추가하고 로그를 기록하려는
디렉토리를 지정합니다. 예를 들면 다음과 같습니다.

```sh
yarn train --gpu --logDir /tmp/mnist-acgan-logs
```

그다음 텐서보드를 설치하고 로그 디렉토리를 지정하여 실행합니다:

```sh
# 텐서보드가 이미 설치되어 있다면 건너 뜁니다.
pip install tensorboard

tensorboard --logdir /tmp/mnist-acgan-logs
```

텐서보드가 터미널에 URL을 출력할 것입니다.
브라우저를 열고 이 URL에 접속하면 텐서보드의 Scalar 대시보드에서 손실 곡선을 볼 수 있습니다.

### 브라우저에서 생성 데모 실행하기

브라우저에서 데모를 실행하려면 별도의 터미널에서 다음 명령을 실행합니다:

```sh
yarn
npx http-server
```

브라우저 데모가 시작되면 `./generator` 폴더에서 생성자 모델과 메타데이터를 로드합니다.
로드에 성공하면 이 생성자를 사용하여 가짜 MNIST 숫자를 생성하고 바로 브라우저에 출력합니다.
(가령 아직 모델을 훈련하지 않아) 모델 로드에 실패하면 "원격 모델 로드하기" 버튼을 클릭하여
원격에 있는 모델을 로드할 수 있습니다.

### tfjs-node-gpu를 사용해 CUDA GPU에서 모델 훈련하기

tfjs-node를 사용하여 CPU에서 훈련하는 것보다 GPU를 사용하면 합성곱 연산이 몇 배 빠르기 때문에
tfjs-node-gpu를 사용해 CUDA 가능 GPU에서 모델을 훈련하는 것이 권장됩니다.

기본적으로 [훈련 스크립트](./gan.js)는 tfjs-node를 사용해 CPU에서 실행됩니다.
GPU에서 실행하려면 다음 코드를

```js
require('@tensorflow/tfjs-node');
```

다음과 같이 바꾸세요.

```js
require('@tensorflow/tfjs-node-gpu');
```
