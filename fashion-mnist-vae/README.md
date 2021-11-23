ㅋㅋㅋㅋㅋㅋ# TensorFlow.js 예제: 변이형 오토인코더

이 예제는 노드에서 TensorFlow.js를 사용해 [변이형 오토인코더](https://blog.keras.io/building-autoencoders-in-keras.html)를 훈련하는 방법을 보여줍니다.

이 모델은 [패션 MNIST](https://github.com/zalandoresearch/fashion-mnist) 데이터셋에서 훈련됩니다.

이 예제는 https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py 에 있는 변이형 오토인코더를 기반으로 하는 다층 퍼셉트론을 참고했습니다. 오토인코더 작동 방식에 대한 설명은 이 [튜토리얼](https://blog.keras.io/building-autoencoders-in-keras.html)를 참고하세요.

## 노드 환경 준비:
```sh
yarn
# 또는
npm install
```
ㅋㅋㅋㅋㅋㅋ
yarn이 설치되어 있지 않다면 아래 명령에서 ```yarn```을 ```npm run```으로 바꿀 수 있습니다.

## 데이터 다운로드

```yarn download-data``` 을 실행하거나 아래와 같이 수동으로 다운로드 할 수 있습니다.

이 [페이지](https://github.com/zalandoresearch/fashion-mnist#get-the-data)에서 [train-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)을 다운로드하세요.

다운로드한 파일의 압축을 풀고 `train-images-idx3-ubyte` 파일을 예제 폴더 안에 있는 `dataset` 폴더에 넣으세요.

## 훈련 스크립트 실행:
```sh
yarn train
```

또는 CUDA를 설치했다면 다음 명령을 사용할 수 있습니다.

```sh
yarn train --gpu
```

에포크가 끝날 때마다 프리뷰 이미지를 출력하고 훈련이 끝나면 모델을 저장합니다.
에포크마다 이미지는 점점 더 의류 아이템과 비슷해져야 합니다.
(일반적인 경우 0에 가까워지는 것과 달리) 훈련이 잘 끝날 경우 손실은 40~50 범위에 있습니다.

[CUDA 호환](https://www.tensorflow.org/install/install_linux) 리눅스 시스템을 사용하고 있다면
GPU 패키지를 사용해 보세요. 이렇게 하려면 [train.js](./train.js)에 있는
`require('@tensorflow/tfjs-node')`를 `require('@tensorflow/tfjs-node-gpu');`로 바꿉니다.

### 텐서보드로 모델 훈련 모니터링하기

`yarn train` 명령의 `--logDir` 플래그를 사용하면 로그 디렉토리에 배치별 손실 값을 기록할 수 있습니다.
따라서 [텐서보드](https://www.tensorflow.org/guide/summaries_and_tensorboard)를 사용해 모니터링할 수 있습니다.

예를 들어:

```sh
yarn train --logDir /tmp/vae_logs
```

그다음 별도의 터미널에서 텐서보드를 실행합니다:

```sh
pip install tensorboard  # 텐서보드를 설치하지 않은 경우


tensorboard --logdir /tmp/vae_logs
```

텐서보드 프로세스가 실행되면 콘솔에 http:// URL을 출력합니다.
브라우저로 해당 주소에 접속하면 손실 곡선을 볼 수 있습니다. 예를 들면 다음 그림과 같습니다.

![Example loss curve from the VAE training (TensorBoard)](./vae_tensorboard.png)

## 모델 결과 확인하기

훈련이 종료되면 다음 명령을 실행합니다.

```sh
npx http-server
```

출력된 URL에 접속하면 생성된 이미지를 볼 수 있습니다.

![screenshot of vae results on fashion mnist. A 30x30 grid of small images](fashion-mnist-vae-scr.png)
