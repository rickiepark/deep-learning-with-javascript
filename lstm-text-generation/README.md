# TensorFlow.js 예제: 텍스트 생성 LSTM 훈련하기

[라이브 데모](https://ml-ko.kr/tfjs/lstm-text-generation)

## 개요

이 예제는 TensorFlow.js를 사용해 LSTM 모델을 훈련하여 니체의 글이나 TensorFlow.js 소스 코드 같은
텍스트 말뭉치에 있는 패턴을 기반으로 랜덤한 텍스트를 생성하는 방법을 보여줍니다.

이 LSTM 모델은 문자 수준에서 작동합니다. `[numExamples, sampleLen, charSetSize]` 크기의 텐서를 입력으로 받습니다.
입력은 원-핫 인코딩된 `sampleLen` 개의 문자로 이루어진 문장입니다.
문자는 `charSetSize` 개의 고유한 문자 집합에 속해 있습니다.
모델은 `[numExamples, charSetSize]` 크기의 텐서를 출력합니다.
이 텐서는 입력 시퀀스 뒤를 이을 문자에 대한 모델의 예측 확률을 나타냅니다.
그다음 이 애플리케이션은 다음 문자를 위해 예측 확률을 기반으로 랜덤한 샘플을 뽑습니다.
다음 문자가 결정되면 이 문자의 원-핫 인코딩을 입력 시퀀스에 연결하여 다음 타임 스텝의 입력을 만듭니다.
이런 과정이 지정된 길이의 문자 시퀀스를 생성하기 위해 반복됩니다.
무작위성(다양성)은 온도 파라미터로 조정합니다.

UI를 사용해 하나 이상의 [LSTM 층](https://js.tensorflow.org/api/latest/#layers.lstm)으로 구성된 모델을 만들 수 있습니다.

또한 이 예제는 훈련 결과를 브라우저 세션 간에 유지할 수 있도록
TensorFlow.js의 [모델 저장 API](https://js.tensorflow.org/tutorials/model-save-load.html)를
사용해 훈련된 모델을 브라우저의 IndexedDB에 저장하는 방법을 보여줍니다.
이전에 훈련된 모델을 IndexedDB로부터 로드하면, 이를 사용해 텍스트를 생성하거나 훈련을 이어갈 수 있습니다.

## 사용법

### 웹 데모 실행하기

이 웹 데모는 모델 훈련과 텍스트 생성을 제공합니다. 데모를 실행하려면 다음 명령을 사용합니다:

```sh
yarn && npx http-server
```

### Node.js에서 모델 훈련하기

Node.js에서 모델을 훈련하는 것이 브라우저 환경보다 빠릅니다.

훈련을 시작하려면 다음 명령을 사용합니다:

```sh
yarn
yarn train shakespeare \
    --lstmLayerSize 128,128 \
    --epochs 120 \
    --savePath ./my-shakespeare-model
```

- `yarn train`의 첫 번째 매개변수(`shakespeare`)는 모델이 훈련할 텍스트 말뭉치를 지정합니다.
  지원하는 텍스트 데이터는 `yarn train --help`의 출력을 참고하세요.
- 매개변수 `--lstmLayerSize 128,128`는 다음 문자 예측 모델이 각각 128개의 유닛을 가진 두 개의 LSTM 층을 포함한다는 것을 지정합니다.
- `--epochs` 플래그는 훈련 에포크 횟수를 지정합니다.
- `--savePath ...` 플래그는 훈련이 끝난 후 모델을 저장할 경로를 지정합니다.

컴퓨터에 CUDA 지원 GPU가 설치되어 있다면 명령줄에 `--gpu` 플래그를 추가하여 GPu에서 모델을 훈련할 수 있습니다.
이렇게 하면 훈련 속도를 더 높일 수 있습니다.

### 저장된 모델 파일을 사용해 Node.js에서 텍스트 생성하기

위 명령은 훈련이 끝난 후 `./my-shakespeare-model` 폴더에 일련의 모델 파일을 만듭니다.
이 모델을 로드하고 이를 사용해 텍스트를 생성할 수 있습니다. 예를 들면 다음과 같습니다:

```sh
yarn gen shakespeare ./my-shakespeare-model/model.json \
    --genLength 250 \
    --temperature 0.6
```

이 명령은 세익스피어 말뭉치에서 랜덤하게 텍스트 샘플을 샘플링하고 이를 텍스트를 생성하기 위한 시드로 사용합니다.

- 첫 번째 매개변수(`shakespeare`)는 텍스트 말뭉치를 지정합니다.
- 두 번째 매개변수는 이전 절에서 생성한 모델의 JSON 파일이 저장된 경로를 지정합니다.
- `--genLength` 플래그를 사용해 생성할 문자 개수를 지정합니다.
- `--temperature` 플래그를 사용해 생성 과정의 확률성(무작위성)을 지정할 수 있습니다.
  이 값은 0보다 크거나 같아야 합니다. 값이 클수록 생성된 텍스트의 무작위성이 증가합니다.
