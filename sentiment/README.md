# TensorFlow.js 예제: 감성 분석

이 데모는 TensorFlow.js를 사용해 텍스트에 대한 감성 분석을 어떻게 수행하는지 보여줍니다.

`tf.loadLayersModel()`을 사용해 원격 URL이나 로컬 파일시스템에서 사전 훈련된 모델을 로드합니다.

두 종류(CNN과 LSTM)의 모델을 지원합니다.
긍정과 부정으로 레이블된 IMDb의 25,000개 영화 리뷰에서 모델을 훈련합니다.
이 데이터셋은 [파이썬 케라스](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification)에서 제공하며
[imdb_cnn](https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py)와
[imdb_lstm](https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py)를 기반으로 케라스에서 모델을 훈련합니다.

데모를 실행하려면 다음 명령을 사용합니다.

```sh
npx http-server
```

[이 예제의 라이브 데모가 있습니다!](https://ml-ko.kr/tfjs/sentiment)

## tfjs-node에서 모델 훈련하기

다음과 같이 tfjs-node를 사용해 모델을 훈련합니다.

```sh
yarn
yarn train <MODEL_TYPE>
```

`MODEL_TYPE`은 필수 매개변수로 어떤 모델을 훈련할지 지정합니다. 가능한 옵션은 다음과 같습니다:

- `multihot`: 모델이 시퀀스에 있는 단어의 멀티-핫 인코딩을 받습니다.
  데이터 표현과 모델 복잡도 측면에서 이 예제에서 사용하는 모델 중에 가장 단순한 모델입니다.
- `flatten`: 시퀀스에 있는 모든 단어의 임베딩 벡터를 펼치는 모델입니다.
- `cnn`: 드롭아웃 층을 포함한 1D 합성곱 모델
- `simpleRNN`: SimpleRNN 층(`tf.layers.simpleRNN`)을 사용한 모델
- `lstm`: LSTM 층(`tf.layers.lstm`)을 사용한 모델
- `bidirectionalLSTM`: 양방향 LSTM 층(`tf.layers.bidirectional`와 `tf.layers.lstm`)을 사용한 모델

기본적으로 tfjs-node의 Eigen 커널을 사용해 CPU에서 훈련합니다.
다음처럼 `--gpu` 플래그를 추가하면 GPU를 사용해 훈련할 수 있습니다.

```sh
yarn train --gpu <MODEL_TYPE>
```

훈련 데이터와 메타데이터가 다운로드되어 있지 않으면 웹에서 두 파일을 다운로드합니다.
모델 훈련이 완료되면 `dist/resources` 폴더에 모델과 `metadata.json` 파일을 저장합니다.
그다음 `npx http-server` 명령을 실행하여 웹 페이지에 있는 "사전 훈련된 로컬 모델 로드하기"를 클릭하면
로컬에서 훈련된 모델을 사용해 브라우저에서 추론을 수행할 수 있습니다.

`yarn train` 명령에는 다음과 같은 옵션이 있습니다:

- `--maxLen`: 시퀀스 길이를 지정합니다.
- `--numWords`: 어휘 사전의 크기를 지정합니다.
- `--embeddingSize`: 임베딩 벡터의 차원을 지정합니다.
- `--epochs`, `--batchSize`, `--validationSplit`: 훈련에 관련된 설정입니다.
- `--modelSavePath`: 훈련 종료 후 모델과 메타데이터를 저장할 위치를 지정합니다.
- `--embeddingFilesPrefix`: 임베딩 벡터와 레이블 파일을 저장할 경로 프리픽스를 지정합니다. 자세한 내용은 다음 절을 참고하세요.
- `--logDir`: 훈련하는 동안 손실과 정확도를 텐서보드 로그로 기록합니다. 예를 들어 다음 명령으로 모델을 훈련하면:

  ```sh
  yarn train lstm --logDir /tmp/my_lstm_logs
  ```

  별도의 터미널에서 다음 명령으로 텐서보드 서버를 실행할 수 있습니다:

  ```sh
  pip install tensorboard   # 텐서보드가 설치되어 있지 않다면
  tensorboard --logdir /tmp/my_lstm_logs
  ```

  그다음 브라우저를 열어 텐서보드에서 제시한 http:// URL(기본값: http://localhost:6006)에 접속하여
  손실과 정확도 곡선을 볼 수 있습니다.

  아래 링크는 TensorBoard.dev에 있는 다양한 모델의 훈련 손실입니다:

  - [`multihot`](https://tensorboard.dev/experiment/8Ltk9awdQVeEdIqmZF6UZg/#scalars)
  - [`flatten`](https://tensorboard.dev/experiment/8dYnJmlDRe21vNJrHYB3Yg/#scalars)
  - [`cnn`](https://tensorboard.dev/experiment/pP6s7BozQESnbXXQy1rJtQ/#scalars)
  - [`simpleRNN`](https://tensorboard.dev/experiment/zl266tMbRuKAr4PBsny8XQ/#scalars)
  - [`lstm`](https://tensorboard.dev/experiment/VHKxx8OnSze7glfzqCXi9A/#scalars)
  - [`bidirectionalLSTM`](https://tensorboard.dev/experiment/osVo6vAaR0SZWzUvElz5ow/#scalars)

자세한 훈련 코드는 [train.js](./train.js) 파일에 있습니다.

### 임베딩 프로젝터에서 단어 임베딩 시각화하기

단어 임베딩을 사용하는 모델(예를 들어, `cnn`이나 `lstm`)을 훈련했다면
`yarn train` 명령이 모델 훈련이 끝난 후 임베딩 벡터와 단어 레이블을 저장할 수 있습니다.
이렇게 하려면 `--embeddingFilesPrefix` 옵션을 사용합니다.

```sh
yarn train --maxLen 500 cnn --epochs 2 --embeddingFilesPrefix /tmp/imdb_embed
```

위 명령은 두 개의 파일을 생성합니다:

- `/tmp/imdb_embed_vectors.tsv`: 단어 임베딩의 숫자 값을 탭으로 구분한 파일. 각 라인은 단어의 임베딩 벡터입니다.
- `/tmp/imdb_embed_labels.tsv`: 이전 파일의 벡터에 상응하는 단어 레이블로 구성된 파일. 각 라인은 단어입니다.

이 파일들을 [임베딩 프로젝터](https://projector.tensorflow.org/)에 바로 업로드하여
[T-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)나
[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) 알고리즘을 사용해 시각화를 수행할 수 있습니다.

샘플 스크린샷을 참고하세요:
![image](https://user-images.githubusercontent.com/16824702/52145038-f0fce480-262d-11e9-9313-9a5014ace25f.png)
