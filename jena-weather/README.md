# TensorFlow.js 예제: 예나 날씨 데이터 시각화와 예측

이 데모에 포함된 내용은 다음과 같습니다.
- [tfjs-vis](https://www.npmjs.com/package/@tensorflow/tfjs-vis) 라이브러리를 사용하여 시계열 데이터 시각화하기
- 여러 종류의 모델을 사용해 순차 입력 데이터를 기반으로 미래의 값을 예측하기
  - 선형 회귀
  - 다층 퍼셉트론 (MLP)
- 과소적합, 과대적합, 과대적합을 줄이기 위한 여러가지 기법
  - L2 규제
  - 드롭아웃

이 데모에 사용된 데이터는 [예나 날씨 데이터셋](https://www.kaggle.com/pankrzysiu/weather-archive-jena)입니다.

이 예제는 TensorFlow.js에서 다음 API를 사용합니다.

- `tf.data.generator()`: 제너레이터 함수에서 `tf.data.Dataset` 객체를 생성합니다.
- `tf.Model.fitDataset()`: `tf.data.Dataset` 객체를 사용해 `tf.Model`을 훈련하고
  또 다른 `tf.data.Dataset` 객체를 사용해 훈련 에포크가 끝날 때마다 모델의 검증 성능을 평가합니다.
- `tfvis.show.fitCallbacks()`: 모델 훈련에서 배치와 에포크가 끝날 때마다
  훈련 세트와 검증 세트의 손실을 그래프로 그립니다.
