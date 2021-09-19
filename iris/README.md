# TensorFlow.js 예제: 붓꽃 품종 분류하기

이 데모는 TensorFlow.js Layers API를 사용해 클래식한
[붓꽃 품종 데이터셋](https://en.wikipedia.org/wiki/Iris_flower_data_set)을 분류하는 방법을 보여줍니다.

여기서 보여주는 모델 구축 방법은 다음과 같습니다:
* `tf.loadLayersModel()`을 사용해 URL에서 사전 훈련된 모델을 로딩하기
* 브라우저에서 밑바닥부터 모델을 만들고 훈련하기

이 데모는 또한 훈련 과정을 실시간으로 시각화하기 위해
`Model.fit()` 메서드의 `callbacks` 매개변수를 사용하는 방법을 보여줍니다.

이 모델은 두 개의 `Dense` 층으로 구성됩니다.
하나는 `relu` 활성화 함수를 사용하고 뒤따르는 층은 `softmax` 활성화 함수를 사용합니다.

이 데모는 다음 명령으로 실행합니다.

```sh
$ npx http-server
```

지금 바로 [온라인 데모](http://ml-ko.kr/tfjs/iris/)를 확인할 수도 있습니다!
