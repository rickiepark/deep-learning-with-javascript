# TensorFlow.js 예제: MNIST CNN 전이 학습

이 데모는 TensorFlow.js의 층 API를 사용해 전이 학습을 어떻게 수행하는지 보여줍니다.

 * 간단한 합성곱 신경망을 MNIST 데이터셋의 처음 다섯 개 숫자 [0..4]에서만 파이썬 케라스로 훈련했습니다. 원격 URL에 만들어진 모델이 저장되어 있고
`tf.loadLayersModel()`를 사용해 TensorFlow.js로 로딩합니다.
 * 합성곱 층을 동결하고 밀집 층은 숫자 [5..9]를 분류하기 위해 브라우저에서 미세 튜닝합니다.

지금 바로 [온라인 데모](http://ml-ko.kr/tfjs/mnist-transfer-cnn/)를 확인할 수도 있습니다!
