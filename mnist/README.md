# TensorFlow.js 예제: MNIST 숫자 분류하기

이 예제는 (layers API를 사용하여) MNIST 모델을 훈련하는 방법을 보여줍니다.

이 예제와 관련된 [튜토리얼](https://www.tensorflow.org/js/tutorials/training/handwritten_digit_cnn)도 함께 확인하세요.

이 모델은 모델이 훈련하는 동안 5 스텝마다 1,000개의 랜덤한 테스트 세트 샘플에서 정확도를 계산하여 손실과 정확도를 그래프로 그립니다.
적은 수의 샘플에서 정확도를 계산하면 훈련 시간을 줄일 수 있습니다.

노트: 현재 전체 MNIST 이미지 데이터셋은 하나의 PNG 이미지로 저장되어 있습니다.
`data.js`에 있는 코드가 이를 `Tensor`로 변환합니다.

지금 바로 [온라인 데모](http://ml-ko.kr/tfjs/mnist/)를 확인할 수도 있습니다!
