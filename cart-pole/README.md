# TensorFlow.js 예제: 강화 학습 - 카트-막대 문제

![Cart Pole Demo](./tfjs-cart-pole-demo.png)

[See this example live!](https://ml-ko.kr/tfjs/cart-pole/)

## 개요

이 예제는 TensorFlow.js를 사용해 간단한 강화 학습 문제를 해결하는 방법을 보여줍니다.
구체적으로 TensorFlow.js의 층과 그레이디언트 API를 사용해 정책 그레이디언트 방법을 구현합니다.
이 구현을 사용해 전통적인 카트-막대 제어 문제를 해결합니다.
이 강화 학습 문제는 다음 문헌에서 소개되었습니다:

- Barto, Sutton, and Anderson, "Neuronlike Adaptive Elements That Can Solve
  Difficult Learning Control Problems," IEEE Trans. Syst., Man, Cybern.,
  Vol. SMC-13, pp. 834--846, Sept.--Oct. 1983
- Sutton, "Temporal Aspects of Credit Assignment in Reinforcement Learning",
  Ph.D. Dissertation, Department of Computer and Information Science,
  University of Massachusetts, Amherst, 1984.

나중에 OpenAI의 짐(gym) 환경 중의 하나가 되었습니다:
  https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

이 예제에 있는 강화 학습 알고리즘의 핵심은 다음과 같습니다([index.js](./index.js) 참조):

1. 시스템의 관측된 상태가 주어지면 왼쪽 또는 오른쪽으로 미는 힘을 결정하는 정책 네트워크를 정의합니다.
   이 결정은 완전히 결정적이지 않습니다. 이진 확률 분포에서 랜덤한 샘플을 뽑아 실제 행동으로 변환합니다.
2. 오래 진행된 게임이 양의 보상을 받고, 짧게 진행된 게임이 음의 보상을 받는 식으로
   각 게임에 대한 보상 값을 계산합니다.
3. 실제 행동에 대한 정책 네트워크 가중치의 그레이디언트를 계산하고 2단계에서 얻은 보상으로
   이 그레이디언트의 스케일을 조정합니다. 스케일을 조정한 그레이디언트를 정책 네트워크의 가중치에 추가합니다.
   이를 통해 동일한 시스템 상태가 주어지면 정책 네트워크가 오래 진행되는 게임이 되도록 행동을 선택하게 됩니다.

정책 그레이디언트 방법에 대한 자세한 내용은 다음을 참고하세요:
  http://www.scholarpedia.org/article/Policy_gradient_methods

카트-막대 문제를 보여주는 그래픽은 다음을 참고하세요:
  http://gym.openai.com/envs/CartPole-v1/

### 특징:

- 신경망의 층 개수와 층의 크기(유닛 개수) 등과 같은 정책 네트워크의 구조를 지정할 수 있습니다.
- 브라우저에서 정책 네트워크를 훈련하면서 카트-막대 시스템을 동시에 화면에 그릴 수 있습니다.
- 시각화와 함께 브라우저에서 테스트할 수 있습니다.
- 정책 네트워크를 브라우저의 IndexedDB에 저장할 수 있습니다. 저장된 정책 네트워크는 나중에 로드되어
  테스트에 사용하거나 이어서 훈련할 수 있습니다.

## 사용법

```sh
yarn && npx http-server
```
