# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# 16비트와 8비트 가중치 양자화의 이산화를 직관적으로 이해하기 위한 그림을 그립니다.

from matplotlib import pyplot as plt
import numpy as np


def quantize(w, bits):
  """
  가중치 양자화를 시뮬레이션합니다.

  Args:
    w: (a numpy.ndarray) 양자화할 가중치
    bits: (int) 양자화에 사용할 비트 수: 8 또는 16

  Returns:
    세 개의 원소를 가진 튜플:
      w_quantized: uint8 또는 uint16 타입의 numpy.ndarray로 표현된 w의 양자화 버전
      w_min: 역양자화에 필요한 w의 최솟값
      w_max: 역양자화에 필요한 w의 최댓값
  """
  if bits == 8:
    dtype = np.uint8
  elif bits == 16:
    dtype = np.uint16
  else:
    raise ValueError('지원하지 않는 양자화 비트입니다: %s' % bits)

  w_min = np.min(w)
  w_max = np.max(w)
  if w_max == w_min:
    raise ValueError('w가 모두 동일한 값이기 때문에 양자화를 수행할 수 없습니다')
  w_quantized = np.array(
       np.floor((w - w_min) / (w_max - w_min) * np.power(2, bits)), dtype)
  return w_quantized, w_min, w_max


def dequantize(w_quantized, w_min, w_max):
  """
  가중치 역양자화를 시뮬레이션합니다.

  Args:
    w_quantized: uint8 또는 uint16 타입의 numpy.ndarray로 표현된 w의 양자화 버전
    w_min: 역양자화에 필요한 w의 최솟값
    w_max: 역양자화에 필요한 w의 최댓값

  Returns:
    w: (numpy.ndarray) 역양자화된 가중치
  """
  if w_quantized.dtype == np.uint8:
    bits = 8
  elif w_quantized.dtype == np.uint16:
    bits = 16
  else:
    raise ValueError(
        '지원하지 않는 양자화 값입니다: %s' % w_quantized.dtype)
  return (w_min +
          w_quantized.astype(np.float64) / np.power(2, bits) * (w_max - w_min))


def main():
  # sine 곡선을 그리기 위해 사용하는 x 축의 포인트 개수
  n_points = 1e6
  xs = np.linspace(-np.pi, np.pi, n_points).astype(np.float64)
  w = xs

  w_16bit = dequantize(*quantize(w, 16))
  w_8bit = dequantize(*quantize(w, 8))

  plot_delta = 1.2e-4
  plot_range = range(int(n_points * (0.5 - plot_delta)),
                     int(n_points * (0.5 + plot_delta)))

  plt.figure(figsize=(20, 6))
  plt.subplot(1, 3, 1)
  plt.plot(xs[plot_range], w[plot_range], '-')
  plt.title('Original (float32)', {'fontsize': 16})
  plt.xlabel('x')

  plt.subplot(1, 3, 2)
  plt.plot(xs[plot_range], w_16bit[plot_range], '-')
  plt.title('16-bit quantization', {'fontsize': 16})
  plt.xlabel('x')

  plt.subplot(1, 3, 3)
  plt.plot(xs[plot_range], w_8bit[plot_range], '-')
  plt.title('8-bit quantization', {'fontsize': 16})
  plt.xlabel('x')

  plt.show()


if __name__ == '__main__':
  main()
