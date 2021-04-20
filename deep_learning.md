---
description: 아직 가다듬지 않았고 강의를 보면서 순간순간 기록을 위해  정리해놓음. 수정필요
---

# Deep\_learning

 딥러닝 내부는 특징추출\(convolution + relu max pooling\) 과 classification로 나뉜다.

## numpy 

0 차원 : scalar

1차원 : vector

2차원 : 행렬

3차원 이상 : tensor 

**ravel** : 배열 일차원 정렬    

expand\_dims\(\) : 1차원 추

zeros & ones : 0으로 채워진 numpy 배열 생

arange\(x, y\) : create a array from x to y

broadcast : 연산하려는 두 행렬의 shape가 같지않더라도 행이나 열 등 어느 인자가 같은 부분이 있으면 복사하여 연산하게 해주는 연산‌  
 딥러닝 내부는 특징추출\(convolution + relu max pooling\) 과 classification로 나뉜다.  
  


Graph Visualization : 그래프 시각화

plot 그래프 옵션 :

* b,g,r : 블루, 초록색, 빨강, C\(청록\), y\(노랑\), k\(검은색\), w\(흰색\)
* 마커 : o\(원\), v\(역삼각형\), ^\(삼각형\), s\(네모\), +\(플러스\), .\(dot\)

이미지 시각  
 딥러닝 내부는 특징추출\(convolution + relu max pooling\) 과 classification로 나뉜다.  
  
‌  
open\_cv의 cv2를 이용해서 두 이미지를 합치기 위해 resize할때 그림의 크기\(x,y, z\(컬러\)\)에서 x,y의 순서를 바꿔서 적어줘야됨



Tensorflow 특징 :

Numpy Array와 호환이 쉽다.

대중적이며 Tensorboard, TFLite, TPU등을 지원



PyTorch : 

쉽고 빠르며 파이써닉하다\(코드의 간결함\)

성장률이 높고 연구용으로 자주쓰인다.



Activation functions :

![source : https://docs.paperspace.com/machine-learning/wiki/activation-function](.gitbook/assets/image%20%287%29.png)

주로 Sigmoid나 Softmax를 사용한다. ReLU도 자주 사용하며, 은닉층에는 Relu 활성화함수를 사용하고 출력층에 Sigmoid를 사용하는 방법으로 정확도를 높일 수 있다.

Softmax함수는 위 그림에는 나와있지 않지만 출력이 0~1을가지며 그 모든 출력의 합이 1이되게 하 \(sum = 1\)함수이다.



* 모델 성능 향상에 필요한 방법 :



Augmentation : 데이터를 증폭켜 이미지에 변화를주어 모델에 적용

augmentation을 통해 여러 환경에서도 적응이 되도록 모델에게 hard train



