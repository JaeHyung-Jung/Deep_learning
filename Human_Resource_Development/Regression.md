#### 김성겸박사님 공백으로 8월말 인턴계약까지 인력양성을 맡게됐음

#### 8.10~8.11 KAIST학생(3명)과 딥러닝 (Regression)회의 => 회의 이후 기초적인 딥러닝 가이드

## What is MachineLearning & Deep learning?
기계학습ML은 딥러닝을 포함하며 딥러닝보다 더 넓은 범주이다.
몇년전부터 기계학습의 한 분야인 딥러닝이 자연어처리, 컴퓨터비전, 시계열 등의 분야에서 최고의 성능을 보여주고있다.

딥러닝과 기계학습은 학습이 되는 원리로 구별할 수 있다.
머신러닝은 빅데이터를 학습하는 알고리즘을 통해 결과를 추출한다.
딥러닝은 인공신경망(Aritificial-Neural-Network)를 통해 학습되며 단순한 인공신경망이 아닌 주로 Convolution연산을 사용하는 깊은 NeuralNetwork를 거치면서 학습되는것이 특징이다.

## What is Regression?
MachineLearning에 사용되는 Regression이란?

Machine_Learning의 기법으로 y=Wx+b와같이 독립변수 x에 가중치(W), 편향(b)를 곱하고 합하여 나오는 종속변수y와같은 식으로
기계학습을 통해 결과를 추출하는 방법이다.

예를들어 운동을 했을때 땀이나는것을 이용하여 운동을 얼마나(소모 kcal기준) 하였을 때 어느정도의 땀이 나오는가를
y(땀의 량) = W(가중치)*x(운동량) + b(bias)로 두고 학습을 진행하여 운동량에대한 땀의 배출량을 계산하는 것과같이 기계학습을 할 수 있다.


### Linear Regression
앞서 설명한 y=Wx+b가 일반적인 선형 회귀분석이다.

![image](https://user-images.githubusercontent.com/79160507/128653202-c149f583-db59-4a5e-a29f-155cbe09a028.png)
선형 회귀분석을 통한 데이터분석은 위의 그림과 같다.




### Logistic Regression
