#### 김성겸박사님 공백으로 8월말 인턴계약까지 인력양성을 맡게됐음

#### 8.10~8.11 KAIST학생(3명)과 딥러닝 (Regression)회의 => 회의 이후 기초적인 딥러닝 가이드

## What is MachineLearning & Deep learning?
기계학습ML은 딥러닝을 포함하며 딥러닝보다 더 넓은 범주이다.
몇년전부터 기계학습의 한 분야인 딥러닝이 자연어처리, 컴퓨터비전, 시계열 등의 분야에서 최고의 성능을 보여주고있다.

딥러닝과 기계학습은 학습이 되는 원리로 구별할 수 있다.
머신러닝은 빅데이터를 학습하는 알고리즘을 통해 결과를 추출한다. 
딥러닝은 인공신경망(Aritificial-Neural-Network)를 통해 학습되며 단순한 인공신경망이 아닌 주로 Convolution연산을 사용하는 깊은 NeuralNetwork를 거치면서 학습되는것이 특징이다.

좀더 깊게 들어가면 Regression은 x(입력값) y(결과값)을 모두 받아 그 데이터를 분석하는 지도학습이다.
지도학습이란 정답을 받아 오차를 줄여나가는 방향으로 학습하는 방식을 말한다.

딥러닝은 지도학습(Classification, Detection, ..)뿐만 아니라 비지도학습(정답이 주어지지않고 데이터만으로 학습을하는 ex)Anomaly_Detection)과 강화학습등 여러 방법으로 학습을 할 수 있다.

## What is Regression?
MachineLearning에 사용되는 Regression이란?

Machine_Learning의 기법으로 y=Wx+b와같이 독립변수 x에 가중치(W), 편향(b)를 곱하고 합하여 나오는 종속변수y와같은 식으로
기계학습을 통해 결과 분석,추출출하는 방법이다.

예를들어 운동을 했을때 땀이나는것을 이용하여 운동을 얼마나(소모 kcal기준) 하였을 때 어느정도의 땀이 나오는가를
y(땀의 량) = W(가중치)*x(운동량) + b(bias)로 두고 학습을 진행하여 운동량에대한 땀의 배출량을 계산하는 것과같이 기계학습을 할 수 있다.


### Linear Regression
앞서 설명한 y=Wx+b가 일반적인 선형 회귀분석이다.

![image](https://user-images.githubusercontent.com/79160507/128653331-0c4d5eec-61c6-414f-bdb5-6a3b657ac9f2.png)
선형 회귀분석을 통한 데이터분석은 위의 그림과 같다.
위의 각 점이 결과값이고 빨강색 직선이 분석값(y=wx+b)이다.
각 점에대해서 정답(직선)과의 차이를 loss라고하며 loss가 작을수록 학습이 정확하게 잘 됐다는것을 의미한다.

Multi Linear Regression또한 존재하며 이는 독립변수x가 여러가지인 경우이다.
MLR을 식으로 나타내면 다음과 같다. y= W1x(1) + W2x(2) + W3x(3) +... Wnx(n) +b

MLR은 여러요소(Xn)에 의해 y가 결정되는 식이며 
MLR가 쓰이는 예를들자면 미세먼지의 총량을y라하면 x1(PM2.5) x2(CO) x3(NO2)등의 여러 입자에의해 y가 결정되는 경우 MLR을 통해 회긔분석을 한다.

### Logistic Regression
Logistic Regression은 binary Classification에서 주로 쓰이며, Binary Classificaitno이란
y값이 (0,1)인 이진분류이다.

Binary Classificatino의 대표적인 예로는 y가 [합격인지 불합격]인지, 또는 [참이냐 거짓이냐] 등이 있다.

|------|---|
|x(score)|y(pass)|
|65|non-pass|
|80|pass|
|85|pass|
|55|non-pass|
|90|pass|
|60|non-pass|
|100|pass|   
(Binary Classification 예시)

|Binary_Classification|Score,Pass|If(score>=80),then pass|
|------|---|---|
|x(score)|y(pass)||
|65|테스트2|테스트3|
|테스트1|테스트2|테스트3|
