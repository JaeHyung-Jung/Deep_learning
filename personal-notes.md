# personal notes

MNIST등의 데이터 불러올때 첫 명령어

1. import numpy as np
2. import matplotlib.pylplot as plt
3. import tensorflow as tf

학습과정 : Data -&gt; \*\(**Model -&gt; logit -&gt; Loss -&gt; Optm**\) -&gt; Result

\*과정을 반복한 후 일정수준 이상의 정확도가 나오면 결과 도출  

## Feature Extraction\(특징추출\) 

filters : layer에서 나갈 때 몇 개의 filter를 만들것인지

kernel\_size : filter의 Weight의 사이즈

strides : 몇개의 pixel을 skip하면서 지나갈 것인지\(사이즈에 영향을 줌\)

padding : zero padding을 만들것인지, VALID는 Padding이 없고, SAME은 Padding이 있음

activation : Activation Function을 만들것인‌  
MNIST등의 데이터 불러올때 첫 명령어  
  
‌  
import numpy as np  
  
import matplotlib.pylplot as plt  
  
import tensorflow as tf  
  
‌  
Feature Extraction\(특징추출\)   
‌  
filters : layer에서 나갈 때 몇 개의 filter를 만들것인지  
  
‌  
kernel\_size : filter의 Weight의 사이즈  
  
‌  
strides : 몇개의 pixel을 skip하면서 지나갈 것인지\(사이즈에 영향을 줌\)  
  
‌  
padding : zero padding을 만들것인지, VALID는 Padding이 없고, SAME은 Padding이 있음  
  
‌  
activation : Activation Function을 만들것인지



## Optimizing 

Metrics : 모델을 평가하는 방‌  
MNIST등의 데이터 불러올때 첫 명령어

evaluate\(\) 함수를 통해 모델에 대한 정확도를 평가한다.  
  
‌  
import numpy as np  
  
import matplotlib.pylplot as plt  
  
import tensorflow as tf  
  
‌  
Feature Extraction\(특징추출\)   
‌  
filters : layer에서 나갈 때 몇 개의 filter를 만들것인지  
  
‌  
kernel\_size : filter의 Weight의 사이즈  
  
‌  
strides : 몇개의 pixel을 skip하면서 지나갈 것인지\(사이즈에 영향을 줌\)  
  
‌  
padding : zero padding을 만들것인지, VALID는 Padding이 없고, SAME은 Padding이 있음  
  
‌  
activation : Activation Function을 만들것인‌  
MNIST등의 데이터 불러올때 첫 명령어  
  
‌  
import numpy as np  
  
import matplotlib.pylplot as plt  
  
import tensorflow as tf  
  
‌  
Feature Extraction\(특징추출\)   
‌  
filters : layer에서 나갈 때 몇 개의 filter를 만들것인지  
  
‌  
kernel\_size : filter의 Weight의 사이즈  
  
‌  
strides : 몇개의 pixel을 skip하면서 지나갈 것인지\(사이즈에 영향을 줌\)  
  
‌  
padding : zero padding을 만들것인지, VALID는 Padding이 없고, SAME은 Padding이 있음  
  
‌  
activation : Activation Function을 만들것인지  
  
​  
  




## Training

epochs : 한글로 설명하자면 회독이다 \(1회독, 2회독 등..\). 즉, 모든 노드에 해당되는 데이터셋을 한번씩 학습한 횟수가 epochs이다. 

_좋은 비유가 생각나서 설명해보자면, 책을 읽을때 한페이지 한페이지씩 읽어나가 모든 페이지를 다 읽으면 책을 다 읽은것이고 이것을 1회독이라한다. 여기서 회독이 epoch와 같고 각 페이지를 data라고 생각하면된다. 책을 두번, 세번읽으면 2회독 3회독이고 이는 딥러닝에서도 마찬가지로 epoch\(2\), epoch\(3\)이다._

_data가 5만개인 datasets의 epoch가 1이라면 5만개의 각 데이터가 모두 한번씩 train됐다는 의미이다._

batch : 나누어진 데이터셋의 갯수.  한 epoch을 돌기위해 한 모델에 한번에 들어가는 데이터의  갯수

shuffle : overfitting을 피하기 위해 섞어주는 기능 \(같은 순서의 data가 계속 들어오면 overfitted 될 수 있음\)





## Activation & Loss Function

### Activiation Function

#### Sigmoid :

output node에서 나온값을 0~1의 범위로 조정해주는 함수

A function that adjusts the value from the output node in the range of 0 to 1.

#### Softmax :

outputnode에서의 모든값이 0~1의 범위에 속하며 모든 값을 더하여 1이되는 조건을 가진 함수

A function that has the condition that all values ​​in the output node fall within the range of 0 to 1, and all values ​​are added together to become 1.

### Loss Function

#### Cross Entropy : 



#### Binary Entropy :


### 강의내용 정리 :
Batcg Normalization에서 Normalization은 FCL->RElU를 거친 학습데이터의 분포가 일정치않은것을 일정하게 만들어주어 학습률을 쉽게 결정하게 해준다.   Normalization연산에는 Scaling계수(r)과 Bias가 필요하고 값을 학습시킬수 있다.

Convolution net :
- 종류 : LeNet, AlexNet(2012), VGG(2014), GoogleNet(2014)
- 

!!!(중요) : Google Net 
- inception.v1 : 1x1, 3x3, 5x5 convolution등의 여러 convolution연산과 pooling(1개)의 연산을 병력적으로 연산시켜 병목현상을 통해 학습시킴, BottleNeck을 통해 연산량
이러한 병목현상중에 중간과정에서도 분류기를 두어 Classification이 가능함(Gradient Vanishing 방지)


!!!!!!!!!!!!!!!(매우중요) ResNet(Residual Network) '2015 
- 계층 152까지 높이며 error 혁신적으로 줄임. ResNet을 통해 Human Performance 보다 띄어넘음
#### Skip-Connection이 주요한 역할을함 
    - 일반적인 구조 : conv layer -> relu 반복 
    - residual 구조 : conv layer -> relu -> x(입력의 identity) 더해줌 ==> 쉽게말해 feature 추출전부분을 추출후에 더한다.
        - pre-Activation : wegiht를 후에두어 [BN->ReLU->weight] 순으로 하는방식 (: Gradient-Highway가 가능하게)

#### Dense Net(Res Net을 계승함) 
![image](https://user-images.githubusercontent.com/79160507/115515022-aaa00880-a2bf-11eb-8929-2bd5a1cb8eab.png)
  - 전체적 구조 : 각 Block끼리 모두 서로 연결된 형태
  - ResNet의 연장선이며 Dense Block이 필요하다. Dense Block내부는 Pre-Activation 구조이다.
  - 이전 feature map에 concatenate(계속해서 이어붙여 연장)해줌 이와같은 방식은 구조적으로 Skip-Connection과 같다. 계속해서 concetanate되기때문에 첫번째 layer가 이후에도 connection되기때문에
 
 
 Net별 특성 정리
 - VGG : CONV2D -> pooling을 반복적으로 해주며 flatten해줌 (최종적으로 1x1xN(무수히 큼) 형태로, 스칼라가 channel이 굉장히 긴 형태)
 - ResNet : 
 - DenseNet :


RNN : 주로 자연어처리에 쓰이는 net이다. (다중입력 -> 단일출력 / 단일입력 -> 다중출력 / 다중입력 -> 다중출력 등이 있다)
  - 순차데이터(Sequential data) : 순서가 의미가 있으며 순서가 달라질 경우 의미가 손상되는 데이터

Regularization :
일반적으로 train 에서는 loss값이감소하지 않지만 test, validation에서 loss값이 감소하도록 정규화 해주는 과정

- Weight Decay : Weight L-2 Norm을 최소화 하는 정규화 기법. ==> Weight Decay를 통해 weight가 너무 복잡하여 overfitting되는 것을 막는다.
- 정규화는 영상처리 알고리즘에 적용하기 좋다

Batch Normalization : 데이터 분포를 정규화
- BN은 학습 할동안은 연산량을 올리지만 test할때에는 영향을 주지않는다.

# CNN의 발전 :

1) Vanilla CNN : Conv2D -> Relu
2) ResNet : (BN -> ReLU -> Conv )반복 -> layer + identity
3) DenseNet : BN -> Relu -> Conv 모든 layer들이연결

Self-Normalization neural network 
- SELU 함수 : ReLu는 음수값을 0으로하지만 SELU는 음수값을 Exponential하게활성화한다. ( 0이하는 지수함수꼴로 -2로 수렴, 0이상은 x=y로 출력)
- alpha-dropout : Relu에 Drop-out이 적용되는것처럼 SeLU에 적합한 Drop-out방식 
- ==> SNN의 결과 : 세심한 수학적 연구(selu, alpha-dropout등을 수많은 실험을 통해 나온 계수를 통하여 개념적, 직관적 효율적인 layer의 구조보다 훨씬 더 성능좋은 net을 만들 수 있다)

Imbalanced Data(데이터가 너무작거나 특정 class의 data가 다른 class의 data에비해 훨씬 많은경우) :
Classify할 때 class의 양이 확실히다를때 Imbalanced Data라 하고 이런 data는 여러 방법으로 classify 할 수 있다.
1) random under sampling : minority, majority class중에서 majority class에서 minotity class의 수만큼 임의로 data를뽑아 비율을 맞춰주는 방법
2) random over sampling : minority class의 데이터를 반복적으로 두어 majority class의 비율과 맞춰주는 방법. (minority class의 가중치를 증가시키는 것과 유사)
3) SMOTE(synthetic minority oversampling technique) : minority class를 combination하고 linear combination을 추가
--> 3-1) Borderline-SMOTE : 경계에 있는 샘플만 오버샘플링하여 smote하는 효과적인 방법


## 데이터 증강 (이미지의 변형, 회전등)
- invariance : 불변성 (회전하고 scale하면 invarian하다. 그러나 평행이동하고 회전하면 variant함)
  - CNN은 Affine Transform(2차원 변환)에 대해 Variant하다. 즉, affine transform된 영상을 다른 영상으로 인식
- Noise 삽입 : 다양한 방법(gaussian, jpeg)등으로 augmentation 할수있다.

정규화 :
- L-2 Normalization(==Ridge)  : 아주 큰 가중치에 패널티부여(-> 가중치를 비교적 평평하게 해줌)
  => L-2에서의 람다값 : 람다값이 높을수록 정규분포에가깝다.
- MSE Loss : 평균을 나타내는 특성이있다.
- MAE Loss : 오차가 큰 에러값을 무시하는 특성을가지고 outlier에 크게 영향을 받지않는다.

* 과소적합 데이터 해결 :
- prefetch(GPU에 미리 데이터를 올려줌으로써 학습 속도 향상)


## 부족한 데이터셋 문제해결(가장 현실적인 문제해결, 매우 중요하다고생각) :
- Bordeline SMOTE 사용 (Bordeline SMOTE를 통해 dataset을 rescaling하여 학습시키면 학습률이 증가)




## '04.21.(수) question summarizations
- 질문 #1 : np.shape할때 왜 차원이 중첩해서 늘어나지않고 초기화되냐 ?   ==> x == np.exand()형태로 variable에 저장해주지 않아서 그렇다

- 질문#2 : cross-entropy 직관적 설명   ==> 직접 scalar 넣어보면서 이해함

- 질문#3 : plt plot할때 3차원(x, y, 1)이아니라 1의값이 n이어도 출력되는가?   ==> plt.plot [w,h,c]형태면 가능. 강의에서 (x,y,1)형태로 한 이유는 rgb channel없이 height, wight만 추출하기위해

- 질문#4 : kaggle competition(cactucs)에서 datasets를 train으로 저장할때 train.zip 확장자명 그대로 사용할 수 없는가?   ==> datasets를 local로저장하여 알집풀어서 local로 수행함

- 질문#5 : dropout(x)에서 x의 수치적용방식 질문 (x만큼 dropout되는지 x만큼 연산을 하는건지(==(1-x)만큼 dropout))   ==> x만큼 dropout됨. 즉 dropout(1)하면 오류발생

- 질문#6 : train set을 학습시킨후 evaluate로 출력한 정확도와 model.fit(test sets, epochs=1)했을때 정확도의차이가 생기는 이유?   ==> model.fit(test, epoch=1)했을때 정확도가 evaluate보다 조금더 올라갈 것이라 생각하였으나 학습과정에서 무조건 accuracy가 올라가는것이 아님(대체적으로 epoch가 늘어날수록 accuracy가 상승하는건 맞음)

- 질문#7 : forward, backward(backprogation)에서의 수식적 개념이해와 미분방식등의 직접적인 이해가 필요한가?   ==> 어떤식으로 작동하는지 이해하되 수식과같이 직접 loss function, convolution net등을 짜볼필요까진없다 Frame Work가 있기때문에

- 질문#8 : Batch Normalization는 데이터의 분포를 정규분포화 시켜 학습에 용이하기 위함을 알았다. 그런데 ReLU를 통과한 데이터의 분포가 왜 0보다 작은값도 분포로 그대로 나오는가에 대해서 질문?   ==> ReLU를 통과한 데이터들의 분포를 나타낸것임. 즉 분포자체를 ReLU통과시킨게 아니라 ReLU로 통과시킨 data들을 분포화한것

- 질문#9 : dense()는 직전의 flatten()의 값보다 작거나 커도된다.    

- 질문#10 :  ![image](https://user-images.githubusercontent.com/79160507/116063361-d9532000-a6bf-11eb-8d66-9257869d5ccf.png)
Dense(32) -> BN -> BN -> Dense(32) 하면 Concetanation 돼서 -> BN 64된다는데 Dense(32)가아니라 (32+32 = 64) concetanation되는과정 아예모르겠음 왜저렇게되는지 
BN은 W(Z)+B / Z= WX +B --> W(Z) = W(WX+B) + B인데 
   
- 질문#11 :     
   
- 질문#12 :  
   
- 질문#13 :  
   
- 질문#14 :  
   


!! Model.summary()에서 Param의 값은 (필터 크기) x (#입력 채널(RGB)) x (#출력 채널) + (출력 채널 bias)

# 질문할것들
1) param값 구하는건 채널크기 x filter의 크기 + b(채널) 인것은 알겠으나 그 다음 layer의 param구할때 왜 그전 bias는 뺴주는가. 
2) 만약 Dense로 flatten()값보다 크게해주면 어떤식으로 늘어나는건가. maxpooling처럼 축소하는건 영역을 일정하게 나눠서 최대값이나 최소값을 추출해주는등으로 축소하는 방식을 알겠는데 dense(x), flatten() = y 일때 x>y면 x가 y의 정수배로 나누어 떨어지지 않으면 어떤식으로 추가되는건가.
3) model.summary()에서 각 layer의 param는 필터크기와 입출력채널의 크기로 결정된다. 그렇다면 입력이나 출력데이터의 H, W는 관여하지않는데 왜그런가?


`21. 4. 20~ 주말 숙제
![image](https://user-images.githubusercontent.com/79160507/115550458-d8e40f00-a2e4-11eb-9061-52942e461246.png)

output shape : 
Convd2d = (None, 28, 28, 16)
Convd2d1 = (None, 28, 28, 16)
maxpooling2d = (None, 14, 14, 16)
Conv2d_2 = (None, 14, 14, 32)
Conv2d_3 = (None, 14, 14, 32)
maxpooling2d_1 = (None, 7, 7, 32)
Conv2d_4 = (None, 7, 7, 64)
Conv2d_5 = (None, 7, 7, 64)
flatten = (None, 7*7*64)
dense = (None, 128)
dense_1 = (None, 10)

Params :
// filter size 3x3 = 9 는 이 모델에서는 고정값
(필터크기 x 입력채널크기 x 출력채널크기 + 출력채널수 (==출력채널의 b 총갯수))
(9 x 16 + 16) = 160
(9 x 16 x 16 + 16) = 2320
(9 x 16 x 32 + 32) = 4640
(9 x 32 x 32 + 32) = 9248
(9 x 32 x 64 + 64) = 18496
(9 x 64 x 64 + 64) = 36928
(1 x (3136(=7*7*64)) x 128 + 128) = 401536
(1 x 128 x 10 + 10) = 1290




