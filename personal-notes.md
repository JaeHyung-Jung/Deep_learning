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











## 

## 

##   ‌


## '04.21.(수) question summarizations
- 질문 #1 : np.shape할때 왜 차원이 중첩해서 늘어나지않고 초기화되냐 ?   ==> x == np.exand()형태로 variable에 저장해주지 않아서 그렇다

- 질문#2 : cross-entropy 직관적 설명   ==> 직접 scalar 넣어보면서 이해함

- 질문#3 : plt plot할때 3차원(x, y, 1)이아니라 1의값이 n이어도 출력되는가?   ==> plt.plot [w,h,c]형태면 가능. 강의에서 (x,y,1)형태로 한 이유는 rgb channel없이 height, wight만 추출하기위해

- 질문#4 : kaggle competition(cactucs)에서 datasets를 train으로 저장할때 train.zip 확장자명 그대로 사용할 수 없는가?   ==> datasets를 local로저장하여 알집풀어서 local로 수행함

- 질문#5 : dropout(x)에서 x의 수치적용방식 질문 (x만큼 dropout되는지 x만큼 연산을 하는건지(==(1-x)만큼 dropout))   ==> x만큼 dropout됨. 즉 dropout(1)하면 오류발생

- 질문#6 : train set을 학습시킨후 evaluate로 출력한 정확도와 model.fit(test sets, epochs=1)했을때 정확도의차이가 생기는 이유?   ==> model.fit(test, epoch=1)했을때 정확도가 evaluate보다 조금더 올라갈 것이라 생각하였으나 학습과정에서 무조건 accuracy가 올라가는것이 아님(대체적으로 epoch가 늘어날수록 accuracy가 상승하는건 맞음)

- 질문#7 : forward, backward(backprogation)에서의 수식적 개념이해와 미분방식등의 직접적인 이해가 필요한가?   ==> 어떤식으로 작동하는지 이해하되 수식과같이 직접 loss function, convolution net등을 짜볼필요까진없다 Frame Work가 있기때문에

- 질문#8 :     

- 질문#9 :     

- 질문#10 :  
   
- 질문#11 :     
   
- 질문#12 :  
   
- 질문#13 :  
   
- 질문#14 :  
   







