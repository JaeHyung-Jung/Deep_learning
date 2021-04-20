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











