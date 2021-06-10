Object detection : Classification + Localization(Bounding Box Regression)이다.

## 2stage_detector & 1stage_detector   

#### 2stage_detctor   
2stagge detctor는 Classification이후 Localizaiton하는 과정으로 디텍팅한다.   ex : RCNN, fastRCNN, fasterRCNN, Resnet, DenseNet]   
Region Proposal : 객체를 인식하여 Bounding box를 그려주는 방식은 기존의 Sliding Window방식이 있었으나 모든 화면을 인식하여 찾는방법이 비효율적이라   
Selective Search를 통해 객체가 있을만한 지역을 알고리즘으로 찾아 Bounding box를 그려주는 Region Proposal방식을 사용하고 있다.

#### 1stage_detctor   
1stage detector는 classification과 localization을 동시에하는 과정으로 디텍팅한다.   ex : YOLO, SSD, FocalLoss, RefineDet

### RCNN Family
* Region Proposal : 카테고리(클래스)와 무관하게 물체의 영역을 찾는 모듈   
* CNN : 각각의 영역으로부터 고정된크기의 feature vector를 생성(특징추출)   
* classification : 분류를 위한 linear regression 학습모델 SVM

#### SVM : 선형분류와 비선형분류에서 사용될 수 있으며 데이터가 어느 카테고리에 속하는지 판단하는 모델을 생성한다.

#### RCNN
1. Input image   
2. Extract region proposals(~2K)
3. Compute CNN features
4. Classify regions (is it person? or vehicle ? ...)
RCNN의 구조 : CNN, SVM, Bounding box regression 세가지의 모델을 사용하여 이미지를 region proposal -> feature extraction -> classificaiton 하는 구조
RCNN 단점 : 오래걸린다, 복잡하다. BackPropagation이 안된다. 구조적으로 CNN과 SVM이 분리되어있다.

Roi Pooling : 
region proposal후 제각각의 사이즈로 잘린 객체들을 같은 사이즈로 변환하여 FCL에 넣어주는 방식

#### Fast RCNN
RCNN의 두단계를 거친 classifciation하는 과정을 통합함

Fast RCNN의 구조 : 
1. 전체 이미지를 CNN을 통과시켜 특징추출
2. Selective search를 통해 찾은 각각의 Roi를 Roi pooling을 진행시켜 고정된 같은 크기의 vector를 추출
3. 위의 Roi pooling된 vector들을 FCL을 통과시켜 두개의 브랜치로 나눔
4-1. 하나의 브랜치는 softmax를 통과시켜 Classify
4-2. 나머지 하나의 브랜치는 bounding box regression을 통해 selective search로 찾은 BB의 위치 조정

 
#### Faster RCNN


#### Mask RCNN


#### 1stage_detctor   
1stage detector는 classification과 localization을 동시에하는 과정으로 디텍팅한다.   ex : YOLO, SSD, FocalLoss, RefineDet

#### YOLO
googlenet을 backbone네트워크로 사용하였으며 feature extraction하며 Convlayer * 4, FCL * 2 하여 최종size = 7*7*30

YOLO의 특징 : gried 범위로 image를 각 7*7사이즈로 나누어 각 cell을 gridcell로 한다.   
region proposal을 사용하여 BB를 추출하는 Faster_RCNN과 다르게 FC를이용하여 바로 BB를 의 조정을 예측 

#### NMS : bounding box들을 비교하여 교집합이 큰 BB만 남겨서 대표적인 BB만 남기는 방식

### Segmentation   
   
#### Semantic segmentation   
: 객체를 클래스별로 분류 (ex : 차는 차종류끼리 분류)

#### Instance segmentation   
: 각 객체를 개별로 분류 (ex : 차를 차1, 차2, 차3등으로 분류)

## Faster_RCNN   
* FasterRCNN 간단 코드 with Tensorflow : https://hansonminlearning.tistory.com/32
* MaskRCNN git with Tensorflow : https://github.com/Kanghee-Lee/Mask-RCNN_TF
* Object detction in tensorflow_hub : https://tfhub.dev/s?module-type=image-object-detection 

