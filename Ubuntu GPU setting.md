# 설치하는 순서 및 방법

### 1) Nvidia Driver 설치(설치가능한 driver찾고 Cuda, Cudnn, TF & Pytorch 호환되는 버전의 driver)
### 2) Cuda Toolkit설치(GPU Compute Compatibility 호환가능한 버전)
###   2-1) Cuda설치 후 PATH설정 (PATH설정 해주어야 nvcc -V로 Cuda 버전확인 가능)
### 3) Cudnn 설치(Cuda와 Compatibility 확인)
###   3-1) chmod a+r로 실행가능하게 권한 부여
### 4) tensorflow, tensorflow gpu 설치 : pip install tensorflow, pip install tensorflow-gpu
###   4-1) tf에서 device나열 code로 GPU인식 확인 / gpu_is_avaiable비추천

	
