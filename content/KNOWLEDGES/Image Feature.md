# Feature(in Computer Vision)
## What is 'Feature' in Computer Vision?
image에서 추출한 정보를 수치적인 값으로 표현한 것
사람이 이해하기 어려움

image에서 추출된 feature는 원래 image보다 훨씬 낮은 dimension을 가짐
dimension이 줄어들었기 때문에 image 묶음을 처리하는 과정의 어려움이 감소함
s
feature는 descriptor라고 부르기도 함


## Feature의 종류
![[Pasted image 20241014102453.png|500]]
### Global Feature
전체 image에서 feature를 추출, image를 큰 틀에서 표현하고 image 전체를 일반화
global feature를 비교? -> 전체 image에 대한 평가

- 추출되는ㄴ 정보들:
	- contour representation
	- shape descriptor
	- texture features

- 사용되는 task: 
	- image retrieval
	- image classification
	- scene recognition

- 추출 방식:
	- SIFT
	- SUFT

### Local Feature
image의 patch들에서 feature를 추출
local feature를 비교? -> image의 각 부분에 대한 평가


- 추출되는 정보들:
	- texture in image patch
	- shape matrics
	- invariant moments
	-  Histogram Oriented Gradients (HOG)

- 사용되는 task: 
	- object detection
	- object tracking
	- image matching
	- 3D reconstruction


#### Local Feature의 예시
![[Pasted image 20241014112457.png|600]]

### Local & Global feature fusion
인식의 정확도를 향상시키지만, 계산 오버헤드가 발생(더 많은 연산량이 소요)
