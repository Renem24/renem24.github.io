> “Emerging Properties in Self-Supervised Vision Transformers”
> 
> Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal Piotr Bojanowski, Armand Joulin

> [!summary] 
> DINO: self-**di**stillation with **no** labels
> self-supervised pre-training on ViT features
> DINO는 framework이다.


### Figures
- Figure 1
![[Pasted image 20240905112459.png]]


- Figure 2
![[Pasted image 20240905113036.png|300]]

- Figure 3 
![[Pasted image 20240905113621.png|400]]

**Algorithm 1** DINO PyTorch pseudocod w/o multi-crop
```
# gs, gt: student and teacher networks 
# C: center (K) 
# tps, tpt: student and teacher temperatures 
# l, m: network and center momentum rates 

gt.params = gs.params
for x in loader: # load a minibatch x with n samples 
	x1, x2 = augment(x), augment(x) # random views 
	
	s1, s2 = gs(x1), gs(x2) # student output n-by-K 
	t1, t2 = gt(x1), gt(x2) # teacher output n-by-K 
	
	loss = H(t1, s2)/2 + H(t2, s1)/2 loss.backward() # back-propagate 
	
	# student, teacher and center 
	updates update(gs) # SGD 
	gt.params = l*gt.params + (1-l)*gs.params 
	C = m*C + (1-m)*cat([t1, t2]).mean(dim=0) 
	
def H(t, s): 
	t = t.detach() # stop gradient 
	s = softmax(s / tps, dim=1) 
	t = softmax((t - C) / tpt, dim=1) # center + sharpen 
	return - (t * log(s)).sum(dim=1).mean()
```

## Abstract
self-supervised learning이 ViT(convnet의 대체재)에게 새로운 특성을 주는 건 아닐까라는 의문을 가졌다.
→ 따라서 ViT에 self-supervised learning을 적용했다.

1. self-supervised ViT feature들은 image의 semantic segmentation에 대한 explicit한 information을 담고있다.
	-  supervised ViT나 convnet으로는 이러한 information들이 명확히 드러나지 않았었다.

2. self-supervised ViT feature들은 k-NN classifier과 매우 잘 맞는다.

또 주목한 것들은 **[[Momentum Encoder]]**, **[[Multi-crop Training]]**, **ViT를 small patch들로 사용**하는 것이다.

이러한 아이디어를 모아서 만든 것이 DINO이다.

## Introduction
Visual Recognition 분야에서, [[Transformer]]는 [[CNN(Convolutional Neural Networks)]]에 대한 대안으로 부상하고 있다.
여기에 [[Natural Languae Processing(NLP)]]에서 주로 사용하는 training strategy을 적용한다.
이는 대규모의 data로 pre-training하고, target dataset으로 fine-tuning을 하는 방법이다.
-> 이러한 방식이 ViT(Vision Transformer)이다.

But, ViT의 단점들이 아직 장점보다 많다.
- 높은 계산비용, 많은 training data의 필요성, 유니크한 특성을 나타내지 못하는 feature들

이 논문에 대한 motivation은, NLP 분야에서 Transformer가 성공한 주요 요인이 [[self-supervised learning]]이라는 것이다.
ex. BERT, GPT

- self-supervised learning in NLP : sentence 내의 word들 사용 -> pretext task들을 생성
	-> richer learning signal을 제공
- supervised learning in NLP : 1 sentence -> 1 label(학습한 label들 중 하나)

image에서도 NLP와 유사하다.
- self-supervised learning in Image : 1 image -> 1 concept(학습한 label들 중 하나)
- supervised learning in Image : DINO(본 논문)의 내용

#### self-supervised ViT의 흥미로운 특성들
- self-supervised ViT feature들은 explicit하게 scene layout 정보, 특히 object boundary 정보를 담고 있다.
	- self-attention module의 마지막 블록에서 이를 볼 수 있다.
- self-supervised ViT feature들은 k-NN으로 쉽게 분류된다.
	- fine-tuning, linear classifier, data augmentation 사용 X

self-supervised learning에서 공통적으로 segmentation mask가 나타났지만, k-NN에 좋은 성능을 보이는 건 momentum encoder와 multi-crop augmentation 같은 확실한 요소를 결합했을 때에만 해당되었다.

ViT에 더 작은 patch들을 넣었을 때, 결과 feature의 성능이 향상되었다.


#### DINO의 concept
knowledge **di**stillation with **no** labels
DINO는 framework이다.

teacher network의 output을 직접 예측하는 것으로 self-supervised training을 단순화했다.
training에는 cross entropy loss를 사용한다.
- teacher network은 momentum encoder를 통해서 빌드한다.

teacher output에 centering과 sharpening을 사용해서 collapse를 피했다.
<-> 다른 모델들은 predictor, advanced normalization, contrastive loss를 사용했다.

DINO는 flexible하고 convnet과 ViT 모두 architecture 수정이나 normalization없이 사용할 수 있다.


#### Evalution
DINO+ViT는 ImageNet linear classification benchmark로 validate했다.
DINO+convnet에 사용한 convnet은 ResNet-50이다.


## Related work
### Self-supervised learning
