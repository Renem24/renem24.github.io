# DINO(Meta)
---
#DINO #self-supervised_learning #ViT #knowledge_disilation

> "Emerging Properties in Self-Supervised Vision Transformers"  
> 2021-05  
> arXiv  
> Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jegou, Julien Mairal, Piotr Bojanowski, Armand Joulin  
> Facebook AI Research, Inria, Sorbonne University  
> https://arxiv.org/pdf/2104.14294

> [!abstract]  
> In this paper, we question if self-supervised learning provides new properties to Vision Transformer (ViT) that stand out compared to convolutional networks (convnets). Beyond the fact that adapting self-supervised methods to this architecture works particularly well, we make the following observations: first, self-supervised ViT features contain explicit information about the semantic segmentation of an image, which does not emerge as clearly with supervised ViTs, nor with convnets. Second, these features are also excellent k-NN classifiers, reaching 78.3% top-1 on ImageNet with a small ViT. Our study also underlines the importance of momentum encoder, multi-crop training, and the use of small patches with ViTs. We implement our findings into a simple self-supervised method, called DINO, which we interpret as a form of self-distillation with no labels. We show the synergy between DINO and ViTs by achieving 80.1% top-1 on ImageNet in linear evaluation with ViT-Base.
> 
> ViT에 self-supervised learning을 도입
> - self-supervised ViT feature들이 sementic segmentation에 대한 explicit한 정보들을 포함하게 됨
> - self-supervised ViT feature들은 k-NN classfier와 매우 잘 맞음


> [!note] 
> **DINO**: self-**di**stillation with **no** labels
> - self-supervised pre-training on ViT features
> - DINO는 framework


---
## 1. Introduction 

Visual Recognition분야에서, CNN대신 Transfromer, 즉, ViT를 사용하는 쪽으로 연구의 흐름이 바뀌어가고 있음

하지만 ViT의 단점들이 여전히 존재
- high computation cost
- require more training data
- feature들이 unique property들을 가지지 못함

#### Motivation
NLP에서 Transformer가 성공한 요인이 [[self-supervised learning]](ex. BERT, GPT)이므로, ViT에도 이를 적용해보자.

#### supervised learning vs. self-supervised learning
**supervised learning**:
- 하나의 대상으로부터 하나의 concept만 학습 가능
	- (NLP) 1 sentence -> 1 label
	- (CV) 1 image -> 1 concept

**self-supervised learning**
- 더 풍부한 learning signal을 학습 가능

### 연구과정에서 확인된 Self-supervised ViT의 특성

- self-supervised ViT feature들은 **explicit**한 **scene layout** 정보, 특히 **object boundary** 정보를 담고 있음
	- sementic segmentation 정보
- self-supervised ViT feature들은 **k-NN**으로 쉽게 분류 가능
	- fine-tuning, linear classifier, data augmentation 사용하지 않아도 좋은 분류 성능을 보임
		(78.3% top-1 accuracy on ImageNet)
	- momentum encoder, multi-crop augmentation을 사용한 경우에만 이러한 특성이 드러남
- ViT에 더 작은(더 많은) patch를 사용할수록 추출된 feature가 더 좋은 분류 성능을 보임

### DINO의 conept
"knowledge **di**stillation with **no** labels"

ViT를 위한 framework

---
## 2. Related Work
### Self-supervised learning

기존의 일반적인 방식은 [[Instance Classification]]으로 많은 image에 대해서는 적용이 어려웠음

- [[NCE(Noise Contrastive Estimator)]]
- [[BYOL]]

### Self-training & Knowledge distilation
- [[Self-Training]]
- [[Knowledge Distilation]]

self-supervised learning + knowledge distillation
- 이전 연구들:
	- pre-trained fixed teacher network 사용
	- model compression & performance gain
	- knowledge distillation을, self-supervised learning으로 post-processing 단계로 사용
- DINO:
	- training 과정에서 teacher network도 같이 학습되며 업데이트
	- training 과정에서 teacher와 student network가 같은 architecture를 가지고 둘다 distilation을 적용(**Codisilation**)
		- teacher는 student로부터 distilate됨

---
## 3. Approach
### 3.1. Self-Supervised Learning with Knowledge Distillation

- DINO가 사용한 **recent self-supervised approaches**:
	- "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments"
	- "Exploring Simple Siamese Representation Learning"
	- "A Simple Framework for Contrastive Learning of Visual Representations"
	- "Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning"
	- "Momentum Contrast for Unsupervised Visual Representation Learning"

- DINO가 사용한 **knowledge distilllation method**:
	- "Distilling the Knowledge in a Neural Network"


> [!figure 2]
> ![[Pasted image 20241017102834.png]]


##### Algorithm 1: DINO PyTorch pseudocode w/o multi-crop.
```
# gs, gt: student and teacher networks
# C: center (K)
# tps, tpt: student and teacher temperatures
# l, m: network and center momentum rates

gt.params = gs.params  # initialize teacher parameters from student

for x in loader:  # load a minibatch x with n samples
    x1, x2 = augment(x), augment(x)  # random views

    # forward passes for student and teacher networks
    s1, s2 = gs(x1), gs(x2)  # student output n-by-K
    t1, t2 = gt(x1), gt(x2)  # teacher output n-by-K

    # loss calculation and backpropagation
    loss = H(t1, s2)/2 + H(t2, s1)/2
    loss.backward()  # back-propagate

    # student, teacher, and center updates
    update(gs)  # SGD update for student
    gt.params = l * gt.params + (1 - l) * gs.params  # update teacher
    C = m * C + (1 - m) * cat([t1, t2]).mean(dim=0)  # update center

# loss calculation function
def H(t, s):
    t = t.detach()  # stop gradient for teacher
    s = softmax(s / tps, dim=1)  # softmax for student with temperature
    t = softmax((t - C) / tpt, dim=1)  # center and sharpen for teacher
    return -(t * log(s)).sum(dim=1).mean()  # cross-entropy loss

```
#### Knowledge distilation method in DINO (how to train Student network?)

student network을, teacher network의 output으로 train하는 learning paradaigm 
- **networks($g$)**:
	- student network $g\theta_s$ 
	- teacher network $g\theta_t$


주어진 input image $x$에 대해서, 두 network 모두 $K$ 차원(dimension)의 확률 분포(probability distribution)를 output으로 도출
- **output K-dim probaility distribution**($P$): network(g)의 output을 normalize 한 것
	- student network의 output $P_s$
	- teacher network의 output $P_t$
	$$P_s(x)^{(i)} = \frac{\exp \left( g_{\theta_s}(x)^{(i)} / \tau_s \right)}{\sum_{k=1}^{K} \exp \left( g_{\theta_s}(x)^{(k)} / \tau_s \right)}$$
	- $\tau_s$ ($> 0$):temperature parameter
		- output distribution의 sharpness를 조절
	- $P_t$도 위 식과 동일한 구성을 가짐($\tau_t$ 포함)

주어진 fixed teacher network($g\theta_t$)의 output distribution에 맞추기 위해, student network의 parameter $\theta_s$의 cross-entropy loss를 minimize
- parameter $\theta_s$의 cross-entropy loss
	$$\min_{\theta_s} H(P_t(x), P_s(x))$$
	- $H(a, b) = -a \log b$

- student 학습(self-supervised learning) 구체적인 방법
	1. **multi-crop strategy**으로 여러 view(crop)들을 생성
		- 주어진 하나의 image에 대해서, $V$개의 view(crop)를 생성
			- **global views 2개($x^g_1$ & $x^g_2$)** + **local views $V-2$개(더 작은 resolution으로 구성)**
	2. student network은 **모든 view들**을 사용 & teacher network은 **global view들**만 사용
		- student network은 [[Local-to-Global Correspondence]] 능력을 가짐

- multi-crop strategy를 적용한 parameter $\theta_s$의 cross-entropy loss
	$$\min_{\theta_s} \sum_{x \in \{x_1^g, x_2^g\}} \sum_{x' \in V, x' \neq x} H(P_t(x), P_s(x'))$$
	- $V$가 2인 경우에도 사용 가능

- DINO에서 사용하는 basic setup:
	- multi-crop strategy: multi-crop에 대한 standard setting에 따름
		- global view: resolution $224^2$(원본의 50% 이상)
		- local view: resolution $96^2$(원본의 50% 이하)
	- student & teacher:
		- student와 teacher는 같은 architecture $g$를 공유하고, 다른 parameter 값 $\theta_s$, $\theta_t$을 사용
	- student의 training:
		- Eq.(3)(위의 식)을 [[SGD(Stochastic Gradient Descent)]]를 사용하여 minimize

#### Teacher network
knoweledge distilation(teacher→student)과 달리, teacher network $g{\theta_t}$는 student network의 past iteration들로부터 만들어져야 함
Section 5.2에서 여러가지 방식들을 적용하였고, 이 중 [[EMA(Exponential Moving Average)]]를 사용하는 [[Momentum Encoder]] 방법이 가장 DINO에 적합함

- DINO에 적용한 Momentum Encoder:
	- update rule: $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$
		- $\lambda$는 0.996 $\rightarrow$ 1 로 cosine schedule에 따라 조정
		- "Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning"
		- 의미: teacher parameter에 student parameter를 약간씩 반영하여 update
	- 원래의 [[Momentum Encoder]](contrastive learning의 queue를 대체)가 사용되는 방식과 다르게 사용됨
	- contrastive learnig과 queue가 모두 없으므로, **self-training**의 **mean teacher**의 의미에 가까움

DINO의 teacher network은 exponential decay를 사용하는 [[Polyak-Ruppert Averaging]]과 유사한 [[Model Ensembling]] 형태를 보임

training동안, teacher는 student보다 더 높은 성능을 보였기에 student를 더 잘 가이드할 수 있었음

#### Network architecture
network $g$: $\ g = h \circ f$
- **Backbone** $f$:
	- [[ViT(Vision Transformer)]]
	- [[ResNet]]
- projection head $h$:
	- 3 layer [[MLP(Multi-layer Perceptron)]]
		- 2048 dim 
		-  +$\ell_2$ normalization + weighted normalized K dim FC layer
	- SwAV의 설계와 유사
		- ["The best of both worlds: Combining recent advances in neural machine translation"](https://arxiv.org/abs/1804.09849)

CNN과 달리, ViT는 Batch Normalization을 사용하지 않는 것이 default(BN-free)
- DINO도 BN-free

#### Avoiding Collapse
다른 self-supervised 방식들은 model collapse를 피하기 위해 아래와 같은 해결방법이 필요함
- [[operations to avoid COLLAPSE in self-supervised methods]]

DINO도 여러 normalization 방법들을 사용해서 안정화시킬 수 있지만,
momentum teacher의 output을 **centering**과 **sharpening**하는 것만으로도 collapse를 피할 수 있음
- centering:
	- 한 dimension이 지배적이게 되는 것을 방지
	- 균일한 분포가 되도록 할 수 있다는 단점
- sharpening:
	- <-> centering

##### Centering
batch size에 대한 의존성 감소: batch size와 상관없이 안정적으로 학습 가능

centering은 first-order batch statistics에만 기반하기 때문에,
teacher에 bias term $c$만 추가하는 것과 같은 의미를 가짐 $g_t(x) \leftarrow g_t(x) + c$ 
- center $c$는 [[EMA(Exponential Moving Average)]]의 형태로 update
	$$c \leftarrow mc + (1 - m) \frac{1}{B} \sum_{i=1}^{B} g_{\theta_t}(x_i)$$
	- $m$($>0$): rate parameter
	- B: batch size

##### Sharpening
단순히 teacher softmax normalization의 temperature $\tau_t$에 작은 값을 사용하는 것만으로 적용 가능

### 3.2. Implementation and evaluation protocols
 DINO training implementation details & evaluation protocols

#### Vision Transformer
DeiT의 implementation을 따름
- ["Training data-efficient image transformers & distillation through attention"](https://proceedings.mlr.press/v139/touvron21a)

> [!Table 1]
> ![[Pasted image 20241017155517.png|600]]

- image patch의 resolution $N \times N$:
	- N=16("/16")
	- N=8("/8)

DINO에서의 ViT 진행과정
- image patch들을 linear layer를 거쳐 embedding으로 변환
- learnable한 `[CLS]` token을 추가
- pre-norm을 사용하는 Transformer network에 집어넣기

#### Implementation details
Pre-training setup
- dataset: ImageNet 
- optimizer: adamw 
- batch size: 1024
- distributed environ: ViT-S/16 사용 시, 16 GPUs
- learning rate:
	- 첫 10 epoch(warmup)동안 다음 linear scaling rule에 따라서 linear하게 증가
		- $lr = 0.0005 ∗ \text{batchsize}/256$ 
	- 이후, cosine schedule에 따라서 decay
- weight: 0.04 $\rightarrow$ 0.4로 cosine schedule에 따라서 decay
- temperature:
	- $\tau_s$: 0.1로 설정 
	- $\tau_t$: 첫 30 epoch(warmup)동안 0.04 $\rightarrow$ 0.07로 linear하게 증가
- data augmentation: [[BYOL]]에 방식을 따름
	- color jittering
	- Gaussian blur
	- solarization
- multi-crop: bicubic interpolation를 사용해서 position embedding들을 image의 scale에 맞게 조정

#### Evaluation protocols
self-supervised learning의 표준 protocol
- frozed feature들에서 linear classifier를 학습