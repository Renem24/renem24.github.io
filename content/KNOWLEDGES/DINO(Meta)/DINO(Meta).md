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


## Introduction 

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

## Related Work
### Self-supervised learning.

기존의 일반적인 방식은 [[Instance Classification]]으로 많은 image에 대해서는 적용이 어려웠음

[[NCE(Noise Contrastive Estimator)]]