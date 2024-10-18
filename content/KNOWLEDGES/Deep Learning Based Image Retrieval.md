# Deep Learning Based Image Retrieval
---
## Deep Learning Based Image Retrieval의 분류
### 1. Supervision Type

- **Supervised**: 레이블이 붙은 데이터를 사용하여 모델을 훈련
- **Unsupervised**: 레이블이 없는 데이터에서 배운 패턴을 기반으로 모델을 훈련
- **Semi-supervised**: 일부 레이블이 있는 데이터와 많은 레이블이 없는 데이터를 혼합하여 훈련
- **Weakly-supervised**: 불완전하거나 부정확한 레이블을 가진 데이터로 모델을 훈련
- **Pseudo-supervised**: 기존 데이터를 기반으로 생성된 레이블로 모델을 훈련
- **Self-supervised**: 데이터 자체로부터 레이블을 생성하여 사용

### 2. Network Type

- **Convolutional**: 이미지 처리에 최적화된 네트워크.
- **Autocoder**: 입력 데이터를 압축하고 복원하여 특징을 학습
- **Siamese & Triplet Networks**: 데이터의 유사성을 측정하기 위해 pair 또는 triplet의 입력을 사용
- **Generative Adversarial Networks (GANs)**: 이미지 생성을 위해 두 개의 네트워크(생성자와 판별자)를 사용
- **Attention Network**: 특정 입력 부분에 집중하여 성능을 향상
- **Reinforcement Learning**: 행동 기반 학습을 통해 최적의 결정을 수행

### 3. Descriptor Type

- **Binary Descriptors**: 이진 형식을 사용하는 설명자로, 빠르게 비교 가능
- **Real-valued Descriptors**: 연속적인 값을 가지며, 더 많은 정보 표현이 가능
- **Aggregating Descriptors**: 여러 설명자를 조합하여 보다 강력한 특징을 형성

### 4. Retrieval Type

- **Cross-modal**: 서로 다른 형식의 데이터 간의 검색을 다루는 방식입니다.
- **Sketch-based**: 스케치를 통한 이미지 검색 시스템입니다.
- **Multi-label**: 하나의 이미지가 여러 레이블을 가질 수 있도록 하는 검색.
- **Instance**: 특정 객체 인스턴스를 검색합니다.
- **Object**: 특정 객체의 존재 여부를 기반으로 검색합니다.
- **Semantic**: 이미지의 의미를 기준으로 검색합니다.
- **Fine-grained**: 이미지의 세부 사항이나 작은 차이를 다루는 검색입니다.
- **Asymmetric**: 쿼리와 데이터 간의 비대칭적 관계를 고려합니다.

![[Pasted image 20241015122416.png]]
> [!QUOTE]
> **Fig. 2.** Taxonomy used in this survey to categorize the existing deep learning-based image retrieval approaches.  
> Source: "Deep Learning for Instance Retrieval: A Survey" 
> IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.  
> [https://doi.org/10.1109/TPAMI.2021.3072238](https://doi.org/10.1109/TPAMI.2021.3072238)

---
### Datasets used for 'CBIR'

| Dataset                  | Year | Classes | Training  | Test    | Image Type               |
| ------------------------ | ---- | ------- | --------- | ------- | ------------------------ |
| CIFAR-10 [45]            | 2009 | 10      | 50,000    | 10,000  | Object Category Images   |
| NUS-WIDE [46]            | 2009 | 21      | 97,214    | 65,075  | Scene Images             |
| MNIST [47]               | 1998 | 10      | 60,000    | 10,000  | Handwritten Digit Images |
| SVHN [48]                | 2011 | 10      | 73,257    | 26,032  | House Number Images      |
| SUN397 [49]              | 2010 | 397     | 100,754   | 8,000   | Scene Images             |
| UT-ZAP50K [50]           | 2014 | 4       | 42,025    | 8,000   | Shoes Images             |
| Yahoo-1M [51]            | 2015 | 116     | 1,011,723 | 112,363 | Clothing Images          |
| ILSVRC2012 [52]          | 2012 | 1,000   | ~1.2 M    | 50,000  | Object Category Images   |
| MS COCO [53]             | 2015 | 80      | 82,783    | 40,504  | Common Object Images     |
| MIRFlicker-1M [54]       | 2010 | -       | 1 M       | -       | Scene Images             |
| Google Landmarks [55]    | 2017 | 15 K    | ~1 M      | -       | Landmark Images          |
| Google Landmarks v2 [56] | 2020 | 200 K   | 5 M       | -       | Landmark Images          |
| Clickture [57]           | 2013 | 73.6 M  | 40 M      | -       | Search Log               |

CIFAR-10과 MNIST dataset만 각 category별 sample 수가 동일
(나머지 dataset들은 sample 수가 일정하지 못함)

---
