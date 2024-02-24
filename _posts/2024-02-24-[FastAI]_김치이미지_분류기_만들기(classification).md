---
layout: post
title: [FastAI] 김치이미지 분류기 만들기 (classification)
description:
date: 2024-02-24 21:00:00 +09:00
categories: [딥러닝, CV]
tags: [fastai, 딥러닝, 이미지 분류]
---

> **fastai를 활용해 이미지를 활용해 분류 문제를 해결하는 모델을 만든다.**  
> fastai 튜토리얼을 간략하지만 자세하게 설명하고, 최적화를 위한 실험을 진행한다.  
> 특히, fine_tune과 내가 epoch를 조절하는 기준에 대해 자세히 설명할 예정이다.
> {: .prompt-info }

## 0. FastAI란 무엇인가?

[##_Image|kage@EFCwV/btsBgyNm8VX/9sovhKcNHz1WDlk9av72Wk/img.png|CDM|1.3|{"originWidth":425,"originHeight":234,"style":"alignCenter","width":341,"height":188,"caption":"fast.ai는 딥 러닝 및 인공 지능에 중점을 둔 비영리 연구 그룹이다. 2016년 Jeremy Howard와 Rachel Thomas가 딥 러닝 민주화를 목표로 설립했다.","filename":"blob"}_##]

FastAI는 딥러닝을 위한 `고수준 API`를 제공하는 Python 라이브러리이다. **PyTorch**를 기반으로 하며, 딥러닝 모델을 더 쉽고 빠르게 구현할 수 있도록 설계되었다. **FastAI는 초보자도 쉽게 사용할 수 있는 직관적인 API**를 제공한다. 복잡한 모델도 몇 줄의 코드로 구현할 수 있다. Pytorch를 기반으로 만들었다 말했듯이, PyTorch의 모든 기능을 활용할 수 있다. 복잡한 딥러닝 모델을 빠르게 프로토타이핑하고 실험할 수 있게 도와주기 때문에 딥러닝 입문자에게 매우 편리하다. 다양한 데이터셋과 모델에 대한 사전 훈련된 가중치를 제공하기 때문에 빠른 시작을 가능하게 한다.

> 딥러닝을 위한 고수준 API를 제공하는 다른 라이브러리와는 어떤 점이 다를까?
>
> Keras는 주로 backend에서 TensorFlow를 사용하는 반면,  
> **fastai**와 PyTorch Lightning은 Pytorch를 기반으로 구축되었다.

> 이 게시글은 fastai 공식문서([https://docs.fast.ai/](https://docs.fast.ai/))를 주로 참고하였으며,  
> 딥러닝 모델에 대한 부분은 Pytorch 공식문서([https://pytorch.org/vision/main/models.html#using-models-from-hub](https://pytorch.org/vision/main/models.html#using-models-from-hub))를 참고하여 작성하였다.

[fastai - Welcome to fastai

fastai simplifies training fast and accurate neural nets using modern best practices

docs.fast.ai](https://docs.fast.ai/)

[Models and pre-trained weights — Torchvision main documentation

Shortcuts

pytorch.org](https://pytorch.org/vision/main/models.html#using-models-from-hub)

---

## 1. 모델 생성 계기

_밥을 먹다가 문득, 내 식탁에는 '김치'가 빠지지 않을정도로 김치를 즐겨 먹는 것 같다는 생각이 들었다. 좋아하는 음식을 '김치'라고 말할일은 없을 것 같지만, 반찬 중 김치가 없으면 허전할 것 같은 마음이 들었다. 다양한 김치가 있지만, 나는 보통 '배추김치'와 '오이소박이'를 자주 먹는다. 그렇다면 다른 김치반찬이 있다면 내가 구분할 수 있을까? 하는 재미있는 생각이 들었다. (내가 만든 모델보다 Score가 더 낮으면 어쩌지...?)_

_그래서, 실행에 옮겨보았다. **김치 사진을 찍으면 이게 어떤 김치인지 알려주는 분류 모델**을 만들어야겠다고 생각했다. 전부터 활용해보고싶었던 **fastai**를 활용해 간단하게 구현해보았다._

[##_Image|kage@4Mjyf/btsBc2oqDCK/FOHO2NG9cE6vpRlaepGkqk/img.png|CDM|1.3|{"originWidth":1394,"originHeight":406,"style":"alignCenter","width":588,"height":171}_##]

---

## 2. 데이터 수집

[https://www.aihub.or.kr/](https://www.aihub.or.kr/ "AI 허브")

[##_Image|kage@k3rY6/btsBfK7NEBa/dt98kRVhDC0YLkOGM7T2VK/img.png|CDM|1.3|{"originWidth":2494,"originHeight":918,"style":"alignCenter","link":"https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&amp;topMenu=&amp;aihubDataSe=data&amp;dataSetSn=79","isLinkNewWindow":true,"title":"AI 허브 내 &amp;#39;한국 이미지(음식)&amp;#39; 데이터를 다운받은 사이트","caption":"\bAI HUB에 있는 '한국 이미지(음식)'의 다양한 음식 이미지 중, '김치' 사진에 대한 데이터를 얻었다. 이미지를 클릭하면 해당 사이트로 이동한다."}_##]

다행히, **AI 허브**에 '김치'데이터가 있었다. **AI 허브**는 모델 구축을 위한 데이터를 대량으로 얻을 수 있는 사이트이다.

'한국 이미지(음식)' 데이터에서 다양한 음식 카테고리 중 '김치' 폴더에 데이터를 구했다. 각 '카테고리'에 대해 이미지가 1,000장씩 있었는데, 코드를 효율적으로 실행하기 위해 각 카테고리의 이미지 인덱스 \[0:500\]는 살리고, \[500:\] 이후는 삭제했다. 따라서 내가 학습한 데이터의 구조는 다음과 같다.

```
김치/
│
├── 배추김치/
│   ├── 이미지1.jpg
│   ├── 이미지2.jpg
│   ├── ...
│   └── 이미지500.jpg
│
├── 오이소박이/
│   ├── 이미지1.jpg
│   ├── 이미지2.jpg
│   ├── ...
│   └── 이미지500.jpg
│
├── 파김치/
│   ├── 이미지1.jpg
│   ├── 이미지2.jpg
│   ├── ...
│   └── 이미지500.jpg
│
├── 열무김치/
│   ├── 이미지1.jpg
│   ├── 이미지2.jpg
│   ├── ...
│   └── 이미지500.jpg
│
├── 깍두기/
│   ├── 이미지1.jpg
│   ├── 이미지2.jpg
│   ├── ...
│   └── 이미지500.jpg
│
├── 부추김치/
│   ├── 이미지1.jpg
│   ├── 이미지2.jpg
│   ├── ...
│   └── 이미지500.jpg
│
├── 무생채/
│   ├── 이미지1.jpg
│   ├── 이미지2.jpg
│   ├── ...
│   └── 이미지500.jpg
│
├── 백김치/
│   ├── 이미지1.jpg
│   ├── 이미지2.jpg
│   ├── ...
│   └── 이미지500.jpg
│
├── 나박김치/
│   ├── 이미지1.jpg
│   ├── 이미지2.jpg
│   ├── ...
│   └── 이미지500.jpg
│
├── 갓김치/
│   ├── 이미지1.jpg
│   ├── 이미지2.jpg
│   ├── ...
│   └── 이미지500.jpg
│
└── 총각김치/
    ├── 이미지1.jpg
    ├── 이미지2.jpg
    ├── ...
    └── 이미지500.jpg
```

---

## 3. 개념적으로 어떤 모델을 만들지 설계해보자!

[##_Image|kage@dvnEn6/btsBgyOzUqi/DKLnARqpkQ7f8I0AQtjMmk/img.png|CDM|1.3|{"originWidth":872,"originHeight":774,"style":"alignCenter","width":263,"height":233}_##]

---

모델은 본질적으로 인간의 편의를 돕고, 효율성을 극대화하기 위한 도구이다. 여기서 '도구'는 인간의 노동을 보조하는 것이 주된 목적이며, 그 과정에서 시간과 자원을 절약하게 해준다. 따라서, 모델을 효율적으로 사용하기 위해서는 '**내가 가진 데이터**'와 그 데이터를 응용해 얻을 수 있는 '**내가 원하는 결과물**'에 대한 정의를 명확하게 할 수 있어야한다.

> **INPUT** : '김치 이미지'  
>                ('배추김치', '오이소박이', '파김치', '열무김치', '깍두기', '부추김치', '무생채', '백김치', '나박김치', '갓김치', '총각김치')  
>  **OUTPUT**  : (모델이 학습하지 않은) 새로운 김치 사진에 대해 '**어떤' 김치인지 분류 결과를 알려준다.**

**그렇다면, 모델은 각 이미지가 '어떤'김치인지 학습을 하고, 새로운 김치사진에 대해 어떤 김치인지 알려주어야한다.**

---

## 4. 코드 소개

### 4-1) 필요한 라이브러리 설치

개발 환경은 Google Colab에서 진행했다.

fastai 라이브러리가 설치되어있지 않기에, 코드 환경에 fastai 라이브러리를 설치했다.

```bash
# fastai 라이브러리를 설치한다.
! pip install fastai
```

```python
# vision(이미지) 데이터를 다룰 예정 -> 관련 library를 모두 import 해준다.
from fastai.vision.all import *

# 프로젝트 위치
import pathlib
import os

# 메모리를 효율적으로 사용하기 위해, 경고 옵션을 '무시'한다.
import warnings
warnings.filterwarnings(action='ignore')

# google colab 환경에서 작업
# 파일크기(이미지 5500장)가 컸기 때문에, 올려두는 것 보다 구글 드라이브에 mount 해주는것이 속도면에서 더 빨랐다.
from google.colab import drive
drive.mount('/content/drive')
PROJECT_DIR = "/content/drive/MyDrive/프로젝트파일위치"
```

---

### 4-2) 데이터를 입력하기 위한 사전 작업

김치의 종류를 확인하기 위해, 폴더 이름을 확인하는 코드를 작성한다.

```python
os.listdir(PROJECT_DIR)  # 김치 종류 폴더 이름
```

[##_Image|kage@GW450/btsBjppySXC/BEjsJppg7z8CswImTEK051/img.png|CDM|1.3|{"originWidth":226,"originHeight":442,"style":"alignCenter","width":107,"height":209,"alt":"os.listdir(PROJECT_DIR)의 코드 실행 결과","caption":"os.listdir(PROJECT_DIR)의 코드 실행 결과"}_##]

김치 종류가 11가지라는 것을 코드로 확인할 수 있다.

```python
# get_image_files()를 활용해 path안에 담긴 파일 위치의 image path를 모은다.
fnames = get_image_files(PROJECT_DIR)
```

이미지의 정보는 폴더의 이름을 통해 알 수 있다.

이미지에 메타데이터로 직접 분류 라벨링이 되어있는것이 아니라, 종류별로 폴더에 정리되어있는것이기 때문에,

**이미지 파일 경로에서 폴더 directory 이름을 구해 이름을 슬라이싱해서 정답 라벨링을 구해야한다.**

```python
fnames[4000].parts  # 라벨 정답 정보를 얻기 위해 경로 분석하기

# 결과
# ('/', 'content', 'drive', 'MyDrive', '폴더명', '김치', '백김치', 'Img_034_0203.jpg')
```

4000번째 파일은 '백김치'에 대한 203번째 이미지라는 것을 확인할 수 있다.

마지막에서 두번째 인덱스에서 '김치 종류'에 대한 정보를 확인할 수 있고, 마지막 인덱스에서 '이미지이름'을 확인할 수 있기 때문이다.

그렇다면, 경로의 \[-2\]인덱스를 각 파일의 경로에 있는 이미지를 불러와 mapping 하는 작업이 필요하다.

```python
# 라벨링 정보를 불러오는 함수를 정의한다.
def label_func(fname):
    return str(fname.parts[-2])

# 데이터 블록 객체를 생성한다.
dblock = DataBlock(get_items = get_image_files,
                   get_y     = label_func)
```

fasfai의 `DataBlock()` 객체는 **데이터를 로드하고 변환하는 과정을 간소화하고,**

**데이터셋을 구성하는 방법을 정의하는 데 사용된다.** `DataBlock()`의 하이퍼파라미터는 다양한 요소가 있는데, 살펴보고 정의해보자.

다른 설정을 시도해보고싶다면, 공식문서를 참고하길 바란다.

method가 새롭게 업데이트 되는 경우가 더러 있기 때문에 혹시 오류가 발생한다면, **공식문서를 참고하기를 추천**한다.

[fastai - Data block

High level API to quickly get your data in a DataLoaders

docs.fast.ai](https://docs.fast.ai/data.block.html)

---

### 4-3) 모델의 파이프라인(pipeline) 구성

데이터 블록 `DataBlock()` 객체에 대해 더 자세히 알아보자.

**`DataBlock()`는 데이터를 모델에 공급하기 위한 파이프라인을 구축하는 역할을 한다.**

_**데이터 전처리, 데이터셋 구성, 배치 처리 및 데이터 증강(데이터 증식)과 같은 작업을 조율하여**_

_**효율적인 데이터 로딩 및 학습 과정을 가능하게 한다.**_

```python
# 데이터 블록 생성
datablock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                        get_items=get_image_files,
                        get_y = label_func,
                        splitter=RandomSplitter(valid_pct=0.2, seed=42),
                        item_tfms=Resize(460),  # 큰 사이즈로 초기 리사이즈
                        batch_tfms=[*aug_transforms(size=224, min_scale=0.75),
                        Normalize.from_stats(*imagenet_stats)]
)
# 데이터 로더 생성
dls = datablock.dataloaders(PROJECT_DIR)

# 데이터 확인
dls.show_batch(max_n=9)
```

[##_Image|kage@dMoY1h/btsBjenasUZ/IKFfYgDpQvGwYC0Eqy4GPK/img.png|CDM|1.3|{"originWidth":1436,"originHeight":1470,"style":"alignCenter"}_##]

> **blocks = (ImageBlock, CategoryBlock**) : 이미지를 다루고, '분류'문제이기에 명시해둔다.
> ImageBlock 은 입력 데이터(이미지)를 나타내고, CategoryBlock 은 타겟 데이터(레이블)를 나타낸다.
>
> **get_items &** **get_y** : _데이터셋에서 아이템을 어떻게 가져올지 정의한다._  
> 예를 들어, get_image_files 함수는 디렉토리에서 모든 이미지 파일을 가져오는 데 사용된다.
>
> **splitter** : 데이터를 훈련 세트와 검증 세트로 어떻게 나눌지 정의한다. RandomSplitter 는 데이터를 무작위로 분할한다.
>
> **item_tfms** : 각 아이템에 개별적으로 적용되는 변환을 정의한다. (이미지 크기 조정, 데이터 증강 등)
>
> **batch_tfms** : 배치 단위로 적용되는 데이터 변환을 정의한다. (이미지 크기 조정, 데이터 증강 등)
>
> **Normalize.from_stats(\*imagenet_stats)** : _데이터 정규화_  
> _정규화 과정은 모델이 다양한 이미지 데이터에 대해 더 잘 일반화하고, 학습 과정에서 더 빠르고 안정적으로 수렴하도록 돕는다._

> **이미지 데이터 size**를 정하는 작업의 의미
>
> [##_Image|kage@b6IK2h/btsBjdCMQCm/iT9oe3k9J4NnCqK1keWp3K/img.png|CDM|1.3|{"originWidth":950,"originHeight":110,"style":"alignCenter","width":654,"height":76,"caption":"위의 코드를 더 꼼꼼하게 설명해보자면,"}_##]
>
> _**초기에 큰 이미지로 시작하여 모델이 미세한 특징을 학습**하도록 하고,_  
> _(많은 정보 활용 가능, 그러나 계산 비용이 높다.)  
> _  
> _학습 효율성을 높이기 위해 후속 배치에서 **이미지 크기를 줄이고 데이터 증강을 적용**했다._  
> (계산 효율성을 높이면서 모델 학습에 필요한 크기로 이미지를 줄였다.)

```python
# 카테고리 라벨 종류 확인하기
dls.vocab
```

[##_Image|kage@DusPb/btsBgJbJOad/T08EkkyqG8wCd8OKEjSvfK/img.png|CDM|1.3|{"originWidth":1740,"originHeight":64,"style":"alignCenter","caption":"dls.vocab에 저장된 라벨의 종류."}_##]

---

### 4 - 4) 이미지 분류 모델 생성

```python
learn = vision_learner(dls, resnet18, metrics=[accuracy, error_rate, Recall(average='macro'), Precision(average='macro')])
```

모델을 선택하는 방법에 대해 자세히 서술해보겠다.

fastai는 앞서 설명했듯이, PyTorch를 기반으로 만들어졌기 때문에 PyTorch의 모델을 활용할 수 있다.

(pytorch model 공식문서 참고)

[Models and pre-trained weights — Torchvision main documentation

Shortcuts

pytorch.org](https://pytorch.org/vision/main/models.html#using-models-from-hub)

[##_Image|kage@bczu3h/btsBiRlkPfV/CYO2mmjvVxMtaz6FbXKQdk/img.png|CDM|1.3|{"originWidth":1188,"originHeight":1310,"style":"alignCenter","width":425,"caption":"사이트를 들어가보면, 다양한 딥러닝 분류 모델이 있는데,"}_##][##_Image|kage@ZuKks/btsBipJirCr/pMFKCjpkkkQysU9lK1tnx1/img.png|CDM|1.3|{"originWidth":1772,"originHeight":1282,"style":"alignCenter","width":659,"height":477,"caption":"예를 들어, EFFICIENTNETV2를 활용하고자 해당 모델을 눌러주면, pytorch (fastai)에서 활용 가능한 모델 명이 나온다. 이 중 가장 작은 모델인 'effieicnetnet_v2_s' 를 활용하기위해 별도로 모델 라이브러리를 import 하거나 다른 모듈을 가져다쓰지 않아도 된다. 'effieicnetnet_v2_s' 이런식으로 string타입이 아니라, 그냥 effieicnetnet_v2_s&amp;nbsp; 이렇게 적어주면 된다."}_##]

모델을 내 데이터에 맞게 변형하고 학습하기 전에, **적절한 학습률(learning rate)을 찾는 메서드**를 사용해보겠다.

```python
# 적절한 학습률을 찾기 위해 사용, 이 메서드는 다양한 학습률에서 손실을 계산하여 가장 좋은 학습률을 추천한다.
learn.lr_find()
```

[##_Image|kage@7eH0L/btsBiTpYNG7/AGxVNyxqw7k1mQki1uDwfk/img.png|CDM|1.3|{"originWidth":567,"originHeight":437,"style":"alignCenter"}_##]

.lr_find()를 통해 찾은 최적의 Learning Rate는 파인튜닝을 할 때 사용할 수도 있고, 모델 학습을 할 때 사용할수도 있다.

---

- _fine_tune()에 적용할 때 learning_rate의 하이퍼파라미터 변수 명 예시_

```python
learn.fine_tune(freeze_epochs = 1, epochs=3, base_lr=1e-4)
```

- _fit_one_cycle()에 적용할 때 learning_rate의 하이퍼파라미터 변수 명 예시_

```python
learn.fit_one_cycle(1, lr_max=1e-4)
```

---

파인튜닝을 하고, 모델을 훈련시켜 학습시키기 이전에 위에서 만든 **모델의 구조를 확인**해보자.

fastai는 PyTorch를 기반으로 만들어졌기 때문에, 밑에 코드는 PyTorch 문법을 따른다.

```python
# 모델 구조 확인하기
learn.model
```

```
Sequential(
  (0): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (1): Sequential(
    (0): AdaptiveConcatPool2d(
      (ap): AdaptiveAvgPool2d(output_size=1)
      (mp): AdaptiveMaxPool2d(output_size=1)
    )
    (1): fastai.layers.Flatten(full=False)
    (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.25, inplace=False)
    (4): Linear(in_features=1024, out_features=512, bias=False)
    (5): ReLU(inplace=True)
    (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.5, inplace=False)
    (8): Linear(in_features=512, out_features=11, bias=False)
  )
)
```

_전체를 완전히 이해하고 활용할 필요는 없으며, 각 레이어의 역할과 순서 등을 읽고 이해하는 수준으로 넘어가면 된다._

---

### 4-5) 📖 파인 튜닝이란?

_파인튜닝을 하기 전에, 파인튜닝이 무엇인지 간략하게 읽고 넘어가자._

[##_Image|kage@bp3jiX/btsBfKPdVN7/x6M6ipCjFXldfgQJZSZCEk/img.png|CDM|1.3|{"originWidth":215,"originHeight":235,"style":"alignCenter","width":264,"height":289}_##]

_**분류를 위한 FC(Fully-connected) 레이어를 추가하고 이 레이어를 파인튜닝하여 해당 작업에 맞게 조정할 수 있다.**_

_즉, 레이블이 없는 대규모 데이터에서도 학습된 모델의 일반적인 이미지 특성을 활용하면서도 새로운 분류 작업을 수행할 수 있게 된다. Fine-tuning은 선행학습된 모델의 일부 또는 전체 파라미터를 새로운 작업에 맞게 업데이트하는 과정이므로, **비교적 적은 데이터로도 효과적인 모델을 학습할 수 있다.**_

---

### **4-6) 파인튜닝 하기**

```
learn.fine_tune(epochs=5, freeze_epochs=1)
```

[##_Image|kage@wyOT2/btsBgI4zYrb/tsQKqIUywMTL2lXmhImWv0/img.png|CDM|1.3|{"originWidth":1496,"originHeight":496,"style":"alignCenter"}_##]

> 참고로, 마지막 epoch은 모델을 훈련하기에 좋은 epochs는 아니다.

> _**train_loss의 감소 폭**이 **valid_loss의 감소 폭**을 앞질렀고, **train_loss**와 **valid_loss**의 격차가 벌어졌기 때문이다._  
> _epochs는 이런 방식으로 평가지표를 봐가며 성능 최적화를 위해 노력해야한다.  
> _  
> **_5번째 epoch보다, 오히려 4번째 epoch를 채택하는게 더 나은 선택이다.  
> _**  
> _나머지는 비슷한데다, 5번째 epoch의 **train_loss**가 4번째보다 더 낮기 때문에 더 좋은 선택처럼 보일 수 있으나,_  
> _**train_loss**와 **valid_loss**의 차이가 벌어지는게 보인다.  
> _  
> _이렇게 되면 **train 데이터**에 **Overfitting** 될 가능성이 매우 높아진다. ➡️ **새로운 데이터에 대해 높은 예측력을 보여주지 못한다.**_

> \_**모델을 만들때는 훈련 데이터뿐만 아니라  
> 새로운 데이터에도 일반화할 수 있는 능력을 갖도록** \*\*모델을 구축하고 향상시켜야한다.
>
> \*\*\_
>
> _모델이 다양한 데이터와 다양한 상황에서 효과적으로 작동할 수 있도록 해야한다._  
> _모델은 가상의 데이터 혹은 추상적인 문제가 아닌, **현실의 문제를 해결하는데 도움이 되어야하기 때문에**_  
> _모델이 내가 가진 데이터에 오버피팅되지 않도록 유의해야한다._

파인튜닝 이후에 모델 구조는 어떻게 변했을까?

변화된 모델 구조를 다시 확인해보자.

```
learn.model
```

```
## 결과 ##

Sequential(
  (0): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (1): Sequential(
    (0): AdaptiveConcatPool2d(
      (ap): AdaptiveAvgPool2d(output_size=1)
      (mp): AdaptiveMaxPool2d(output_size=1)
    )
    (1): fastai.layers.Flatten(full=False)
    (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.25, inplace=False)
    (4): Linear(in_features=1024, out_features=512, bias=False)
    (5): ReLU(inplace=True)
    (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.5, inplace=False)
    (8): Linear(in_features=512, out_features=11, bias=False)
  )
)
```

파인튜닝을 거친 뒤 모델의 구조에 변화가 생겼다.

앞서 잠깐 언급했듯, 파인튜닝을 통해 모델은 **초기에는 기존의 지식을 활용**하면서도,

내가 가진 데이터와 작업에 맞게 점차적으로 **최적화**되는 과정을 거치게 된다.

**결과적으로 모델은 내 데이터에 최적화된 구조로 발전하며, 높은 정확도와 성능을 제공한다.**

_모델 레이어의 어떤 부분이 어떻게 수정되었고 레이어가 내가 가진 데이터에 최적화되는 방식을 꼼꼼히 살펴보자._

```python
Sequential(
  (1): Sequential(
  # AdaptiveConcatPool2d: 이 레이어는 입력 데이터의 공간 차원을 동적으로 조정하여 평균 풀링과 최대 풀링을 수행한다.
  # 두 풀링의 결과를 concatenate하여 하나의 텐서를 생성한다.
    (0): AdaptiveConcatPool2d(
      (ap): AdaptiveAvgPool2d(output_size=1)
      (mp): AdaptiveMaxPool2d(output_size=1)
    )
    # fastai.layers.Flatten: 이 레이어는 다차원의 텐서를 평평하게(1차원으로) 펼치는 역할을 한다.
    # 주로 풀링 레이어 다음에 사용되어 평탄한 특징 벡터를 생성한다.
    (1): fastai.layers.Flatten(full=False)
    # BatchNorm1d: 이 레이어는 1차원 데이터를 위한 배치 정규화를 수행한다. 배치 정규화는 신경망의 안정성과 학습 속도를 향상시키는 데 도움이 된다.
    (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # Dropout: 이 레이어는 학습 중에 뉴런을 무작위로 비활성화하여 과적합을 줄이는 데 도움을 준다. 여기서는 p 매개변수로 드롭아웃 비율을 조정한다.
    (3): Dropout(p=0.25, inplace=False)
    (4): Linear(in_features=1024, out_features=512, bias=False)
    (5): ReLU(inplace=True)
    (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.5, inplace=False)
    # Linear: 이 레이어는 완전 연결된 (fully connected) 레이어를 나타내며, 입력 특징을 출력 클래스로 매핑하는 데 사용된다.
    # 여기서는 출력 클래스가 11개인 fully connected 레이어를 정의하고 있다.
    (8): Linear(in_features=512, out_features=11, bias=False)
  )
)
```

> **1. Adaptive Pooling 레이어**의 추가로, 모델은 다양한 입력 이미지 크기와 해상도에 더 잘 적응할 수 있게 되었다.  
> 파인튜닝 후에도 입력 다양성을 처리할 수 있어 더 유연한 모델이 되었다.
>
> **2. Batch Normalization과 Dropout 레이어**의 추가로, 모델의 학습이 더 안정적으로 이루어진다.  
> 이것은 파인튜닝 후에도 새로운 데이터셋에 대해 더 효과적인 학습을 가능하게 한다.
>
> **3. Flatten 레이어**의 추가로, 이전 훈련된 모델의 특징 맵을 1차원 벡터로 변환하여 분류 레이어에 전달할 수 있다.  
> 이것은 파인튜닝 작업에 필수적인 변화다.
>
> **4. Linear 출력 레이어**의 변경으로 모델은 파인튜닝 대상 작업에 맞게 출력 클래스 수를 조정한다.  
> 모델을 다른 분류 작업에 쉽게 재사용할 수 있도록 한다.
>
> **5. Batch Normalization과 Dropout 레이어**의 추가로 파인튜닝 이후에도 모델은 과적합을 효과적으로 방지하며,  
> 새로운 데이터셋에 잘 일반화될 수 있도록 돕는다.

> **freeze_epochs와 epochs가 뭐지? 왜 굳이 나눈거지?**
>
> **freeze_epochs 단계**에서는 모델의 FC 레이어를 학습하고 Convolutional 레이어는 고정된다. 이렇게 함으로써, **모델의 초기 미세 조정은 주로 데이터와 관련된 특별한 패턴 및 클래스를 파악하는데 집중한다.** Convolutional 레이어는 이미지의 특징을 추출하는 역할을 하며, 이러한 특징은 대부분의 이미지 분류 작업에서 유용하게 유지된다.

**epochs 단계**에서는 모델의 모든 레이어를 함께 학습한다. 이 단계에서는 **모델이 전체 구조를 새로운 작업에 맞게 미세 조정하고 더 많은 데이터로 성능을 향상시킨다.** 이 단계에서는 초기 미세 조정에서 학습된 FC 레이어와 함께 Convolutional 레이어도 조금씩 조정된다. **모델은 이전에 파악한 데이터 관련 패턴을 보완하고 미세 조정을 통해 새로운 작업에 더 적합하게 맞추려고 노력한다.**

모델의 관점에서 설명해보자면,

> _초기에 모델을 빠르게 적응시키기 위해 FC 레이어를 먼저 학습하면 **적은 양의 데이터로도 좋은 성능을 얻을 수 있다.**_  
> _그 후 전체 모델을 미세 조정하면서 더 많은 데이터를 활용하여 모델을 개선하는 방식으로 나누어서 학습하면 더 **효과적으로 모델을 조정**하고 **학습 속도**를 높이며 성능을 향상시킬 수 있다._

> [##_Image|kage@by2UmT/btsBl54Xv0v/PcsmCZTKDMWJF1hIub64c1/img.png|CDM|1.3|{"originWidth":512,"originHeight":512,"style":"alignCenter","width":165,"height":165}_##]
>
> 여기서 fastai의 편리함을 느낄 수 있다.  
> **단지 몇줄의 직관적인 고수준 API만으로 모델을 만들고, 파인튜닝까지 할 수 있다!**  
> 모델의 구조를 파악하고 한줄한줄 수정하고 이해할 필요가 없다.

---

### 4-7) 모델 학습

```
learn.fit_one_cycle(3, lr_max=1e-4)
```

fit_one_cycle() 메서드는 "one cycle learning rate policy"를 사용하여 모델을 학습한다.

학습률을 동적으로 조절하는 방법으로, _lr_max_ 를 통해 최대 학습률을 지정할 수 있다.

**이 방법은 학습률을 처음에는 점진적으로 증가시키고, 이후 점진적으로 감소시키는 방식으로 작동한다.**

이러한 접근 방식은 모델이 **더 빠르게 수렴**하도록 돕고, 일반적으로 **더 나은 성능**을 달성할 수 있다.

---

### 4-8) 결과를 시각적으로 확인하기

```
learn.show_results(max_n=9)
```

모델이 실제 정답을 맞춘 경우, 초록색으로 나오지만

모델이 실제 정답을 맞추지 못한 경우에는 빨간색으로 나온다.

[##_Image|kage@bxUBul/btsBiQ8WvmC/hTeV9atQnMy10gAQ52u0GK/img.png|CDM|1.3|{"originWidth":1426,"originHeight":1498,"style":"alignCenter","width":470,"height":494}_##]

---

### 4-9) 모델 성능 평가지표로 모델의 성능 확인하기

```
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(7,7))
```

**Confusion Matrix**를 통해 모델의 분류 능력을 파악할 수 있다.

[##_Image|kage@c8bLIJ/btsBgwpKgv5/ORjWiQkJJnMfRQdDmKG9lK/img.png|CDM|1.3|{"originWidth":1366,"originHeight":1404,"style":"alignCenter","width":489,"height":503}_##]

```
interp.most_confused()  # 잘못 분류한 case 수가 출력된다.
```

[##_Image|kage@bFuqUZ/btsBmQUajgd/F73JSQJdkUWiIgYsl56uUK/img.png|CDM|1.3|{"originWidth":460,"originHeight":974,"style":"alignCenter","width":189,"height":400}_##]

```
interp.plot_top_losses(k=9) # 가장 losses가 큰 이미지 출력해보기
```

[##_Image|kage@DZvYK/btsBjBi73QH/flu6Ek5LVQBxDhOtLWVB91/img.png|CDM|1.3|{"originWidth":1854,"originHeight":1596,"style":"alignCenter","width":545,"height":469}_##]

```
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# testset에 대한 예측결과확인

y_pred, y_true = learn.get_preds()
print(confusion_matrix(y_true, np.argmax(y_pred, axis=1)))
print(classification_report(y_true, np.argmax(y_pred, axis=1)))
```

[##_Image|kage@lA3Fd/btsBlda06S5/ruQzldwVD1CRZJz0VzSrhK/img.png|CDM|1.3|{"originWidth":928,"originHeight":1014,"style":"alignCenter","width":420,"height":459}_##]

---

### 4-10) 모델 외부로 추출하기

```
# 모델 추출하기
# () 에는 모델의 저장 위치를 경로로 입력한다.
learn.export('identifying_Kimchi_model.pkl')
```

이 메서드를 사용하면 모델의 가중치(weights)와 모델 아키텍처(architecture)를 포함한 모델 전체를 파일로 내보낼 수 있다.

_모델을 내보내면 해당 파일에는 모델 아키텍처와 학습된 가중치가 저장되므로,_

**_나중에 불러와서 예측을 수행하거나 추가적인 학습을 진행할 수 있다._**

```
# 내보낸 모델을 불러올 때는 load_learner 함수를 사용한다.
learn = load_learner('identifying_Kimchi_model.pkl')
```

_➡️ **모델을 저장하고 다시 불러올 수 있으므로 모델을 보다 효과적으로 관리하고 재사용할 수 있다.**_

---

### 4-11) 모델 성능을 높이기 위한 실험

|     | item_tfms   | batch_tfms                                                                                                            | 정규화 | Model             | fine_tune                                              | fit_one_cycle    |
| --- | ----------- | --------------------------------------------------------------------------------------------------------------------- | ------ | ----------------- | ------------------------------------------------------ | ---------------- |
| 1   | Resize(460) | \[\*aug_transforms(size=224, min_scale=0.75)                                                                          | 적용   | resnet18          | epochs=3, freeze_epochs=1                              | (2, lr_max=1e-4) |
| 2   | Resize(460) | \[\*aug_transforms(size=224, min_scale=0.75)                                                                          | 적용   | efficientnet_v2_s | epochs=3, freeze_epochs=1                              | (2, lr_max=1e-4) |
| 3   | Resize(460) | aug_transforms(size=224, min_scale=0.75, flip_vert=True, max_rotate=20, max_zoom=1.2, max_lighting=0.3, max_warp=0.2) | 적용   | resnet50          | epochs=3, freeze_epochs=1, base_lr=1e-4, pct_start=0.3 | 2, lr_max=1e-4   |

[##_Image|kage@bnkqz4/btsBiPIY3Re/GFO3FJ9EfYOfbA01lJDyrk/img.png|CDM|1.3|{"originWidth":1492,"originHeight":178,"style":"alignCenter","caption":"첫번째"}_##][##_Image|kage@bgVfL0/btsBim8djqu/VbPYAnYXBKb8c4FRDWmiV1/img.png|CDM|1.3|{"originWidth":1484,"originHeight":196,"style":"alignCenter","caption":"두번째"}_##][##_Image|kage@X9QNV/btsBmSdsGzY/WMRMvsKUs5fXQSERxAKCR1/img.png|CDM|1.3|{"originWidth":1488,"originHeight":182,"style":"alignCenter","caption":"세번째"}_##]

오히려 데이터를 증강하고, 더 무거운 모델을 사용할수록 성능이 더 떨어졌다.

[##_Image|kage@bp477B/btsBfKPjZSf/qAo1486QcPJn5K433hZDXK/img.png|CDM|1.3|{"originWidth":1802,"originHeight":1596,"style":"alignCenter","width":607,"height":538}_##]

잘 분류하지 못한 이미지를 확인해보니, 김치 이외에 '배경'사진 처리가 관건이었던 것을 알 수 있다.
