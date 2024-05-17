---
layout: post
title: Chat Vector | A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages 리뷰
description:
date: 2024-05-17 09:00:00 +09:00
categories: [논문 리뷰]
tags: [LLM, Chat Vector]
# image: # preview img (해상도가 1200 x 630인 이미지)
#   path: /path/to/image
#   alt: image alternative text
# pin: true # 게시글 고정
---

이 논문의 원문은 다음과 같다.  
<https://arxiv.org/pdf/2310.04799>

{: .prompt-tip}

> 논문의 핵심 아이디어는 다음과 같다.
> 기존 모델에서 파생된 CP모델에 추가적인 벡터를 추가하면 특별한 학습 없이도 기존의 대규모 언어 모델에 대화 기능을 부여하고 명령 수행 능력과 인간의 가치에 맞추어 조정할 수 있다는 것입니다.

![image](https://github.com/uujeong/uujeong.github.io/assets/86465999/d56bbe09-3c13-4b65-919a-f3faabdf2515){: .shadow .rounded-10 .w-75}

## 용어 정리

CP : Continual pre-trained model
SFT : Supervised Fine-tuning - 특정 도메인으로 구축된 양 질의 데이터 Instruct, Input, Reponse 쌍을 통해서 fine-tune 하는 과정이며 instruct-tuning이라고도 함
RLHF : Reinforcement Learning with Human Feedback

## 0. Abstract

오픈소스 LLM 모델이 매우 급변하고있는데, `data` 부족에 대한 이슈로 오픈소스 기반 LLM이 주로 `영어`에 집중되어있다. 이 문제를 해결하기 위해, `chat vector` 컨셉을 소개한다. chat vector를 통해 인간 수준의 가치를 가진 (pre-trained 된) LLM모델을 매우 간단한 방식으로 활용할 수 있다. chat vector를 단순히 CP모델에 추가하면, **<mark>추가적인 훈련 없이</mark>** 새로운 언어에 대한 대화 능력을 갖추게 할 수 있다.

1. **Instruction Following** : chat vector를 통해 LLM이 특정 명령을 수행할 수 있도록 할 수 있다.
2. **Toxicity Mitigation** : chat vector를 통해 LLM이 인간의 가치에 맞추어 조정할 수 있다.
3. **Multi-turn dialogue** : chat vector를 통해 LLM이 다중 대화를 수행할 수 있다.

{: .prompt-tip}

> chat vector를 통해 간단하고, 효과적이고, 대화에서 다방면으로 활용가능한 LLM(pre-trained)을 만들 수 있다.

## 1. Introduction

최근 LLM이 많은 주목을 받고 매우 빠르고 급격하게 성장해왔지만, 오히려 이러한 빠른 성장으로 인해 여러 한계에 마주하고 있다. 특히 `data` 부족으로 인해 영어 이외 다른 언어에 대한 제약이 생겼다.

![image](https://github.com/uujeong/uujeong.github.io/assets/86465999/6b41b88f-41c4-4cf5-9a48-f9358b611d67){: .shadow .rounded-10 .w-75}

```markdown
Figure 1: An illustration to demonstrate the difference
between the traditional approach and our method. The
blue arrows on the top right side depict the conventional
method of constructing a non-English LM. First, an
open-source PLM (e.g. LLaMA2) undergoes continual
pre-training (CP) on the target language, followed by
SFT and RLHF alignment procedures. In contrast, the
gray arrow on the left illustrates how we obtain the chat
vector through simple parameter subtraction. This chat
vector can be added to the CP model to produce the
chat model in the target language, as depicted by the
dual-color arrow.
```

> 파란색 화살표가 비영어권 언어 LLM의 전통적인 설계 방식을 의미한다.
> PLM (예를 들어 LLaMA2)가 지속적인 pre-training (CP)을 타깃 언어(한국어)로 SFT, RLHF 등을 통해 수행한다. -> 매우 비효율적이다.
> 대조적으로, 회색 화살표는 chat vector를 통해 CP모델에 간단하게 추가적인 파라미터를 추가하는 방식을 보여준다. -> 매우 효율적이다.

이 chat vector는 듀얼 컬러 방식을 통해 다른언어(한국어) CP 모델에 추가될 수 있다.

To achieve this, we propose an approach to restructure the conventional training paradigm for non-English LLMs from CP → SFT → RLHF to CP + chat vector

chat vector는 LLaMA-2의 사전 학습된 가중치에서 채팅이 강화된 대응 모델인 채팅을 강화한 LLaMA-2-chat의 가중치를 빼서 도출한다.

{: .prompt-tip}

> 지속적인 pre-training 후 chat vector를 통합하는 전략이 LLaMa-2 채팅에 대한 직접적인 pre-training에 비해 우수한 결과를 가져왔다는 것을 보여준다.
> 더욱이, chat vector 를 결합하기 전에 fine-tuning 기법을 적용하면, 데이터 세트의 규모나 사전학습된 LLM의 언어(한국어, 중국어 등)에 관계없이 성능을 최적화한다.

## 2. Related Work

### 2.1. Human Preference Training

> 인간 피드백을 통한 강화학습(RLHF)의 대안으로 꼽히는 방식이다.

1. DPO(Direct Preference Optimization, 데이터 성능 최적화) : 인간 선호도에 맞는 결과 도출한다는 점에서는 동일하지만, RLHF와 달리 `보상 모델`이 필요하지 않다는 장점이 있다.

   - DPO 방식에서 중요한 것은 SFT(Supervised Fine-tuning)를 잘해야한다는 것이다.
   - data를 대량으로 쏟아붓는 대신, quality가 높은 data를 사용, data 분포를 잘 확인해야한다.
     ![image](https://github.com/uujeong/uujeong.github.io/assets/86465999/3911e744-b00a-42fc-8f83-e7434346cc7e){: .shadow .rounded-10 .w-75}  
     DPO는 많은 작업에서 CPO와 유사한 성능을 보여주지만, 데이터 효율성 측면에서는 중간 수준입니다. KTO보다는 못하지만 IPO보다는 나은 데이터 효율성을 가지고 있습니다.

2. IPO (Iterative Performance Optimization, 반복 성능 최적화):
   이 방법은 데이터 효율성이 가장 낮으며, 특히 추론, QA, 수학 작업에서 다른 방법들에 비해 일반적으로 뒤처지는 것으로 보입니다.

3. KTO (Knowledge Transfer Optimization, 지식 전달 최적화):
   이 방법은 대부분의 벤치마크에서 뛰어난 성능을 보여주므로, 추론, QA(질문 응답), 수학 문제 등 다양한 작업에서 지식을 전달하거나 활용하는 데 매우 효과적임을 시사합니다. 특히 작은 데이터 세트에서도 효율적으로 성능을 발휘하는 것으로 나타났습니다.

4. CPO (Contextual Performance Optimization, 문맥 성능 최적화):
   CPO는 대체로 다른 방법들과 비교해 비슷한 성능을 보이지만, 특정 벤치마크(예: PIQA 등)에서는 눈에 띄게 부진합니다. 그러나 데이터 효율성에서는 두 번째로 좋은 성적을 보입니다.

{: .prompt-tip}

> 데이터를 효과적으로 사용하는 면에서 KTO가 최고, -> CPO -> DPO -> IPO 순입니다. 이는 KTO가 데이터가 제한된 시나리오에서도 좋은 결과를 얻을 수 있다는 중요한 장점을 시사합니다.

### 2.2 Task Vector

- Task Vector는 fine-tuning된 모델의 가중치(W)에서 pre-trained된 모델의 가중치(W)를 덜어내면 얻을 수 있다.

- How chat vector works.
  ![image](https://github.com/uujeong/uujeong.github.io/assets/86465999/a5151794-ed52-4005-a915-e801e984be4f){: .shadow .rounded-10 .w-75}

## 3. Methodology

### 3.1 Continual Pre-training (CP)

기존의 사전 훈련된 모델을 특정 도메인이나 작업에 더 적합하도록 추가적으로 훈련시키는 과정을 말합니다. 이 방식은 모델이 이미 배운 지식을 유지하면서 새로운 정보를 통합할 수 있도록 해서, 전체적인 성능을 향상시키고, 특정 작업에 대한 모델의 적응력을 높입니다.

$$
L(\theta_{CP}) = \mathbb{E}_{x \sim D_{CP}} \left[ -\sum_{i} \log P(x_i | x_0, \dots, x_{i-1}; \theta_{CP}) \right]
$$

이 수식을 해석해보자면, 지속적인 사전 훈련 과정에서 모델이 데이터셋에 대해 얼마나 잘 적응하고 있는지를 측정하는 손실 함수를 나타낸다. 모델이 이 손실 함수를 최소화하도록 훈련되면, 주어진 데이터셋의 패턴을 더 잘 이해하고 예측할 수 있게 된다. 이 과정을 통해 모델은 특정 도메인이나 작업에 더 특화되도록 발전할 수 있다.

### 3.2 Chat Vector

LLaMA2-chat같은 모델은 이 모델은 베이스 모델을 기반으로 SFT(Supervised Fine-tuning)과 RLHF(인간 피드백)을 기반으로 LLaMA2를 수정했다.
