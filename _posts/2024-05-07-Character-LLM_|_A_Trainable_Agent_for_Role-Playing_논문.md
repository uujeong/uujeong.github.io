---
layout: post
title: Character-LLM | A Trainable Agent for Role-Playing 논문 리뷰
description:
date: 2024-05-07 14:00:00 +09:00
categories: [논문 리뷰]
tags: [LLM, 화법변환]
# image: # preview img (해상도가 1200 x 630인 이미지)
#   path: /path/to/image
#   alt: image alternative text
# pin: true # 게시글 고정
---

동화 혹은 소설 속 등장인물의 역할별 화법 변환을 구현하기 위해 관련 논문을 공부하고자 한다. 이 논문의 원문은 다음과 같다.
<https://arxiv.org/pdf/2310.10158>

![image](https://github.com/uujeong/toDoList_react/assets/86465999/709a5996-1b00-418f-be00-4d459f6a48bd){: .shadow .rounded-10 .w-75}

## 용어 정리

_이해를 돕기 위해 다음과 같이 논문 내 용어를 정의하겠다._

- [ ] simulacra : 모방체
- [x] LLM : Large Language Model
- [ ] Character-LLM : 특정 인물을 시뮬레이션하는 훈련 가능한 Agent

<!-- tip, info, warning, danger -->

{: .nolineno }

## 0. Abstract

인간 행동을 모방할 수 있는 LLMs의 능력을 활용하여 단순한 행동 이상의 형태로 특정 인물을 시뮬레이션하는 새로운 접근 방식을 제안한다.

{: .prompt-tip}

> 목표 : aim to train an agent with the profile, experience, and emotional states of a specific person instead of using limited prompts to instruct ChatGPT API.

## 1. Introduction

여기서 소개하는 `Character-LLM`이라는 새로운 개념은, 이는 역사적, 허구적 인물들을 모방할 수 있는 훈련 가능한 Agent를 만드는 것을 목표로 한다. 이 Agent들은 개인적인 경험과 특성, 감성을 배우도록 설계되었다. 캐릭터의 경험을 재구성하고 LLM을 사용하여 수집된 개인적 경험을 바탕으로 장면을 추출하여 Agent가 `경험`을 기반으로 캐릭터와 감정을 형성할 수 있도록 한다.  
 특히, ChatGPT와 GPT-4와 같은 모델들이 어떻게 사람들의 일상 활동을 모방하는 데 사용될 수 있는지에 대해 설명한다. 이 모델들은 자세히 구성된 prompt를 사용하는데, 이를 통해 사회에서 특정 역할을 하는 평범한 사람들처럼 행동할 수 있도록 한다는 의미이다.

사람들의 경험, 특성, 감성을 학습하여 Beethoven, Queen Cleopatra, Julius Caesar 와 같은 유명 인물들을 Role play 할 수 있는 **<mark>훈련 가능한 Agent</mark>**를 만드는 것을 목표로 한다.

{: .prompt-info }

> 이런식으로 LLM들은 단순히 Prompt를 받아 처리하는게 아니라 실제 인물의 경험을 '기억'하고 '반영'하여 '행동'하는 방식으로 발전할 수 있는 가능성을 제시하고있다.

{: .prompt-tip }

> 목표 3줄 요약
>
> 1. building trainable agents as a character simulacra -> Character-LLM을 통해!
> 2. 경험에 대한 재구성, 업로드, protective 한 경험을 포함하는 LLM을 활용한 simulacra 훈련
> 3. 훈련된 Agent를 test하고 더 나은 character simulacra를 구축하는데 도움이 되는 결과를 제공

## 2. Related Work

### 2.1 Simulacra of Human Behavior with LLMs

이 연구 Section에서는 대형 언어 모델(LLMs)를 사용하여 인간 행동의 simulacra를 구현하는 이전 연구들을 소개한다. 초기 연구들은 인간처럼 행동하는 Agent 개념을 도입했으며, 비디오 게임 내 NPC(Non-player Character)에 적용되어 게임에서의 인지 기능 지원을 목표로 한다. [선행연구 : (Bates, 1994; Tomas and Johnston, 1981), (Laird and VanLent, 2001 ; Riedl, 2012)]

특히 Park et al. (2023)은 대형 언어 모델을 활용하여 인간의 기억을 합성해 실제와 같은 인간 행동을 모방하는 생성 Agent를 처음으로 소개했다. 이러한 대형 언어 모델은 인간 사회의 방대한 데이터로 훈련되어 인간 행동에 대한 광범위한 지식을 보유하고있다. 여러 연구에서는 이러한 LLM을 사용하여 각 개인의 특성과 해당 행동을 짧은 자연어 설명으로 생성하고, 생성된 정보를 사용하여 사회적 행동을 시뮬레이선하는데 활용한다.

인간 simulation의 예시로 음성 생성이나 딥페이크 생성같은 기술이 연구되고 있다는 표현이 가장 직관적으로 와닿았다.

### 2.2 Specialization of LLMs

LLMs을 인간 행동을 시뮬레이션하는데 사용 하는 것을 소개한다. LLMs의 특화는 LLM개발 중 하나로, LLMs를 character simulacra에 특화시키려고 하고 있기 때문에 LLMs가 어떻게 특화되는지 연구하는게 중요하다고 설명한다.

여기서 소개하는 연구방법론은 모두 Fine-tuning, RLHF, self-instruction tuning을 사용하여 LLMs를 특화시키는 것을 목표로 한다.

## 3. Approach

![image](https://github.com/uujeong/uujeong.github.io/assets/86465999/0a1d9b38-bbe3-4a79-9749-b1d938cf9444){: .shadow .rounded-10 .w-95}

스타일과 어조를 모방하는데 있던 지도 미세 조정(SFT)이나 자연어와 유사한 손으로 만든 규칙과 설명을 제공하는 방식(이전방식)과는 다르게, 사람들이 과거 경험과 사건에 기반하여 다양한 성격을 키우는 방식에서 영감을 받았다.

{: .prompt-info}

> Experience Upload

LLM이 미리 정의된 캐릭터의 정신적 활동과 신체적 행동을 모방하고, 재구성된 경험으로부터 학습하여 그들처럼 행동할 수 있는 능력을 습득하는 혁신적인 학습 프레임워크이다.

그림을 보면 논문의 접근 방식이 더 직관적으로 와닿는데, 특정 character의 `profile collection`에서 과거 경험을 설명하는 특정 화상 장면을 유발한다. 이는 환각(hallucination)을 효과적으로 완화하고 데이터 수렴의 부족을 해결한다. (-> 당연한 이야기)
이 재구성된 장면들로부터 학습함으로써, 우리는 LLMs를 높은 신뢰도를 갖는 여러 캐릭터 에이전트로 특화시킨다.

### 3.1 Building Experience Dataset

이 부분에서는 특정 인물의 경험을 대형 언어 모델(LLM)을 사용하여 재구성하고자 하는 목표에 대해 설명한다. 인간의 경험은 매우 복잡하며, 중요한 이정표와 하찮은 사건들이 뒤섞여 있고, 긴 시간에 걸쳐 있다. 제한된 context와 언어 모델의 본질적인 오류로 인해 일관되고 통합된 경험을 다시 만드는 작업은 어려운 일이다. 따라서, 사실 기반의 경험 재구성 pipeline을 제안한다고 한다. (경험을 재창조하는 것을 포함)

1. 프로필: 캐릭터의 속성에 대한 간결한 설명(전반 정보, 중요한 사건 정보, 초기 어린 시절 등..)
2. 장면: 캐릭터의 상호작용이 펼쳐지는 특정 장소. (시간적, 공간적 context 포함)
3. 상호작용: 캐릭터의 인지 과정, 발화 또는 행동. (represented in plain text)

#### 3.1.1 Profile Collection

![image](https://github.com/uujeong/uujeong.github.io/assets/86465999/9cf7b26a-02cb-4e09-800c-f0d0a2f11dc3){: .shadow .rounded-10 .w-95}

{: .prompt-info}

> comprehensive character profile을 collect하는 과정

#### 3.1.2 Scene Extraction

특정 생애 시기에 character의 experience description을 간결하게 설명하는 프로필 + LLM에게 이 경험 설명을 바탕으로 발생했을 법한 여러 다른 장면을 나열하도록 요청한다. (대략적인 위치와 간단한 배경 일러스트레이션을 포함)
{: .prompt-info}

> -> 물론 간단하게 출력하는건 LLM의 비용 절감을 위한 것!

#### 3.1.3 Experience Completion

추출된 장면은 개인 간의 상세한 상호작용 경험으로 확장된다. 특정 장면 설명과 해당 프로필 조각이 주어진 상태에서, LLM은 캐릭터들 간의 상호작용과 표적 개인의 생각을 포함시키며 장면을 자세히 설명하도록 유도된다. -> 이 내용은 시나리오 형식(머리말은 공간으로 시작)으로 작성되며, 그 다음 각 블록이 특정 캐릭터의 발언이나 표적 개인의 성찰을 나타내는 일련의 블록으로 표현된다. 여기서 중요한 점은, 이 장면이 표적 개인의 관점을 바탕으로 완성된다는 것이다. 따라서, 모든 캐릭터의 반영이 아닌 `표적 개인의 성찰만이 포함`된다.

이 과정을 통해 캐릭터의 경험을 보다 실감나고 자세히 재현할 수 있으며, 특히 개인의 관점에서 그들의 생각과 상호작용을 중심으로 장면을 구성함으로써, 보다 깊이 있는 캐릭터 이해와 시뮬레이션을 가능하게 합니다.

### 3.2 Protective Experience

캐릭터가 본연의 정체성과 모순되는 지식에 대해 질문받을 때 무지와 혼란을 표현하도록 하는 protective experience를 구축하는 방법을 다룬다. 이를 위해 모델을 훈련시켜 캐릭터의 시대나 정체성에 부합하지 않는 지식에 대해서는 대답하지 않고 지식 부족을 표현하게 하는데, 이러한 보호 장면은 캐릭터의 정체성을 보호하면서 동시에 캐릭터와 모순되는 지식을 잊도록 돕는다.
->

### 3.3 Experience Upload

이 부분에서는 특정 캐릭터의 `경험 데이터만`을 사용하여 각 역할에 맞게 LLM 기반 모델을 fine-tuning하는 과정을 설명한다. 예를 들어, LLaMA 모델을 캐릭터별로 특화하여 각각의 캐릭터에 대한 에이전트 모델을 별도로 fine-tuning한다. -> 이렇게 각 캐릭터에 맞춤형 데이터를 사용함으로써, 캐릭터 간 지식 충돌을 제거하고 역할 연기의 정확성을 향상시키는 것이 목표이다.

### 3.4 Compared to Existing Practice

개인 프로필로부터 장면과 상호작용을 유도하는 새로운 방법을 소개. -> 이 방법은 LLM 내의 편향된 분포와 환각을 피하면서 사실에 기반한 시뮬레이션을 가능하게 하기 때문에 신뢰성이 높다. 상호작용이 각 장면에 내재되어 있어 모델의 상호 작용 호출을 줄이면서 더 자연스럽고 믿을 수 있는 상호작용 simulacra를 제공한다.

## 4 Experiments

여러 simulacra의 성능을 평가하기 위해, simulacra를 인터뷰하고 응답의 질을 평가한다. 훈련된 simulacra가 알파카같은 모델보다 우수한 성능을 보이며, 개성있는 답변을 보여준다.

### 4.1 Data Setup

`Data` : 역사적 인물, 상상 속 캐릭터, 유명인 등 다양한 배경의 캐릭터를 포함.  
선택된 캐릭터에 대해 3절에서 언급한 프로토콜을 따라 경험 데이터를 재구성한다.

### 4.2 Training Setup

LLaMA 7B 모델을 기반으로 각 시뮬라크럼을 해당 경험 예시에 따라 fine-tuning. 학습 예제의 시작 부분에 meta-prompt를 삽입, 각 예제마다 배경, 시간, 장소, 관련 인물을 설명하는 간결한 설명을 제공한다.

### 4.3 Evaluation as Interviews

평가는 인터뷰 방식으로 진행했다.

### 4.4 LLM as Judges

![image](https://github.com/uujeong/uujeong.github.io/assets/86465999/b6c18870-8e3a-46b1-a375-54f097dd7b22){: .shadow .rounded-10 .w-95}
![image](https://github.com/uujeong/uujeong.github.io/assets/86465999/57374a98-2fe7-436e-aa3b-006933ed46df){: .shadow .rounded-10 .w-95}

## 5 Conclusion and Future

{: .prompt-info}

> 결론 : 이 논문에서는 특정 인물을 모방하는데 있어 기존의 프롬프트 기반 에이전트보다 효과적인 훈련 가능한 에이전트, 즉 Character-LLM을 구축하는 방법을 제시했다.

## 6. Limitations

1. Evaluation Protocols : Agent를 평가하기 어렵다. (캐릭터에 대한 깊은 이해 필요 + 인간 평가를 진행하기 어려움)
2. Limited data : 캐릭터 프로필을 기반으로 한 장면을 서술, 하지만 실제 인물의 전체 생애나 실제의 한 측면을 충분히 대표X
3. Base model : fine tuning의 결과가 기본 모델에 따라 크게 달라진다.
4. Potential Harm : 캐릭터 simulacra에서 생성된 텍스트는 캐릭터가 결함이 있다면 안된다. 마키아벨리와 같이 생동감 있는 시뮬라크럼은 사람들을 해로운 활동으로 유도할 수도 있다. 따라서 생동감 있는 시뮬라크라를 구축하는 것과 부정적인 생각이 없는 캐릭터를 구축하는 것 사이의 균형이 필요할 수 있다.
