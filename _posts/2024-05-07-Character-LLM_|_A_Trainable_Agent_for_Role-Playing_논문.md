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

역할별 화법 변환을 구현하기 위해 관련 논문을 공부하고자 한다. 이 논문의 원문은 다음과 같다.
<https://arxiv.org/pdf/2310.10158>

![image](https://github.com/uujeong/toDoList_react/assets/86465999/709a5996-1b00-418f-be00-4d459f6a48bd){: .shadow .rounded-10 .w-75}

## 용어 정리

_이해를 돕기 위해 다음과 같이 논문 내 용어를 정의하겠다._

- [ ] simulacra : 모방체

_Character-LLM: A Trainable Agent for Role-Playing_

tip, info, warning, danger

{: .nolineno }

- [ ] Job
  - [x] Step 1
  - [x] Step 2..
  - [ ] Step 3

## 0. Abstract

이 논문의 초록(abstract)은 인간 행동을 모방할 수 있는 LLMs의 능력을 활용하여 단순한 행동 이상의 형태로 특정 인물을 시뮬레이션하는 새로운 접근 방식을 제안한다.

목표 : aim to train an agent with the profile, experience, and emotional states of a specific person instead of using limited prompts to instruct ChatGPT API.

## 1. Introduction

여기서 소개하는 `Character-LLM`이라는 새로운 개념은, 이는 역사적, 허구적 인물들을 모방할 수 있는 훈련 가능한 Agent를 만드는 것을 목표로 한다. 이 Agent들은 개인적인 경험과 특성, 감성을 배우도록 설계되었다. 캐릭터의 경험을 재구성하고 LLM을 사용하여 수집된 개인적 경험을 바탕으로 장면을 추출하여 Agent가 `경험`을 기반으로 캐릭터와 감정을 형성할 수 있도록 한다.  
 특히, ChatGPT와 GPT-4와 같은 모델들이 어떻게 사람들의 일상 활동을 모방하는 데 사용될 수 있는지에 대해 설명한다. 이 모델들은 자세히 구성된 prompt를 사용하는데, 이를 통해 사회에서 특정 역할을 하는 평범한 사람들처럼 행동할 수 있도록 한다는 의미이다.

사람들의 경험, 특성, 감성을 학습하여 Beethoven, Queen Cleopatra, Julius Caesar 와 같은 유명 인물들을 Role play 할 수 있는 **<mark>훈련 가능한 Agent</mark>**를 만드는 것을 목표로 한다.

![image](https://github.com/uujeong/uujeong.github.io/assets/86465999/0a1d9b38-bbe3-4a79-9749-b1d938cf9444){: .shadow .rounded-10 .w-75}

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

이 연구 Section에서는 LLMs을 인간 행동을 시뮬레이션하는데 사용 하는 것을 소개한다. LLMs의 특화는 LLM개발 중 하나로, LLMs를 character simulacra에 특화시키려고 하고 있기 때문에 LLMs가 어떻게 특화되는지 연구하는게 중요하다고 설명한다.

여기서 소개하는 연구방법론은 모두 Fine-tuning, RLHF, self-instruction tuning을 사용하여 LLMs를 특화시키는 것을 목표로 한다.

## 3. Approach
