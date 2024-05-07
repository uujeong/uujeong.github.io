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

역할 별 화법 변환을 구현하기 위해 관련 논문을 공부하고자 한다. 이 논문의 원문은 다음과 같다.
<https://arxiv.org/pdf/2310.10158>

![image](https://github.com/uujeong/toDoList_react/assets/86465999/709a5996-1b00-418f-be00-4d459f6a48bd){: .shadow .rounded-10 .w-75}

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

여기서 소개하는 `Character-LLM`이라는 새로운 개념을 소개하는데, 이는 역사적, 허구적 인물들을 모방할 수 있는 훈련 가능한 Agent를 만드는 것을 목표로 한다. 이 Agent들은 개인적인 경험과 특성, 감성을 배우도록 설계되었다. 캐릭터의 경험을 재구성하고 LLM을 사용하여 수집된 개인적 경험을 바탕으로 장면을 추출하여 Agent가 `경험`을 기반으로 캐릭터와 감정을 형성할 수 있도록 한다.  
 특히, ChatGPT와 GPT-4와 같은 모델들이 어떻게 사람들의 일상 활동을 모방하는 데 사용될 수 있는지에 대해 설명한다. 이 모델들은 자세히 구성된 prompt를 사용하는데, 이를 통해 사회에서 특정 역할을 하는 평범한 사람들처럼 행동할 수 있도록 한다는 의미이다.

사람들의 경험, 특성, 감성을 학습하여 Beethoven, Queen Cleopatra, Julius Caesar 와 같은 유명 인물들을 Role play 할 수 있는 **<mark>훈련 가능한 Agent</mark>**를 만드는 것을 목표로 한다.

{: .prompt-info }

> 이런식으로 LLM들은 단순히 Prompt를 받아 처리하는게 아니라 실제 인물의 경험을 '기억'하고 '반영'하여 '행동'하는 방식으로 발전할 수 있는 가능성을 제시하고있다.

{: .prompt-info }

> 목표 3줄 요약
>
> 1. building trainable agents as a character simulacra -> Character-LLM을 통해!
> 2. 경험에 대한 재구성, 업로드, protective 한 경험을 포함하는 LLM을 활용한 simulacra 훈련
> 3. 훈련된 Agent를 test하고 더 나은 character simulacra를 구축하는데 도움이 되는 결과를 제공

## 2. Related Work

### 2.1 Simulacra of Human Behavior with LLMs
