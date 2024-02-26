---
layout: post
title: Confusion Matrix
description:
date: 2024-02-25 06:00:00 +09:00
categories: [머신러닝, 평가지표]
tags: []
---

분류와 회귀분제

1. 분류(Classification) : 예측해야할 대상의 개수가 정해져있는 문제이다.

- 예를 들어) 이미지에서 개와 고양이 분류, 신용카드 거래가 사기거래인지 정상거래인지 분류

2. 회귀(Regression) : 예측해야할 대상이 연속적인 숫자인 문제

- 예를 들어) 일기 예보에서 내일의 기온 예측, 주어진 데이터에서 집값 예측

3. 평가 지표(Evaluation Metric)

- 분류, 회귀 머신러닝 문제의 성능을 평가할 지표

| Confusion Matrix |                     | Prediction (예측)  |                     |
| ---------------- | ------------------- | ------------------ | ------------------- |
| Negative         | Positive            |
| 실제             | Negative            | TN (True Negative) | FP (False Positive) |
| Positive         | FN (False Negative) | TP (True Positive) |

#### **1\. Accuracy :  (TP + TN) / (TP + TN + FP + FN)**

**(분자)**

| Confusion Matrix |          | Prediction (예측) |     |
| ---------------- | -------- | ----------------- | --- |
| Negative         | Positive |
| 실제             | Negative | ✅ TN             | FP  |
| Positive         | FN       | ✅ TP             |

-----------------------------------------------

**(분모)**

| Confusion Matrix |          | Prediction (예측) |       |
| ---------------- | -------- | ----------------- | ----- |
| Negative         | Positive |
| 실제             | Negative | ✅ TN             | ✅ FP |
| Positive         | ✅ FN    | ✅ TP             |

#### 정확도, 전체 데이터 중 모델이 바르게 분류한 척도를 의미한다.

cf) 불균형한 지표에는 부적합한 평가지표이다.

가령, 100명 중 1명의 암 환자를 예측하여 진단한다고 할 때, 100명을 모두 암환자가 아니라고 진단할 경우, 99%의 정확도를 보인다.

99%라는 숫자는 높아보이지만 결국 1명의 암환자를 정확하게 진단하지 못한다.  -> 적합한 평가지표가 아니다.

---

#### **2\. Precision : TP / (TP + FP)**

**(분자)**

| Confusion Matrix |          | Prediction (예측) |     |
| ---------------- | -------- | ----------------- | --- |
| Negative         | Positive |
| 실제             | Negative | TN                | FP  |
| Positive         | FN       | ✅ TP             |

\-------------------------------------------------

| Confusion Matrix |          | Prediction (예측) |       |
| ---------------- | -------- | ----------------- | ----- |
| Negative         | Positive |
| 실제             | Negative | TN                | ✅ FP |
| Positive         | FN       | ✅ TP             |

\= 정밀도

모델이 positive라고 예측한 값중 실제값이 positive 인 비율.

➡️ negative인 비율이 더 중요한 경우(negative인 데이터를 poisitive라고 판단하면 안될 경우)에 쓰인다.

대표적으로 스팸메일 (일반 메일을 스팸메일로 분류하게되면 중요한 메일을 놓칠 수 있다.)

따라서, 스팸메일을 걸러내지 못하더라도 일반 메일을 스팸메일로 잘못분류하지 않도록 하는 것이 중요하다.

---

#### **3\. Recall : TP / (TP + FN)**

(분자)

| Confusion Matrix |          | Prediction (예측) |     |
| ---------------- | -------- | ----------------- | --- |
| Negative         | Positive |
| 실제             | Negative | TN                | FP  |
| Positive         |  FN      | ✅ TP             |

\----------------------------------------------

| Confusion Matrix |          | Prediction (예측) |     |
| ---------------- | -------- | ----------------- | --- |
| Negative         | Positive |
| 실제             | Negative | TN                | FP  |
| Positive         | ✅ FN    | ✅ TP             |

\= 재현율

실제값이 Positive인 것 중, 모델이 Positive라고 분류한 비율

Positive인 지표가 더 중요할 경우(실제 Positive인 데이터를 Negative라고 판단하면 안될 때 사용한다,)

➡️ 종양을 진단할 때, 악성종양을 음성종양으로 분류하면 환자의 생명이 위급해질 수 있다.

ROC

1\. True Positive Ratio = TP / (TP + FN)

: 1인 case에 대해 1로 잘 예측한 비율

2\. False Positive Ratio = FP / (FP + TN)

: 0인 case에 대해 1로 잘못 예측한 비율

AUC : ROC의 면적을 나타낸 것으로 0과 1사이 값을 가질 수 있고, 1에 가까워질수록 모델이 잘 예측, 0에 가까워질수록 모델이 잘못예측할 경우를 말한다.
