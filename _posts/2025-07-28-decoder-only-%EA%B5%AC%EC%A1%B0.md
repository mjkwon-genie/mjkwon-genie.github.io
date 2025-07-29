---
layout: post
title: "Decoder only 모델의 아키텍처 총정리"
---

1. 모델 입력 및 전처리

● **Tokenization**
입력 문장(raw text)을 토크나이저로 n개의 토큰 id(정수 시퀀스)로 변환

(예: "안녕 나는 제나야" → [101, 455, 999, 203])

● **Token Embedding**
Embedding Table: (vocab_size, m) 크기

각 토큰 id를 임베딩 테이블에서 lookup → (n, m) 임베딩 행렬 생성

(이때 lookup은 행렬 곱이 아니라 단순 인덱스 참조, 빠르고 효율적)

● **Positional Encoding**
각 토큰의 위치 정보를 담은 (n, m) 행렬 생성

Sinusoidal: 위치-차원별로 sin, cos 수식 사용 (Transformer 논문)

Learned: (max_position, m) 크기의 테이블에서 n개만 슬라이스

두 행렬(token + positional)을 더해 최종 (n, m) input matrix 생성

2. Decoder Block (N회 반복)

A. **Pre-Norm (LayerNorm/RMSNorm)**
각 블록의 attention/FFN 앞에 RMSNorm/LayerNorm 적용

RMSNorm은 평균(mean)을 생략해 더 빠르고 간결함 (Llama 계열에서 채택)

B. **Grouped Query Masked Self-Attention**
Query head(h개)는 많고, Key/Value head(h_g개)는 적게 (GQA, ex. 32:8, 8:1~4:1)

입력 (n, m)을 head별로 쪼갬:

Q: h개의 (n, m/h),

K/V: h_g개의 (n, m/h_g)

Masked Attention:

각 토큰은 현재 위치까지(미래는 minus infinity)만 보게 마스킹

softmax는 row별 (각 query별로 확률 합 1)

Flash Attention:

전체 (n, n) matrix를 블록(block) 단위로 연산해 메모리와 속도 효율 극대화

긴 context(수천~수만 토큰)에서도 연산량이 폭증하지 않음

Dropout:

Attention output 등 여러 위치에 적용 (overfitting 방지, 일반화)

C. **LoRA Adapter (선택적)**
Wq, Wv, (Wk, FFN Linear 등) 주요 선형 weight에 부착 가능

각 head별로 (m, r) A matrix, (r, m/h) B matrix 추가

학습시 기존 W는 고정, 오직 저랭크 A/B만 학습

D. **Residual Connection**
Attention/FFN의 입력과 출력을 더해 정보 보존 및 gradient 흐름 안정화

output=f(x)+x

Vanishing/exploding gradient 문제 모두 완화, 깊은 네트워크의 학습 안정화

E. **Feed Forward Network (FFN)**
구조: Linear(m, d_ff) → Activation(GeLU) → Dropout → Linear(d_ff, m) → Dropout

GeLU는 부드러운 비선형성, 역전파 시 자동 미분(gradient) 처리

FFN의 첫/둘째 Linear에도 LoRA 적용 가능

F. **(Residual, Norm, Dropout 반복)**
각 모듈 뒤에 residual + norm + dropout으로 일반화, 학습 안정성, 정보 흐름 모두 확보

3. 최종 출력/생성 단계

● **N개의 Decoder Block 통과**
(n, m) output matrix 획득

● **Final Linear Layer (Fully Connected)**
(n, m) → (n, vocab_size)

각 토큰 위치별로 다음에 올 토큰 후보 전체에 대한 점수(logit) 산출

Embedding table과 weight tying(동일 파라미터 사용)도 일반적

● **Softmax & 확률 해석**
각 row별로 softmax 적용 → 다음 토큰 확률 분포

텍스트 생성 시에는 마지막 row에서 argmax/sampling으로 다음 토큰 선택

● **Autoregressive Generation**
한 번에 한 토큰씩(1-step), n회 반복

각 step마다 input에 새로 생성한 토큰을 추가하여 다시 예측 (chain 방식)

4. 학습 과정/손실 함수

● **Loss Function**
Cross Entropy Loss 사용 (Negative Log Likelihood의 특수 케이스)

(n, vocab_size) 모델 output과, 정답 토큰 id(n,) 비교

내부적으로 softmax+log+NLL이 모두 자동 처리됨

각 위치에서 정답 토큰의 확률이 1이 되도록 모델 파라미터를 업데이트

● **Backpropagation (역전파)**
전체 계산 그래프(선형, 활성화, normalization 등)의 모든 연산이 연산 노드

chain rule(연쇄 법칙)에 따라 각 파라미터에 대해 gradient(미분값) 자동 계산

activation function도 "노드"이므로, gradient가 반드시 필요

● **Optimizer (AdamW)**
base learning rate는 사용자가 직접 지정 (예: 2e-5, 3e-4 등)

각 파라미터별로 adaptive learning rate, momentum, weight decay를 동적으로 계산

최종적으로 gradient와 base lr, 이전 상태, 가중치 크기 등을 종합해
파라미터를 효율적·안정적으로 업데이트

5. Dropout, Residual, 기타 핵심 기법

● **Dropout**
attention, FFN 등 여러 모듈에 적용

위치, 비율(dropout rate)은 하이퍼파라미터로 사용자가 지정

학습 시마다 무작위로 일부 뉴런을 끄면서 overfitting 억제, robust한 모델 학습

● **Residual Connection**
각 block의 입력/출력을 더해

정보 보존/보강, gradient vanishing/exploding 동시 완화

deep network도 안정적으로 학습

이전 정보와 새로운 변화량을 효율적으로 조합

● **LoRA**
기존 weight를 그대로 두고,
저랭크 두 행렬(A/B)의 곱만 학습해
파라미터, 메모리, 연산량을 크게 줄이면서 빠르고 효과적인 파인튜닝 가능

보통 Wq, Wv에 적용, 필요시 K/FFN 등 확장 가능

● **Grouped Query Attention & Flash Attention**
GQA: Query head 수를 크게, Key/Value head 수를 적게 하여 long context 효율화

Flash Attention: Attention 연산을 block-wise로 메모리/속도 혁신적으로 개선

두 방식 모두 최신 LLM에서 함께 적용됨

6. 전체적인 흐름(구조 요약)

입력: Raw 문장 → Tokenizer → (n,) 토큰 id

Embedding: (n, m) 임베딩, positional encoding 더함

N개의 Decoder Block:

Norm → Grouped Query Masked Flash Attention(+Dropout, +LoRA) → Residual

Norm → FFN(+Activation, +Dropout, +LoRA) → Residual

Dropout/Residual/Norm/LoRA 모두 효율적 조합

Output: (n, m) → Linear → (n, vocab_size)

Softmax: 각 위치의 다음 토큰 확률

Autoregressive로 한 번에 한 토큰씩 반복 생성

학습: CrossEntropyLoss(softmax+NLL), AdamW optimizer로 파라미터 업데이트

7. 실제 구현/코드 흐름 예시 (파이토치 기반)

```python
# Forward pass
input_ids = tokenizer.encode("안녕 나는 제나야", return_tensors='pt')
emb = model.embedding(input_ids)  # (n, m)
pos_emb = model.positional_encoding(torch.arange(n))  # (n, m)
x = emb + pos_emb

for block in model.decoder_blocks:
    x = block(x)  # 내부에 Norm, GQA+Flash, Residual, FFN, Dropout, LoRA 포함

logits = model.final_linear(x)  # (n, vocab_size)
loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), labels.view(-1))
# Optimizer step (AdamW 등)
```

8. 최종 요약

LLM Decoder-only 아키텍처는
토큰 임베딩+포지셔널 인코딩 → N개 decoder block(Flash GQA, FFN, Residual, Dropout, LoRA 등)
→ linear+softmax → CrossEntropyLoss+AdamW 학습,
→ Autoregressive 토큰 생성의 전형적인 패턴

각 모듈의 핵심 역할과 조합이 딥러닝 LLM의 성능, 효율, 확장성의 기반이 됨
