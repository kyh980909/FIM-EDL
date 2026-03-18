# Neurocomputing 투고를 위한 추가 실험 정리

## 목적

이 문서는 현재 확보된 실험 결과를 바탕으로, 논문에 어떤
추가 실험을 넣어야 하는지 정리한 메모이다.

기준으로 사용한 결과 파일은 다음과 같다.

- `FIM-EDL/results/paper_tables/iedl_table2/table2_iedl_style.md`
- `FIM-EDL/results/paper_tables/iedl_table3/table3_iedl_style.md`
- `FIM-EDL/results/paper_tables/iedl_ece/table2_ece.md`
- `FIM-EDL/results/paper_tables/iedl_ece/table3_ece.md`
- `FIM-EDL/results/paper_tables/iedl_table4_official_v5/table4_iedl_style.md`
- `FIM-EDL/results/eval/summary_mean_std.csv`

현재 이 논문은 `evidence-aware uncertainty를 위한 adaptive KL
regularization` 논문으로 작성할 때 가장 강하다. 따라서 추가
실험도 논문을 불필요하게 넓히기보다는, 이 핵심 서사를 더
단단하게 뒷받침하는 방향으로 들어가야 한다.

## 현재 결과 요약

### 이미 강한 부분

- I-EDL 스타일 Table 2 비교에서 `Info-EDL`은 4개 모든 과업에서
  `alpha0` 기반 OOD AUPR을 개선한다.
- 가장 큰 개선은 CIFAR-10 OOD 과업에서 나타난다.
  - `CIFAR10 -> SVHN`: `91.48 +/- 0.93 -> 95.93 +/- 1.02`
  - `CIFAR10 -> CIFAR100`: `75.54 +/- 1.49 -> 84.04 +/- 0.26`
- I-EDL 스타일 Table 3 비교에서 `Info-EDL`은 CIFAR-10 전체
  요약 성능이 가장 좋다.
  - `Max.P AUPR`: `98.71 +/- 0.10 -> 98.78 +/- 0.08`
  - `Max.alpha AUPR`: `98.71 +/- 0.10 -> 98.75 +/- 0.10`
  - `Accuracy`: `89.66 +/- 0.81 -> 90.41 +/- 0.35`
- ECE 역시 큰 폭으로 개선된다.
  - `MNIST`: `36.80 +/- 0.32 -> 0.40 +/- 0.14`
  - `CIFAR10`: `40.24 +/- 0.50 -> 2.95 +/- 0.73`

### 아직 방어가 부족한 부분

- 제안한 adaptive controller 때문에 성능이 좋아졌다는 점을
  직접 보여주는 mechanism-level ablation이 아직 부족하다.
- `summary_mean_std.csv`에 있는 fixed-`lambda` EDL 비교는 일부
  score type에서 seed 수가 부족하다(`n=2`). 그래서 현재 상태로는
  메인 테이블 근거로 쓰기에 깔끔하지 않다.
- few-shot 결과는 현재 `Info-EDL`에 불리하므로, 이 논문의 핵심
  실험 방향으로 잡으면 오히려 원고가 약해진다.

## 권장 추가 실험

## 우선순위 1: 메커니즘을 검증하는 ablation

이 실험들이 가장 중요하다. 이 부분이 없으면 리뷰어는
“성능이 좋아진 것은 보이지만, 왜 좋아졌는지는 충분히
분리해서 보여주지 못했다”고 지적할 수 있다.

### 1. 동일한 학습 파이프라인에서 adaptive KL과 fixed KL 비교

다음 조건으로 matched ablation을 수행한다.

- `fixed lambda`
- `adaptive lambda = beta * exp(-gamma * v_FIM)`

다음 요소는 모두 동일하게 유지한다.

- architecture
- optimizer
- training schedule
- seeds
- evaluation pipeline

주요 보고 지표는 다음과 같다.

- Table 2 스타일의 `alpha0` OOD AUPR
- Table 3 스타일의 `Max.P`, `Max.alpha`, `Accuracy`
- 보조 지표로 ECE

이 실험이 필요한 이유:

- 전역 KL weight는 서로 다른 local sensitivity를 갖는 evidence
  state들에 대해 부적절할 수 있다는 논문의 핵심 주장을 직접
  검증할 수 있다.
- 이론 파트와 실험 파트를 가장 깔끔하게 연결해주는 실험이다.

### 2. Controller 구성요소 ablation

각 설계 요소가 실제로 필요한지 확인한다.

- `detach`를 적용한 `v_FIM` gate
- `detach` 없이 적용한 `v_FIM` gate
- `alpha0` 기반 gate
- constant gate

기대한 해석 방향은 다음과 같다.

- `v_FIM`이 `alpha0` 및 constant baseline보다 좋다면, 단순한
  evidence 크기보다 local sensitivity가 더 유의미한 제어 신호라고
  주장할 수 있다.
- `detach`가 도움이 된다면, optimization 관점의 설명이 훨씬
  강해진다.

이 실험이 필요한 이유:

- 리뷰어는 성능 향상이 Fisher information 자체에서 오는지,
  아니면 evidence에 단조적으로 반응하는 아무 weighting이면
  되는지 물을 가능성이 높다.

### 3. `beta`와 `gamma` 하이퍼파라미터 민감도

다음과 같은 작은 grid를 권장한다.

- `beta in {0.5, 1.0, 2.0}`
- `gamma in {0.5, 1.0, 2.0}`

최소한 다음 결과는 포함하는 것이 좋다.

- `CIFAR10 -> SVHN`
- `CIFAR10 -> CIFAR100`
- CIFAR-10 misclassification summary

이 실험이 필요한 이유:

- 현재 논문은 제안 기법이 특정 한 점의 튜닝에만 의존하는
  불안정한 결과가 아니라는 점을 보여줘야 한다.
- Neurocomputing 리뷰어는 성능 개선이 적당한 수준의
  hyperparameter 변화에도 유지되는지 확인하는 경우가 많다.

## 우선순위 2: 강한 보조 실험

위 ablation만큼 핵심적이지는 않지만, 리뷰어 비판을 줄이는 데
실질적으로 도움이 되는 실험들이다.

### 4. MNIST와 CIFAR-10에 대한 reliability diagram

ECE 개선폭이 이미 크기 때문에, 표뿐 아니라 시각적 근거도
추가하는 것이 좋다.

- `I-EDL Ref`의 reliability diagram
- `Info-EDL`의 reliability diagram

권장 데이터셋:

- MNIST
- CIFAR-10

이 실험이 필요한 이유:

- ECE 표는 강하지만 정보량이 짧다.
- calibration 개선을 그림으로 보여주면 discussion에서 인용하기도
  쉽고, 독자가 신뢰하기도 쉽다.

### 5. Evidence-state 분포 분석

다음 값들에 대한 histogram 또는 density plot을 추가한다.

- `alpha0`
- `v_FIM`
- `lambda`

다음 집단을 비교한다.

- 올바르게 분류된 ID 샘플
- 오분류된 ID 샘플
- OOD 샘플

권장 분석 초점:

- `CIFAR10 vs SVHN`
- `CIFAR10 vs CIFAR100`

이 실험이 필요한 이유:

- 이론적으로는 제안 기법이 evidence accumulation 방식을
  재구성한다고 설명하고 있다.
- 그렇다면 실제로 OOD 샘플이나 오류 가능성이 큰 샘플에서
  evidence concentration이나 effective regularization이 다르게
  나타난다는 것을 보여주는 편이 훨씬 설득력 있다.

### 6. Adaptive coefficient의 학습 동역학

학습 epoch에 따라 다음 값을 그린다.

- mean `lambda`
- standard deviation of `lambda`
- mean `v_FIM`
- training loss components

이 실험이 필요한 이유:

- adaptive KL 항을 단순한 정적 공식이 아니라 실제로 작동하는
  training mechanism으로 보여줄 수 있다.
- controller가 거의 상수로 붕괴한 것이 아니라 실제로 학습 과정에서
  작동하고 있다는 점을 방어하는 데 도움이 된다.

## 우선순위 3: 선택적이지만 가치 있는 확장

시간이 허용될 때 고려할 만하지만, 위 실험들보다 우선순위는 낮다.

### 7. 추가 backbone 1개

다음과 같은 추가 backbone 하나를 사용한다.

- WideResNet
- VGG-style CNN

최소 권장 범위:

- CIFAR-10 Table 3 스타일 비교 1개
- CIFAR-10 OOD 과업 1개

이 실험이 도움이 되는 이유:

- 현재 결과가 특정 모델 구조나 구현 세부사항에만 의존한 것이
  아니라는 점을 보여줄 수 있다.

### 8. 현재 표 밖의 추가 matched OOD benchmark 1개

가능한 후보는 다음과 같다.

- `CIFAR10 -> LSUN`
- `CIFAR10 -> TinyImageNet crop / resize`

이 실험이 도움이 되는 이유:

- 현재 OOD 서사는 이미 강하지만, 자연 이미지 계열 OOD benchmark를
  하나 더 넣으면 CIFAR-10 결과를 더 설득력 있게 만들 수 있다.
- 저널 리뷰어가 SVHN와 CIFAR100 외에 더 표준적인 OOD benchmark를
  기대할 가능성에도 대비할 수 있다.

## 우선순위를 낮춰야 하는 실험

### 1. Few-shot 확장

few-shot 실험은 결과가 크게 바뀌지 않는 한 논문 분량을 더
투입하지 않는 편이 좋다.

이유:

- 현재 few-shot 표에서는 `Info-EDL`이 상당히 약하다.
- 이 결과는 핵심 주장에 힘을 실어주지 못하며, 오히려 원고를
  분산시킨다.

### 2. “모든 metric 개선” 서사를 위한 광범위한 탐색

다양한 score variant를 많이 추가해서 universal improvement처럼
보이게 만들려고 할 필요는 없다.

이유:

- 현재 결과는 이미 구조적인 패턴을 보여준다.
  `Info-EDL`은 특히 `alpha0` 기반 OOD detection처럼
  evidence-sensitive uncertainty에서 강하다.
- 넓지만 일관성 없는 논문보다, 좁지만 잘 방어된 논문이 더 강하다.

## 제출 전 최소 실험 패키지

시간이 부족하다면 최소한 다음은 수행하는 것이 좋다.

1. matched `adaptive vs fixed lambda` ablation
2. controller component ablation (`v_FIM`, `alpha0`, constant, detach`)
3. `beta/gamma` sensitivity study
4. calibration visualization figure 1개

이 조합이 비용 대비 리뷰어 방어력 측면에서 가장 효율적이다.

## 논문 내 배치 권장안

### 메인 본문

- matched adaptive-vs-fixed ablation
- controller ablation
- compact `beta/gamma` sensitivity table
- calibration figure 1개

### Appendix 또는 supplementary material

- evidence histogram
- training dynamics plot
- 추가 architecture 결과
- 추가 OOD benchmark 결과

## 이 실험들이 막아주는 리뷰어 질문

### Q1. 왜 이 방법이 동작하는가?

- controller ablation과 evidence-state 분석으로 답할 수 있다.

### Q2. 성능 향상이 adaptive KL 때문인가, 아니면 단순히 KL이 더 들어가서인가?

- matched fixed-vs-adaptive 비교로 답할 수 있다.

### Q3. 하이퍼파라미터에 민감하지 않은가?

- `beta/gamma` 민감도 실험으로 답할 수 있다.

### Q4. calibration 개선이 실제 현상인가, 아니면 표 숫자만 좋은 것인가?

- reliability diagram으로 답할 수 있다.

### Q5. controller가 학습 중 실제로 작동하는가?

- training dynamics 분석으로 답할 수 있다.

## 최종 권고

다음 실험들은 논문을 넓히기 위한 것이 아니라, 지금 이미 보이는
핵심 주장을 더 깊고 단단하게 만들기 위한 방향으로 선택해야 한다.

현재 논문은 다음 주장을 할 때 가장 강하다.

- `alpha0` 기반 OOD detection 개선
- CIFAR-10 confidence ranking과 accuracy 개선
- adaptive evidence-aware KL control에 기반한 메커니즘

따라서 가장 좋은 다음 단계는 새로운 task family를 더 늘리는 것이
아니라, 이 서사를 공격하기 어렵게 만드는 소수의 정교한 ablation과
분석 실험을 추가하는 것이다.
