<br/>
<br/>

# 0. Getting Started (시작하기)
```bash
# 방산_불량분류_통합_정리.ipynb 를 Jupyter/Colab에서 순서대로 실행
pip install pandas numpy scikit-learn statsmodels xgboost streamlit
```
분석 결과 대시보드는 노트북 내 Streamlit 섹션 코드를 `streamlit run`으로 별도 실행합니다.

<br/>
<br/>

# 1. Project Overview (프로젝트 개요)
- 프로젝트 이름: **[방산] 양극산화피막 데이터 기반 불량 조기경보 시스템 구축**
- 프로젝트 설명: 방산 부품(소총·탄피) 표면처리 공정의 전압·전류·온도 센서 값만으로, 검사 이전에 양극산화피막 불량을 자동 분류하고 이상 시점·원인을 추적하는 조기경보 시스템
- 기간: 2026.05

<br/>
<br/>

# 2. Team Members (팀원 및 팀 소개)
- **팀명**: 7조 「제-조」(닉네임 "방탄이") · 튜터 정강민

| 모상혁 | 이채준 | 정의조 | 최지호 | 홍상훈 |
|:---:|:---:|:---:|:---:|:---:|

> 팀 산출물에 개별 담당 업무가 인물별로 세분 기재되어 있지 않아, 4번 항목에는 확인 가능한 본인(최지호) 담당만 정리합니다.

<br/>
<br/>

# 3. Key Features (주요 기능)
- **현황 진단**: 사후 검사 중심 품질관리 → 초기 불량률·자재비 과다, 납기 지연 시 신뢰도 하락 문제를 센서(V·I·T) 기반 선제 진단으로 전환

- **EDA 핵심 통찰**:
  - 전압·전류 분포가 대기/공정 구간에서 뚜렷한 **이봉(Bimodal) 분포**를 형성 — 단일 분포 가정으로는 구간 구분 불가
  - 불량은 전류·전력·전압이 **양방향으로 분산**돼 단방향 평균으로는 상쇄됨 → **정상기준 절대이탈(|z|)** 전환이 성능 정체를 돌파한 지점

- **가설 검정(Mann-Whitney U + BH-FDR 다중비교 보정)**:
  - 평균 전류·전력 절대값이 정상 기준보다 낮으면 산화피막 목표미달 불량 (`ampere_mean_absz` p=2.1e-5, `power_mean_absz` p=8.2e-5) → 채택
  - 평균 전압이 정상 범위를 초과하면 변색 불량 (`volt_mean_absz` p=0.001) → 채택
  - 불량 초반 이상 신호 발생 가설은 기각 (p=0.63, 근거 부족)

- **파생변수 설계**: 전압·전류·온도·전력 4계열 × 평균/표준편차/최대/범위/절대이탈 조합으로 21개 파생 → VIF 다중공선성 제거 → **컴팩트 5피처**로 압축 (PR-AUC 0.41 → 0.91)

- **모델링 + 이상탐지 교차검증**: LOO-CV로 로지스틱회귀·XGBoost·RandomForest 비교, LOF·OneClassSVM·IsolationForest로 지도학습 결과 재검증

- **Streamlit MES형 대시보드**: 불량 판정 + 규칙 기반 불량유형 추정 + SPC 넬슨 규칙 5종 실시간 이상점수 + ROC/PR

<br/>
<br/>

# 4. Tasks & Responsibilities (작업 및 역할 분담)
| 이름 | 담당 |
|:---:|---|
| 최지호 | <ul><li>통계 검정(Mann-Whitney U · BH-FDR) 설계</li><li>파생변수 설계 및 VIF 기반 피처 압축</li><li>모델링(로지스틱회귀·XGBoost·RandomForest 비교)</li><li>이상탐지 교차검증(LOF·OneClassSVM·IsolationForest)</li></ul> |

<br/>
<br/>

# 5. Technology Stack (기술 스택)
## 5.1 Language
| | |
|---|---|
| Python | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |

<br/>

## 5.2 분석 / 모델링
| | | |
|---|---|---|
| Pandas / NumPy | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | 시계열 센서 데이터 처리 |
| statsmodels | ![statsmodels](https://img.shields.io/badge/statsmodels-2C4C7C?style=flat-square) | Mann-Whitney U · BH-FDR 다중비교 |
| scikit-learn | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) | 로지스틱회귀 · RandomForest · LOF · OneClassSVM |
| XGBoost | ![XGBoost](https://img.shields.io/badge/XGBoost-EB0028?style=flat-square) | 모델 비교군 |

<br/>

## 5.3 시각화 / 배포
| | | |
|---|---|---|
| Streamlit | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) | MES형 실시간 판정 대시보드 |
| Tableau | ![Tableau](https://img.shields.io/badge/Tableau-E97627?style=flat-square&logo=tableau&logoColor=white) | 보조 시각화 |

<br/>

## 5.4 Cooperation
| | |
|---|---|
| Git | ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white) |
| Jupyter | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white) |

<br/>
<br/>

# 6. Project Structure (프로젝트 구조)
```plaintext
양극산화피막불량예측/
├── 방산_불량분류_통합_정리.ipynb                       # EDA · 통계검정 · 파생변수 · 모델링 · 이상탐지 전체 코드
├── 결과_보고서_7팀(양극산화피막_불량조기경보) (1).pptx
├── 발표용PPT_제-조팀(양극산화피막_불량조기경보.pptx
└── 발표영상_제-조팀(양극산화피막_불량조기경보).mp4
```

<br/>
<br/>

# 7. Development Workflow (분석 워크플로우)
```
01 개요 (공정 소개 · 문제 정의)
  → 02 EDA (기초통계 · 이봉분포 확인 · 가설 설정·검정 · 파생변수 · 상관구조 · 구간분할)
    → 03 모델링 (피처 선택 → 모델 비교 → 임계치 최적화 → 성능 유의성 검증 → 이상탐지 비교 → 시퀀스 판정 → 원인 추적)
      → 04 대시보드 (공정 현황 · 조기경보 · 이상탐지)
        → 05 향후 계획 (기대효과 · 한계·개선방향)
```

<br/>
<br/>

# 8. Coding Convention
## 통계 검정 원칙
```
다중 가설을 동시에 검정할 때는 BH-FDR로 거짓양성 비율을 통제한다.
가설에서 언급된 변수 외에 관련 변수도 함께 검정해 우연성을 배제한다.
```

## 임계값 설계 원칙
```
미탐(false negative) 비용이 오탐 비용보다 큰 경우,
기본 임계값(0.5)을 그대로 쓰지 않고 재현율 100%를 만족하는 지점으로 재설계한다.
```

## 모델 채택 기준
```
지도학습 결과는 반드시 이상탐지 기법(LOF 등)과 교차검증해
동일한 이상 시점·원인 변수를 가리키는지 확인한 뒤 채택한다.
```

<br/>
<br/>

# 9. 커밋 컨벤션
## 기본 구조
```
type : subject

body
```

## type 종류
```
feat : 새로운 분석/기능 추가
fix : 로직 수정
docs : 문서 수정
refactor : 코드 리팩토링
chore : 환경/패키지 설정
```

## 커밋 이모지
```
📝 코드 작성   🔨 리팩토링   🐛 버그 리포트   🚑 버그 수정
📚 문서 작성   ✨ 새 기능    🚀 배포
```

<br/>
<br/>

# 10. 결과 요약

| 모델 | ROC-AUC | PR-AUC | 비고 |
|---|:--:|:--:|---|
| **로지스틱 회귀** ⭐ | 0.991 | **0.944** | 채택 — 소표본 과적합 적음, 해석·배포 단순 |
| XGBoost | 0.982 | 0.875 | |
| RandomForest | 0.951 | 0.755 | |

- **재현율 100%(11/11, 미탐 0건)**, 오탐률 7% (임계값 0.4391로 재설계)
- 이상탐지 교차검증: **LOF AUC 1.000** > OneClassSVM 0.869 > IsolationForest 0.657
- 한계: 불량 표본 11건에 불과 → Active Learning, 선순환 재학습, 설비별 모델 분리가 다음 과제
