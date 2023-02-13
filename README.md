# Movie Recommendation

### 1-1. 프로젝트 소개

<img src="https://user-images.githubusercontent.com/55279227/218377694-d1d7ca3d-139f-40d9-8ec6-22514ffdb720.png" width="800" height="300"/>

- 본 프로젝트는 **Movie Recommendation**, 즉 유저의 영화 시청 이력 데이터를 바탕으로 유저에게 볼 영화를 추천해주는 대회에서 솔루션을 제시한다.
- Timestamp 정보가 들어간 implicit feedback 데이터를 통해 모든 유저에게 10개의 영화를 추천하여 추천 리스트의 정확성을 평가한다.
- 평가를 위한 정답(ground-truth) 데이터는 sequential recommendation 시나리오를 바탕으로 하는 동시에
    
    복잡한 실제 상황을 가정하여 다음과 같이 원본 데이터에서 추출되었다.
    
    - 특정 시점을 기준으로 이후의 데이터를 전부 추출
    - 특정 시점을 기준으로 이전의 데이터 중 랜덤하게 추출

- 위와 같이 데이터셋이 구성되어 있기 때문에, 단순 sequential한 예측에 최적화된 모델은 static 데이터의 예측에 약점을 보일 수 있고, 반대로 static한 예측에 최적화된 모델은 sequential한 예측에 약점을 보일 수 있다.

- 대회에서는 implicit feedback 기반의 sequential recommendation 시나리오를 바탕으로 사용자의 time-ordred sequence에서 일부 item이 누락된 (dropout)된 상황을 상정하기 때문에, 마지막 아이템 예측 뿐만아니라 중간 아이템 또한 예측해야 한다.


### 1-2. 데이터 요약

- 본 대회는 추천시스템 연구 및 학습 용도로 가장 널리 사용되는 MovieLens 데이터를 implicit feedback의 형식으로 수정하여 사용한다.
- **input**
    - 유저와 아이템의 상호작용 데이터 5,154,471개
    - 아이템의 메타 데이터
        - 장르, 제목, 개봉년도, 감독, 작가
        - 장르와 제목을 제외하고 아이템의 메타 데이터에는 결측이 존재한다.
    - 유저는 총 31360명이며, 아이템은 총 6807개이다.
    - 유저와 아이템의 번호가 0 ~ 1에서 시작해 arange 되어있지 않았기에, 라벨인코딩 후 진행하였다.
- **output**
    - 유저 각 31360명에 대해 10개의 아이템을 추천하여 313600 행의 user, item 컬럼을 가진 csv 파일을 제출한다.

### 1-3. 평가 지표
<img src="https://user-images.githubusercontent.com/55279227/218377636-2b2eff70-9eb9-48d3-9aa4-173a9b270c63.png" width="800" height="150"/>

- Recall@10 - 각 유저에 대해 10개의 영화를 추천하고, 이는 각 유저의 Recall 지표에 대한 평균을 통해 평가된다.
- 통상적인 Recall 지표는 유저가 관심 있는 모든 아이템 중에서 모델이 추천한 아이템 k개가 얼마나 포함되는지의 비율을 의미한다.
- 그러나 대회에서는 k와 사용자가 사용한 아이템 수 $|I_u|$ 중 최소값을 분모로 사용함으로써 지표의 값을 최대 1이 되도록 고정하였다.
    
    이는 유저마다 정답 데이터의 갯수가 다르더라도 Recall 지표가 0과 1 사이로 나오도록 정규화(normalize)하는 효과를 가진다.
    

### 1-4. 장비 정보

- GPU: Tesla v100 / 30.0 GB
- RAM: 88 GB

### 1-5. 개발 환경

- <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white"><img src="https://img.shields.io/badge/vsc-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"><img src="https://img.shields.io/badge/anaconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white"><img src="https://img.shields.io/badge/w&b-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=white">

## 2. 프로젝트 기획

<img src="https://user-images.githubusercontent.com/55279227/218377992-c2af78bc-9b88-4f49-8cd7-2ca22c2eb760.jpg" width="500" height="500"/>
 - 지금까지 학습한 추천시스템의 전반적인 모델들을 구현, 사용

 - 베이스라인 분석, eda 진행, 대회 Validation Set 과 비슷한 Set 구성

 - 하드보팅 등 다양한 앙상블 기법 시도
<img src="https://user-images.githubusercontent.com/55279227/218378054-a4181bee-a6a4-4810-accd-0e130ba58a83.png" width="700" height="300"/>

## 3. 프로젝트 팀 구성 및 역할
| [<img src="https://github.com/soso6079.png" width="100px">](https://github.com/soso6079) | [<img src="https://github.com/Zerotay.png" width="100px">](https://github.com/Zerotay) | [<img src="https://github.com/mbaek01.png" width="100px">](https://github.com/mbaek01) | [<img src="https://github.com/p-idx.png?v=4" width="100px">](https://github.com/p-idx) | [<img src="https://github.com/sj970806.png?v=4" width="100px">](https://github.com/sj970806) |  
| :---: | :---: | :---: | :---: | :---: |  
| [강수헌](https://github.com/soso6079) | [김동건](https://github.com/Zerotay) | [백승렬](https://github.com/mbaek01) | [양성훈](https://github.com/p-idx) | [유상준](https://github.com/sj970806) |

- 강수헌: DeepFM 구현 (negative-sampling, inference), Multi-VAE 구현, Top-k 앙상블 구현
- 김동건: 베이스라인 코드 분석, BPR 논문 공부, 인기도 기반 추천,
    - RecBole 라이브러리를 사용하는 베이스라인 구축, RecBole을 사용하기 위한 다양한 기능 구현, RecBole을 통해 다양한 모델 테스트
- 백승렬: DeepFM 구현 (data preprocessing, validation step), UltraGCN 구현, Top-k inference 앙상블 method 구현
- 양성훈:  베이스라인 분석, memory-based KNN, MF, BPR-MF, NMF 구현
- 유상준: EDA, MultiVAE 베이스라인 작성, EASE 모델 공부, RecBole 라이브러리, wandb sweep 사용(DeepFM, SASRec 하이퍼파라미터 튜닝)

