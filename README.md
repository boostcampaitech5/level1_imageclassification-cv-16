## CV16 

1. 프로젝트 개발환경 구축(Github, WandB, VSCode SSH)

2. EDA를 통한 데이터의 구조 파악 및 의미 분석
 
  - 매우 불균형한 데이터셋 > 데이터 증강, 전처리의 중요성 파악

  - 각 특징을 잘 뽑아낼 수 있도록 3개의 모델을 독립적으로 학습

3. 주어진 baseline 코드를 기반으로 한 최적의 모델, Center crop, resize의 조합 실험
4. Center crop, Resize의 변화에 따른 성능 실험, Learning Rate, Loss, Augmentation, Batch size, Loss+Optimizer 조합 실험
5. 3개의 독립적인 Age, Gender, Mask classification 모델 성능 확인 및 baseline코드 작성
6. 3개의 독립적인 모델의 Loss, Gradient Accumulation, FC layer initialization, Age classification threshold 조절 실험
7. Classification이 잘 되지 않는 Age 클래스에 대해 EDA 수행, 관련 실험 가설 세운 뒤 실험
8. 높은 성능의 모델들로 Soft voting을 통해 최종 결과물 제작
