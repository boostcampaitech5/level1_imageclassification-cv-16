## CV16 

1. 프로젝트 개발환경 구축(Github, WandB, VSCode SSH)

2. EDA를 통한 데이터의 구조 파악 및 의미 분석
    - 매우 불균형한 데이터셋 > 데이터 증강, 전처리의 중요성 파악
    - 각 특징을 잘 뽑아낼 수 있도록 3개의 모델을 독립적으로 학습
3. 주어진 baseline 코드를 기반으로 한 최적의 모델, Center crop, resize의 조합 실험 후 다음 조합을 선택
    - model : resnet34
    - resize : 256, 192
    - Centrer Crop : 375, 200
4. Center crop, Resize의 변화에 따른 성능 실험, Learning Rate, Loss, Augmentation, Batch size, Optimizer 조합 실험 후 다음 조합을 선택
    - Learning : 1e-6
    - Loss : Weight Cross Entropy
    - Optimizer : AdamW
    - Augmentation : HFlip, Gaussian Noise, 
    - Batch Size : 16
    
5. 3개의 독립적인 Age, Gender, Mask classification 모델 성능 확인 및 baseline코드 작성
    - 각 특징에 조금 더 집중할 수 있도록 모델을 독립적으로 학습
    - 성능 향상
    
6. 3개의 독립적인 모델의 Loss, Gradient Accumulation, FC layer initialization, Age classification threshold 조절 실험
    - 성능이 하락하여 조합에서 제외

7. Classification이 잘 되지 않는 Age 클래스에 대해 EDA 수행, 관련 실험 가설 세운 뒤 실험
    - Age 클래스의 >= 60 데이터 Upsampling > 성능 하락
    - Age 모델의 Loss 변화 (Label Smoothing Loss) > 성능 하락
    - Age 모델의 Label 세분화 > 성능 하락
    - Age 모델의 다양한 Augmentation 적용 > 성능 향상
    - Age 모델을 3개의 독립적인 모델(이진 분류)로 분할 > 성능 향상

8. 높은 성능의 모델들로 Soft voting을 통해 최종 결과물 제작
    - 성능 향상
