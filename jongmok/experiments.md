* class imbalance 문제. sex, mask여부는 잘 구분하는 것 같지만 age는 60대 수가 너무 적어 잘 구분하지 않을 것 같다.
  * 의외로 60대 label은 잘 나옴. 대신 30\~60과 구분은 못하는 게 좀 있음.
  * focal loss?
  * stratified split?
  * 60대 data 위주 augmentation?
  * label별 비율에 따라 가중치 주기? - focal loss 사용하면 필요없다는 말도.
* augmentation시, 상하 flip, rotation은 별로 필요가 없을 것 같음.
  * color change(gray scale), brightness 정도?
  * blur?
  * rotation도 한번 해보기
  * augementation 다양하게 시도해보면서 overfitting 줄이기. expr 3에서 training loss도 처참한거 보면 augmentation은 약간만?
  * normalize만 해도 train 하락. validation은 약간만 떨어짐.
* 18class 대신 2, 3, 3, class 사용 후 합치기? - 안좋다는 말이 있음. 다시 해보기
* resnet50대신 SOTA모델 사용
  * efficientnetb0은 딱히 좋아보이지 않음.
  * efficientnetb7(noisy student) - 더 악화됨. train 수렴은 잘 되지만, generalization이 너무 안됨.
  * https://huggingface.co/google/vit-huge-patch14-224-in21k 써보기 - 너무 느림. 한 배치 training foward하는데만 15초 걸림.
* 애매한 클래스가 있으니 voting?
* lr scheduler
  * 만약, training loss는 괜찮아지는데 valid loss는 나빠지고 있을 때(overfitting) scheduling해봤자 valid loss 나빠지는 건 그대로일 것 같았다. - 맞음.
  * 이후 overfitting을 어느정도 해결했을 때, training/valid가 같이 떨어지다가 진동하기 시작하면 그때 scheduler쓰는 걸로.
* cross entropy와 f1 score가 살짝 맞지 않는 것 같음(loss가 떨어져도 f1 score가 떨어지거나, 그 반대의 경우가 있음)
  * f1 loss?
  * label smoothihng?
* gender labeling잘못된 게 보임
* center crop 300?
* center crop 350 250?
* validation 분포는 다르겠지만, baseline과 내가 짠 거 둘 다 validation 잘 나온다. 그냥 쓰던거 쓰면 될 거 같음.
-------------------------------
* Experiment1 - resnet50 pretrained, fine-tuning, earlystop only with 5 patience. `baseline.pt`
  * f1 0.6983, acc 77.1111. 
  * train loss : 0.002479789435818606, acc : 99.9668960571289% f1 score :  0.9996689633967917
  * val loss 0.024892299166205733, acc : 99.3644027709961%, f1 score : 0.9936351129351663
  * Generalization이 필요해 보임.
  * 특정 클래스에 약간 치중된 듯
  * age의 경계에 걸치는 나이(30대 근처, 60대 근처)를 구분을 잘 하지 못하는 것으로 보임
  * 두 개 이상의 클래스가 틀리는 경우는 X
  * wear-incorrect도 많이 보이지만, 의외로 wear-not wear도 많이 보임.
  * 성별 잘못 분류도 있음.
  > age 경계면 9  
  > Wear-Incorrect 구분 6  
  > Wear-Not Wear 4  
  > Male-Female 2  
  > gender label잘못됨 3  
  * train, split reimplementation
    * train loss : 0.039253381324608386, acc : 99.43724060058594% f1 score :  0.9943340872899595
    * val loss : 0.2687831801614898, acc : 92.13453674316406%, f1 score : 0.9185356422781539
--------------------------------
* Experiment2 - EfficientNetB0 pretrained, 나머지는 Experiment1과 동일. `baseline-efficientnetb0`
  * train loss : 0.0024297411687180335, acc : 99.98014068603516% f1 score :  0.9998015592425159
  * val loss : 0.02403895104103189, acc : 99.47034454345703%, f1 score : 0.9947058589492731
  * loss는 좀 나아짐. 의미있는 수준인지는 모르겠음.
    * 아닌 것 같다. 비슷하지만 좀 떨어짐.
  * train, split reimplementation
    * train loss : 0.13743188210084276, acc : 96.26588439941406% f1 score :  0.9609916190278511
    * val loss : 0.2737329411557165, acc : 91.47245788574219%, f1 score : 0.916047304796789
--------------------------------
* Experiment3 - EfficientNetB0 pretrained, Experiment1 + Transforms. `baseline-transform`
  * HorizontalFlip(p=0.2),
    RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.2),
    GaussNoise(var_limit=(1000, 1600), p=0.3),
  * Normalize with training mean, std
  * train loss : 0.0026765727077841897, acc : 99.96027374267578% f1 score :  0.9996026915268572
  * val loss : 0.03413961021449383, acc : 99.25847625732422%, f1 score : 0.99256610292711
  * train loss, val loss 둘 다 나빠짐. ~~generalization이 된 건가??~~ f1 0.6577, acc 74.8095.
    * 더 하락?? 함????
    * val dataset의 변화가 필요할 것 같음. 뭔가 잘못됐다. -> image들을 모두 train, val split하지 않고, 사람 별로 먼저 split. val에 train에서 나온 사람이 들어가 train, val에 overfitting된것같음.
    * validation 변화해도 하락한건 똑같다. 다른 augmentation이 필요할듯(아니면 원본 + augemted data도 시도해보고)
  * train, split reimplementation
    * train loss : 0.2616934412301092, acc : 91.33341979980469% f1 score :  0.9070708082298238
    * val loss : 0.2754962898285712, acc : 90.41313171386719%, f1 score : 0.9038963333123528
------------------------------
* Experiment4 - EfficientNetB0, Normalize only
  * train loss : 0.14067439172195934, acc : 95.92823028564453% f1 score :  0.9559032220700969
  * val loss : 0.27292504234207887, acc : 91.23411560058594%, f1 score : 0.911813878142406
------------------------------
* Experiment5 - EfficientNetB7_NS, Normalize only
  * train loss : 0.12528701409206, acc : 95.9259262084961% f1 score :  0.9590216154742384
  * val loss : 0.3853867331369764, acc : 88.003173828125%, f1 score : 0.8843565680946837
------------------------------
* Experiment6 - resnet50, Normalize only `efficientb7_ns.pt`
  * train loss : 0.030149765360027045, acc : 99.83448028564453% f1 score :  0.9983411338283734
  * val loss : 0.2762065220434787, acc : 91.47245788574219%, f1 score : 0.9151191911536113
  * normalize안한거랑 사실상 별 차이 없음.
------------------------------
* Experiment7 - resnet50, Normalize only, f1 loss `resnet50-f1loss.pt`
  * f1 : 0.6871, acc : 77.2222
  * train loss : 0.40989489259861284, acc : 99.25847625732422% f1 score :  0.9907090851478969
  * val loss : 0.48181446073418954, acc : 91.49893951416016%, f1 score : 0.913683941468289
  * cross entropy -> f1 loss로 교체
  * expr 6와 비교하면 f1 score는 조금 떨어짐.
  * metric및 loss에 진동성이 좀 있음. lr=1e-5인데, 조금 더 낮출 필요가 있음
  * 생각한 것 보다는 loss와 metric간 완전히 비례하는 관계로 보이지는 않음.
  * f1 loss를 썼는데 오히려 f1 score는 떨어짐.
  * 그래도 지금 training하는 과정에서 validation set으로 성능 평가하면 될듯. 실제 f1 loss와도 비슷한 차이와 성능이 나기 때문.
-------------------------------
* Experiment8 - resnet50, Normalize+RandomBrightnessContrast(contrast_limit=(0.2, 0.8), p=0.4) `resnet50-contrast`
  * f1 : 0.6856, acc : 75.8413
  * train loss : 0.010651914501992859, acc : 99.91393280029297% f1 score :  0.999138836902512
  * val loss : 0.29191566339173053, acc : 92.55826568603516%, f1 score : 0.9240679567623593
  * 지금까지 최고기록 - 제출은 더 안좋음. augmentation 없는 것 보다 더 나쁨.
  * 대비를 강하게 주면 특징을 더 잘 잡아내지 않을까라는 아이디어에서 출발.
  * metric및 loss에 진동성이 좀 있음. lr=1e-5인데, 조금 더 낮출 필요가 있음
-------------------------------
* Experiment9 - resnet50, Expr8 + Centercrop(350, 250)
  * val loss : 0.2857, 91.711, 0.9178
  * 오르고 있는 중이었으니까 center crop 해도 괜찮을듯
-------------------------------
* Experiment10 - efficientnetb0, Expr9 transform
  * train loss : 0.047149082679317285, acc : 99.05985260009766% f1 score :  0.9905778438770385
  * val loss : 0.3489935115232306, acc : 89.83050537109375%, f1 score : 0.8986946314348773
  * lr 5e-6
  * Efficientnet은 사용하면 안되는건가
-------------------------------
* Experiment11 - three head classification, resnet50, Expr9 transform `3head-resnet`
  * train loss : 0.002150280901573342, 0.00409655287554155, 0.005313523910710559, acc : 99.82804870605469% f1 score :  0.9982805558513588
  * val loss : 0.02373085848341238, 0.04018776783765535, 0.24725221756331955, acc : 92.31991577148438%, f1 score : 0.9218842195142808
  * 좋아보이는데 성능 확인 필요.
  * wandb logging도 안함.
  * transform 빼고 logging+training하기