# Oversampling for Imbalanced Data with SMOTE & GAN
* 센서 데이터를 활용한 공정 이상 예측
* 불균형 데이터 처리를 위한 오버 샘플링 - SMOTE, GAN  
* Overview(예정)  
<img> </br></br>

## Summary  
* 👩‍💻 **기여** : 모델링 아이디어 설계 및 모델링 (4명)
  
* 👩‍💻 **역할** : 팀장
  
* 👩‍💻 **기간** : 약 n주
	* 기간 </br></br>

## 작업환경
* **💻 학습** : Jupytaer Notbook
  
* **💻 코드** :  [모델링]()
  
* **💻 발표자료**  : [프로포절 발표](), [중간 발표](), [최종 발표]() </br></br>

## 데이터
* 반도체 생산 공정 과정에서 발생한 센서 데이터 (UCI Machine Learning Repository, SECOM Data set)
* Time : 센서 정보가 기록된 시간  
* 0 ~ 589 : 센서 번호
* Pass/Fail : 이상 여부 (1/-1)  

## EDA
![image](https://github.com/Seohee-Kim/uni-secom/assets/62201733/03ffcb5e-c25c-4314-91c4-77a8da09bbe2)
불균형 데이터는 모델 성능에 악영향을 줄 수 있다고 판단하여, 해당 분석에서는 기존 데이터로 분석을 진행한 후 모델 성능을 향상시킬 수 있는 방법으로 오버샘플링 기법인 SMOTE와 GAN, 두 가지 기법을 적용  

## 로지스틱 회귀
로지스틱 회귀 모델의 정확도는 89%이다. 하지만 정밀도, 재현율, F1-score 등이 매우 낮아 이를 예측 모델로 사용하기에는 부적합하다고 평가하였다. 또한 정오 행렬과 히트맵을 출력해본 결과, 1을 1로 예측하는 비율이 매우 낮음을 확인할 수 있었다.

```python
정확도 : 0.89, 정밀도 : 0.11, 재현율 : 0.10
f1-score : 0.11, auc : 0.52
```
![캡처2](https://github.com/Seohee-Kim/uni-secom/assets/62201733/f0bb44b7-3c2c-4360-b0fd-f495fdb60552)  


## 모델 성능 향상

### 1. 불균형 데이터 처리를 위한 SMOTE 기법의 적용

```python
정확도 : 0.84, 정밀도 : 0.11, 재현율 : 0.20
f1-score : 0.14, auc : 0.54
```
SMOTE 기법 적용 결과 모델의 성능이 아주 약간 향상되었다고 평가할 수 있는데, 비록 정확도는 떨어졌을지라도 재현율, F1-score, AUC가 이전에 비해 향상된 모습을 보였기 때문이다.
하지만 이 효과는 미비하며, 여전히 낮은 정밀도와 재현율을 모델이 보여주고 있기 때문에 로지스틱 회귀 모델을 벗어나 다양한 데이터 마이닝 기법을 탐색하였다.

![1](https://github.com/Seohee-Kim/uni-secom/assets/62201733/c61731b4-449c-4d93-a972-133196bdfc97)


### 2. 다양한 데이터마이닝 기법의 적용

SMOTE 적용 전에 대한 모델링의 결과에서는 KNN, RF, SVM, XGB, LGBM의 성능이 동일하게 93.63%로 나타났다. SMOTE 적용 후에 대한 모델링의 결과에서는 RF, XGB, LGBM의 성능이 동일하게 93.31%로 나타난 것으로 보아, SMOTE 기법이 크게 유의미하지 않았음을 알 수 있다. 또한 SMOTE 기법이 적용되었을 때, KNN의 경우에는 떨어지는 모습도 보임을 알 수 있었다.
따라서 결과를 종합한 결과, RF, XGB, LGBM 세 모델의 성능이 탁월하다고 평가하였고, 그 중 SMOTE 기법을 적용하지 않은 랜덤 포레스트를 잠정적인 최종 모델로 선택하였다.

![2](https://github.com/Seohee-Kim/uni-secom/assets/62201733/0b45c1a4-9d76-4802-a78a-95234d0ebfa5)

### 3. GridSearch를 통한 최종 모델의 하이퍼 파라미터 수정
```python
예측 정확도: 0.9363
```
하이퍼 파라미터 조정 결과, 하이퍼 파라미터 조정 전과 후의 모델의 성능이 동일하게 93.63임을 확인하였다.
따라서 하이퍼 파라미터 수정이 무의미하므로 조정 전 랜덤 포레스트 모델을 계속 사용하기로 결정하였다.

### 4. 불균형 데이터 처리를 위한 GAN 기법의 적용

중요도 top 30의 변수만 포함한 데이터 파일을 생성
* tgan_data: 행 번호와 열 번호는 tgan으로 모델을 생성하면서 자동으로 부여된 번호로 구성되어 있으며, 열 번호는 랜덤 포레스트에서 중요도가 높다고 나온 변수의 오름차순이다.
* Fail_data: tgan_data에서 Pass/Fail 열이 1에 해당하는 행만 저장한 파일이다.

로지스틱 회귀 결과

```python
정확도 : 0.92, 정밀도 : 0.86, 재현율 : 0.66
f1-score : 0.74, auc : 0.82
```
![캡처 3](https://github.com/Seohee-Kim/uni-secom/assets/62201733/1a39b29c-000b-4a9c-b5d9-86ca00b82af2)  

GAN을 적용하지 않은 데이터에 대해 진행한 로지스틱 회귀와, GAN을 적용한 데이터에 대해 진행한 로지스틱 회귀를 비교했을 때, GAN 기법 적용 후의 모델의 성능이 탁월하게 좋아졌음을 확인할 수 있었다.
![4](https://github.com/Seohee-Kim/uni-secom/assets/62201733/baa3d158-1809-4241-a806-45bdb1d06944)

CF. 나아가 혹여나 GAN으로 생성한 데이터 덕분이 아니라, 단순히 변수의 개수가 줄어서 모델의 성능이 좋아진 것은 아닐까 싶어 ‘uci-secom3’에 대해서도 로지스틱 회귀 분석을 실행한 결과, ‘uci-secom3’에 적용한 것은 원본 데이터인 ‘uci-secom’에 적용한 것보다 더 낮은 점수가 나온 것을 확인하였다.

### 5. GAN 기법과 SMOTE 기법의 혼합 적용
GAN 데이터에 SMOTE 기법을 적용했을 때 모델의 성능이 오히려 감소하는 경향으로 보아, GAN과 SMOTE 기법을 혼합 적용하는 것은 유의미하지 않다고 판단하였다.
따라서 GAN 데이터에 대해서 가장 성능이 높은 모델은 랜덤 포레스트와 Light GBM으로, 이 93.63%의 성능은 이전에 냈던 최고 성능과 동일하다. 

![66](https://github.com/Seohee-Kim/uni-secom/assets/62201733/bf06c3af-ee07-4c00-9087-33db1d169fce)

</br></br>
## 결론
최종 모델로 랜덤 포레스트와 Light GBM을 선택하였다.
앞선 분석 단계에서, 기존 데이터에 적용한 로지스틱 회귀의 성능이 매우 낮음에 따라 다양한 모델 성능 향상 기법을 적용하여 SMOTE 적용 전인 랜덤 포레스트 모델을 최종 선택하였었다. 그 이후 이 랜덤 포레스트를 사용하여 중요 변수 상위 30개를 추출할 수 있었으며, 이 변수 30개를 활용한 데이터에 오버샘플링 방법의 하나로 GAN을 적용하여 모든 모델링에 적용하였다.
기존 데이터에서 가장 높은 성능을 보이는 모델은 랜덤 포레스트, SVM, XGBoost, Light GBM이고, GAN 데이터의 SMOTE 기법 적용 전 데이터에 대해 가장 높은 성능을 보이는 것은 랜덤 포레스트와 Light GBM이다. 따라서 GAN 데이터에서의 성능을 고려하여 랜덤 포레스트와 Light GBM을 이상 센서 탐지를 위한 최종 모델로 선택하였다. 이 두 모델의 성능은 동일하게 93.63%이다.

![7](https://github.com/Seohee-Kim/uni-secom/assets/62201733/00c7bf42-e80b-4e94-85fb-cac29e95e485)



</br></br>
## 피드백

</br></br>
## 참고한 레퍼런스  
