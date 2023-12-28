# Oversampling for Imbalanced Data with SMOTE & GAN
* 불균형 데이터 처리  
* 센서 데이터를 활용한 공정 이상 예측 모델링  
<img> </br></br>


## Summary  
* 👩‍💻 **기여** : 모델링 아이디어 설계 및 모델링 (4명)
  
* 👩‍💻 **역할** : 팀장
  
* 👩‍💻 **기간** : 약 3주  
	* 프로포절부터 최종발표까지 (2021.05.12 ~ 2021.06.09) </br></br>

## 작업환경
* **💻 학습** : Jupyter Notbook
  
* **💻 코드** :  [모델링](https://github.com/Seohee-Kim/Oversampling/blob/main/uni-secom_final.ipynb)
  
* **💻 발표자료**  : [프로포절 발표](https://github.com/Seohee-Kim/Oversampling/blob/main/DS2%ED%8C%80_%ED%94%84%EB%A1%9C%ED%8F%AC%EC%A0%88%EB%B0%9C%ED%91%9C.pdf), [중간 발표](https://github.com/Seohee-Kim/Oversampling/blob/main/DS2%ED%8C%80_%EC%A4%91%EA%B0%84%EB%B0%9C%ED%91%9C.pdf), [최종 발표](https://github.com/Seohee-Kim/Oversampling/blob/main/DS2%ED%8C%80_%EC%B5%9C%EC%A2%85%EB%B0%9C%ED%91%9C.pdf) </br></br>


## 데이터
* 반도체 생산 공정 과정에서 발생한 센서 데이터 (UCI Machine Learning Repository, SECOM Data set)  
  
	* Time : 센서 정보가 기록된 시간    
	* 0 ~ 589 : 센서 번호  
	* Pass/Fail : 이상 여부 </br></br>


## EDA - Imbalanced Data  
* 라벨이 약 10:1의 비중으로 불균형하게 분포  </br>
* 먼저 기존 데이터의 모델 성능을 확인하고, 후에 성능 향상을 위한 다양한 방법론을 적용함 </br></br>
![image](https://github.com/Seohee-Kim/uni-secom/assets/62201733/03ffcb5e-c25c-4314-91c4-77a8da09bbe2) </br></br>


## AS-IS (로지스틱 회귀)  
```python
정확도 : 0.89, 정밀도 : 0.11, 재현율 : 0.10, f1-score : 0.11, auc : 0.52
``` 
* 정확도를 제외한 모든 지표가 매우 낮아, 예측 모델로 사용하기에 부적합 </br>
* 정오 행렬과 히트맵을 출력해본 결과, 1을 1로 예측하는 비율이 매우 낮음 </br>

![캡처2](https://github.com/Seohee-Kim/uni-secom/assets/62201733/f0bb44b7-3c2c-4360-b0fd-f495fdb60552) </br></br>


## TO-BE (모델 성능 향상)

### 1. 불균형 데이터 처리를 위한 SMOTE 기법의 적용  
![1](https://github.com/Seohee-Kim/uni-secom/assets/62201733/c61731b4-449c-4d93-a972-133196bdfc97) </br>
* **성능 약간 상승** : 비록 정확도는 떨어졌을지라도, 재현율, F1-score, AUC가 이전에 비해 향상됨  
* 하지만 여전히 낮은 정밀도와 재현율의 개선을 위 다양한 데이터마이닝 기법을 탐색함  </br></br>

### 2. 다양한 데이터마이닝 기법의 적용
![2](https://github.com/Seohee-Kim/uni-secom/assets/62201733/0b45c1a4-9d76-4802-a78a-95234d0ebfa5) </br>
* SMOTE 적용 **전** : KNN, RF, SVM, XGB, LGBM의 성능이 동일하게 93.63%  
* SMOTE 적용 **후** : RF, XGB, LGBM의 성능이 동일하게 93.31% (KNN의 경우, 정확도가 떨어지는 모습도 보임) </br>
* **SMOTE 기법이 유의미하지 않음** 
* RF, XGB, LGBM 세 모델의 성능이 탁월함
* SMOTE 기법을 적용하지 않은 랜덤 포레스트를 잠정적인 최종 모델로 선택함 </br></br>

### 3. GridSearch를 통한 최종 모델의 하이퍼 파라미터 수정
* 하이퍼 파라미터 조정 전과 후의 모델의 성능이 동일한 93.63%임
* **하이퍼 파라미터 조정이 유의미하지 않음** 
* 조정 전 랜덤 포레스트 모델 계속 선택함  </br></br>

### 4. 불균형 데이터 처리를 위한 GAN 기법의 적용
![4](https://github.com/Seohee-Kim/uni-secom/assets/62201733/baa3d158-1809-4241-a806-45bdb1d06944) </br>
* 중요도 top 30의 변수만 포함한 데이터 파일을 생성하여 분석 </br>
	* tgan_data: 열 번호는 랜덤 포레스트에서 중요도가 높다고 나온 변수의 오름차순  
	* Fail_data: tgan_data에서 Pass/Fail 열이 1에 해당하는 행만 저장한 데이터  </br></br>
* **GAN을 적용한 로지스틱 회귀 모델의 성능이 탁월하게 개선됨** </br>
  
	> 💡 단순히 변수의 개수가 줄어서 모델의 성능이 좋아진 것은 아닐까? </br>
	> ‘uci-secom3’에 대해서도 로지스틱 회귀 분석을 실행한 결과,
 	> ‘uci-secom3’에 적용한 것은 원본 데이터인 ‘uci-secom’에 적용한 것보다 더 낮은 점수가 나온 것을 확인
  
</br>  

### 5. GAN 기법과 SMOTE 기법의 혼합 적용  
![66](https://github.com/Seohee-Kim/uni-secom/assets/62201733/bf06c3af-ee07-4c00-9087-33db1d169fce) </br>
* 혼합 적용은 오히려 모델의 성능을 감소시켰음
* 따라서 GAN 데이터에 대해서 가장 성능이 높은 모델은 랜덤 포레스트와 Light GBM (93.63%) </br></br>

## 결론 및 요약  
![7](https://github.com/Seohee-Kim/uni-secom/assets/62201733/00c7bf42-e80b-4e94-85fb-cac29e95e485) </br>

* 기존 데이터에 적용한 로지스틱 회귀의 성능이 매우 낮음에 따라, 다양한 모델 성능 향상 기법을 적용하여 SMOTE 적용 전인 랜덤 포레스트 모델을 최종 선택하였음</br>
* 해당 모델에서 중요 변수 상위 30개를 추출하여, 중요 변수로 구성된 데이터에 GAN을 적용, 모든 모델에 적합</br>
* 최종 모델 : 랜덤 포레스트, Light GBM (93.63%)</br>

<!--
</br></br>
## 피드백

</br></br>
## 참고한 레퍼런스  
-->
