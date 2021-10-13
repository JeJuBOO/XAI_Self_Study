# XAI_Self_Study
* eXpainable Artificial intelligence
* XAI의 종류와 코딩을 정리
---
1. 설명가능한 AI

 대부분 알고있는 딥러닝 모델은 입력에 대한 결과를 예측하고 분류할 뿐 딥러닝의 매우 복잡한 과정속에서 결과를 설명하는 것은 매우 어렵다.
그러나 결과를 설명하는 것은 매우 중요하다. 
딥러닝을 통하여 어떠한 정보를 바탕으로 결과를 도출해내는지 모른다면 사용하려하는 딥러닝 모델의 신뢰성이 현저히 떨어진다. 
또한 분류할 이미지가 적대적 공격(Adversarial Attack)에 의해 우리의 눈으로 구분할 수 없는 장치를 더해 분류에 혼란을 줄 수 있다.
따라서 어떠한 방법에 의해 딥러닝을 속일 수 있다는 사실은 이제 공부할 XAI의 필요성을 더욱 보여준다.

---
2. XAI 기법의 종류
 
 -Intergrated gradients  
 baseline에서 input까지 모든 gradient를 모두 활용하며 주된 픽셀을 찾는다.
 