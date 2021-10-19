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
  Integrated Gradients(IG)모델은 다양한 딥러닝 모델들을 설명하기 위한 모델이다. 이미지처리, 자연어처리, 정형 데이터 등 다양한 분야에 적용 가능하고 대규모 네트워크에도 적용이 가능하기 때문에 특히 인기가 있다.

 baseline에서 input까지 모든 gradient를 모두 활용하며 주된 픽셀을 찾는다.
 
![image](https://user-images.githubusercontent.com/71332005/137852768-265fd90a-39f8-4eb6-8a30-76a257dee74b.png)
 
 IG의 식은 다음과 같다   
 ![image](https://user-images.githubusercontent.com/71332005/137853298-c4133d3d-4fae-43e7-95ba-addf2be543d2.png)
 ![image](https://user-images.githubusercontent.com/71332005/137853346-f4702680-c623-47da-a1fd-fe07cd6f65ca.png)

위의 식에서 다음 식을 통해 baseline과 원본 이미지 사이에 선형 보간을 생성한다.   
![image](https://user-images.githubusercontent.com/71332005/137853523-588582fd-fbd0-4867-857f-fe2d73dbbce6.png)

선형 보간한 이미지들의 gradient를 구하여 이미지에서 픽셀의 영향을 확인할 수 있다.   

![image](https://user-images.githubusercontent.com/71332005/137855470-99c8af7a-c13c-4c7c-9993-27a1d581c0e3.png)
![image](https://user-images.githubusercontent.com/71332005/137855488-f7bed532-886b-422c-a9e6-4502f36f7dd9.png)
![image](https://user-images.githubusercontent.com/71332005/137855503-f37fbd2c-a3c5-44f9-ad4b-ae96f8eadd57.png)



