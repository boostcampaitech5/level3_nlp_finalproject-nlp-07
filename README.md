# **부스트캠프 AI Tech 5기 07조 Final Project**


[CHOSEN](http://chosen.o-r.kr/)

<img width="800" alt="Untitled" src="./utils/img/overview.png">

<br/><br/>

# 👋 반갑습니다.

안녕하세요. 저희 `연어보다자연어` 팀은 2023년 6월 30일 ~ 7월 28일까지  **사용자의 입력에 따른 리뷰 기반 상품 추천 서비스** 를 진행했습니다. 쿠팡의 리뷰데이터를 수집하여 **AI 모델로 요약문을 생성**했습니다. 요약문을 query 로, 리뷰데이터를 context 로 사용하여 **retrieve 모델을 학습**시켰습니다. 이것을 토대로 사용자 입력에 맞는 상품을 추천해주는 리뷰 기반 AI 서비스를 개발했습니다.

<br/>

## 팀원 소개

| <img src="https://avatars.githubusercontent.com/u/81620001?v=4" width = 120> | <img src="https://avatars.githubusercontent.com/u/86578246?v=4" width=120> | <img src="https://avatars.githubusercontent.com/u/126573689?v=4" width=120> | <img src="https://avatars.githubusercontent.com/u/96599427?v=4" width=120> | <img src="https://avatars.githubusercontent.com/u/89494907?v=4" width=120> |
| --- | --- | --- | --- | --- |
| 권지은_T5018<br>[@lectura7942](https://github.com/lectura7942) | 김재연_T5051<br>[@JLake310](https://github.com/JLake310) | 박영준_T5087<br>[@hoooolllly](https://github.com/hoooolllly) | 정다혜_T5189<br>[@Da-Hye-JUNG](https://github.com/Da-Hye-JUNG) | 최윤진_T5218<br>[@yunjinchoidev](https://github.com/yunjinchoidev) |

<br/>


## 팀원 역할

- `권지은`: 데이터 필터링, 요약 모델 학습 데이터 제작, 요약 모델 학습 및 평가, 요약 API 제작
- `김재연`: Project Manager, DPR 학습 및 평가, 데이터 필터링, 프론트엔드 개발, 배포 및 유지/보수
- `박영준`: 데이터 필터링, 요약 모델 학습 데이터 제작
- `정다혜`: 요약 모델 학습 데이터 제작, 요약 모델 학습 및 평가
- `최윤진`: 데이터 수집, Airflow 스케줄링, DB API

<br/><br/>



# 🙌 프로젝트를 소개합니다.

## 문제 정의

세상엔 상품이 너무 많습니다.🤔  동시에 고객이 원하는 **조건**들은 점점 다양해지고 있습니다. 

많은 사람들이 자신에게 맞는 상품들을 고르기 위해 리뷰를 찾아보면서 제품들을 비교하며 고르고 있습니다. 이 과정에서 많은 시간이 허비되곤 합니다.

**저희 팀은 이런 불편함을 확인하고, 해결책으로써 리뷰 데이터에 주목했습니다.** 🔎 상품의 생생한 사용 후기와 느낌이 반영되어 있는 리뷰 데이터는 가치있는 데이터라고 할 수 있습니다. 

**딥러닝 모델을 통해 리뷰를 분석하고 사용자의 입력 조건에 맞는 상품을 추천해준다면 쇼핑에 불필요하게 낭비되는 시간을 절약**할 수 있을 거라는 생각을 하게 되었습니다. 특히 **상품들이 많고 리뷰 수, 사용자가 많은 ‘쿠팡’**이 저희 프로젝트에 가장 적합하다고 생각했습니다. 상품의 종류가 너무 많아 식품 데이터에 한정했습니다.

<br/>

## 우리는 이런 목표를 가지고 있어요.

✅ 사용자가 원하는 조건의 상품을 **한 눈에 모아볼 수 있도록** 하는 거예요.

✅ 사용자가 상품을 비교하고 선택하는 **시간을 줄일 수 있게** 도와주는 거예요.

✅ 피드백을 반영하여 추천 만족도를 **지속적으로** 높이도록 하는 거예요.

<img width="437" alt="Untitled" src="./utils/img/team_vision.png">

<br/><br/>

# ✈️ 먼저 서비스가 어떻게 돌아가는 지 보여드릴게요.

### 📌 한번 사용해보세요. 👉 [CHOSEN](http://chosen.o-r.kr/)

### 📌 바쁘신 분들을 위해 **데모 영상**을 첨부합니다.

- 검색한 상품이 DB 에 저장 된 경우
  
    [![Video Label](http://img.youtube.com/vi/dzFuc4AK6OQ/0.jpg)](https://youtu.be/dzFuc4AK6OQ)
    
    
- 검색한 상품이 저장되지 않아 실시간 크롤링 하는 경우
  
    [![Video Label](http://img.youtube.com/vi/nbzfXA41-X8/0.jpg)](https://youtu.be/nbzfXA41-X8)

    

### 📌 데이터베이스에 검색한 상품이 있는가 없는가에 따라 서비스 플로우가 달라집니다.

- 데이터베이스에 저장된 상품의 경우 10초 안에 결과를 볼 수 있습니다.
- 저장되지 않은 경우엔 실시간으로 데이터를 크롤링을 해서 1분 30초 ~2분 정도 시간이 소요됩니다.


<img width="1626" alt="Flow Chart long" src="https://github.com/hoooolllly/readmetest/assets/126573689/0b972d43-c724-411b-9604-e34e27d2fadb">

<br/><br/>

# ⏰ 저희는 이렇게 프로젝트를 진행했습니다.


이 글을 보는 여러분들께 저희 팀이 **어떤 문제를, 어떻게 해결해나갔는지** 말씀 드리고 싶습니다. 
**네 가지 관점**에서 저희의 이야기를 들려드리겠습니다. 

1️⃣ 어떻게 하면 **양질의 데이터**를 수집할 수 있을까?

2️⃣ 어떻게 하면 문서를 **키워드 중심으로 요약**할 수 있을까?

3️⃣ 어떻게 하면 비정형화된 **사용자의 피드백 데이터**를 다룰 수 있을까?

4️⃣ 어떻게 하면 서비스의 **결과를 보는 시간**을 줄일 수 있을까?




<br/><br/>

# ☝️ Wrap Up


## 프로젝트 타임라인

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e0d116e2-f488-41b8-8197-808778d4397f/Untitled.png)

## 기술 스택

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7ea00a33-012d-4d6b-ae8d-53105e530aa1/Untitled.png)

## 프로젝트 플로우 차트

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3904ae61-8f80-4fc5-89bc-5123867f4ef7/Untitled.png)

## 서비스 아키텍처

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9b7636e4-a5fd-420a-af3f-bec482d0aa01/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/187bdd88-cc44-435e-a3c1-b07b1fd40ff9/Untitled.png)

## ERD

아래 ERD 대로 MySQL 테이블을 만들고 수집한 데이터를 저장해줬습니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/20ca4fbd-2762-4536-b605-5fef674a6c53/Untitled.png)

## 실제 데이터

아래와 같이 구성되어 있답니다. 

📌**상품 데이터**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/649e4323-a134-4a2a-b326-2910b6ea9a10/Untitled.png)

📌 **리뷰 데이터**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a138819f-dc2d-4266-a2e5-e3af50f02554/Untitled.png)

## 자체 평가 의견

### 프로젝트 달성도 및 완성도

- 실시간으로 크롤링하여 결과를 보여주는 부분까지 구현하고 배포하여, 프로젝트 초기에 수립한 목표는 모두 달성했다고 생각한다.
- 외부 데이터셋을 사용하지 않았기 때문에 프로젝트 그 자체로 완전함을 갖추었다고 생각한다.

### 프로젝트를 진행하면서 배운 점 혹은 성장한 점

- 짧은 기간 동안 타이트하게 일정을 조율하는 경험을 할 수 있었다.
- 서비스를 개발하고 알파테스트를 진행하여 피드백을 받았다. 이러한 피드백을 서비스에 반영하고 개선하며 사용자의 입장에서 한 번 더 생각해볼 수 있었다.
- 데이터 수집부터 배포까지 End2End로 AI 서비스를 만들어내면서, 실제 개발 과정에 대한 이해를 기를 수 있었다.
- 해결하고자 하는 문제에 맞는 평가지표를 선택하는 경험을 할 수 있었다.

### 프로젝트를 진행하면서 아쉬웠던 점

- 짧은 기간 동안 프로젝트의 배포까지 완수해야 하다보니, 모델링에 온전히 집중하지 못한 점이 아쉬웠다.
- OpenAI API로 제작한 데이터에 대한 검수가 부족한 것 같아서 아쉬웠다.
- 할루시네이션과 생성 결과 길이에 대한 평가 지표가 부족한 것 같아서 아쉬웠다.
- 피드백 데이터가 생각보다 안 모여서 제대로 활용할 수 없던 점이 아쉬웠다.

### 추후 개발하고 싶은 부분

- ColBERT로 retrieve를 시도해보고 싶다.
- API의 동기 요청을 통해 백그라운드에서 데이터를 수집하여 사용자의 대기 시간을 더 줄여보고 싶다.
- 크롤링 시간을 단축 시키고 싶다.

<br/><br/>

# 🙂 감사합니다.

지금까지 `연어보다자연어` 팀의 여정을 읽어주셔서 감사합니다.

<br/><br/>
해당 프로젝트는 네이버 커넥트재단의 부스트캠프 AI tech 5기에서 진행했습니다.
