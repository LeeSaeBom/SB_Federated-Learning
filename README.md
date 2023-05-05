# SB_Federated-Learning


프로젝트 소개


- 프로젝트명 : 심전도 사용자인증과 적대적 공격


- 프로젝트 목표  

1) 최근 바이오인증 시스템의 위·변조 위험을 해결하고자 심전도 신호기반 사용자인증 기술 연구가 활발히 진행되고 있음. 그러나 딥러닝을 이용한 심전도 사용자인증 모델은 사용자의 오분류를 유도하는 적대적 공격에 매우 취약함.

2) Center node와 local node 통신이 중요한 연합학습에 적대적 공격이 취약하다는 것이 입증됨
   
   ref. Data Poisoning Attacks on Federated Machine Learning (https://ieeexplore.ieee.org/document/9618642)

3) 본 연구는 FGSM, PGD(Projected Gradient Descent), CW(Carlini&Wagner's attack) 등의 적대적 예제를 사용하여 심전도 사용자인증에서 적대적 공격이 보안에 얼마나 취약한지를 분석하고자 함


- 연구내용 및 방법

1) 심전도 딥러닝 기반 사용자인증용 데이터 전처리 및 인증 실험환경 구축
 
2) 심전도 사용자인증 모델의 적대적 공격 취약점 검증 및 분석
 
3) 심전도 사용자인증을 위한 적대적 공격기반 지능형 보안기술 연구 시뮬레이션 
 
 평가지표 : 적대적 공격 전 모델의 성능에서 공격 후 모델의 성능의 차이며, 최소 50% 이상의 공격 성공률을 목표로 함. 
