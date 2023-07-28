import { HeaderLogoBlock } from "@blocks/HeaderLogo";
import React from "react";
import styled from "styled-components";
import arrow from "@assets/images/arrow.png";
import check from "@assets/images/check.png";
import { useNavigate } from "react-router-dom";

const isMobile = function () {
  const match = window.matchMedia("(pointer:coarse)");
  return match && match.matches;
};

const StartPage = () => {
  const navigate = useNavigate();
  if (isMobile()) {
    return (
      <>
        <HeaderLogoBlock />
        <MobileHeaderText>
          번거롭게 찾아볼 필요 없게, <br />
          조건에 맞는 상품을 한 번에 찾아드릴게요!
        </MobileHeaderText>
        <MobileCenterDiv>
          <MobileCenterWrapper>
            <MobileCenterInnerDiv>
              <MobileLeftInnerDiv>
                <MobileLeftInnerText>
                  상품 : 떡볶이 <br />
                  조건 : 양 많고 맵고 식감이 쫄깃한 것
                </MobileLeftInnerText>
              </MobileLeftInnerDiv>
              <MobileArrow src={arrow} alt="" />
              <MobileRightInnerDiv>
                <MobileCheckImg src={check} alt="" />
                <MobileRightText>조건에 맞는 상품을 찾았어요!</MobileRightText>
              </MobileRightInnerDiv>
            </MobileCenterInnerDiv>
            <MobileNextButton onClick={() => navigate("/search")}>
              시작하기
            </MobileNextButton>
          </MobileCenterWrapper>
        </MobileCenterDiv>
      </>
    );
  }
  return (
    <>
      <HeaderLogoBlock />
      <HeaderText>
        번거롭게 찾아볼 필요 없게, <br />
        조건에 맞는 상품을 한 번에 찾아드릴게요!
      </HeaderText>
      <CenterDiv>
        <CenterWrapper>
          <CenterInnerDiv>
            <LeftDiv>
              <LeftInnerDiv>
                <LeftInnerText>
                  상품 : 떡볶이 <br />
                  조건 : 양 많고 맵고 식감이 쫄깃한 것
                </LeftInnerText>
              </LeftInnerDiv>
            </LeftDiv>
            <Arrow src={arrow} alt="" />
            <RightDiv>
              <RightInnerDiv>
                <CheckImg src={check} alt="" />
                <RightText>조건에 맞는 상품을 찾았어요!</RightText>
              </RightInnerDiv>
            </RightDiv>
          </CenterInnerDiv>
          <NextButton onClick={() => navigate("/search")}>시작하기</NextButton>
        </CenterWrapper>
      </CenterDiv>
    </>
  );
};

export default StartPage;

const HeaderText = styled.div`
  // width: 800rem;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #000;
  font-size: 32rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 700;
  line-height: 200%;
  margin: 0 auto;
  margin-top: 20rem;
  margin-bottom: 40rem;
`;

const CenterDiv = styled.div`
  width: 100%;
  height: 64.35vh;
  background: var(--sub-sky, #d3eaff);
  display: flex;
  align-items: center;
`;

const CenterWrapper = styled.div`
  width: 100%;
  height: 555rem;
  // height: 51.38vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
`;

const CenterInnerDiv = styled.div`
  margin: 0 auto;
  // margin-bottom: 85rem;
  height: 400rem;
  width: 90%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  @media screen and (max-width: 1400px) {
    flex-direction: column;
  }
`;

const LeftDiv = styled.div`
  width: 610rem;
  height: 140rem;
  border-radius: 10rem;
  background: #fff;
  padding: 20rem;
  @media screen and (max-width: 1400px) {
    padding: 0rem;
    height: 130rem;
    margin-bottom: 20rem;
  }
`;

const LeftInnerDiv = styled.div`
  width: 600rem;
  height: 130rem;
  border-radius: 10rem;
  border: 5rem solid var(--light-gray, #eee);
  background: #fff;
  margin: 0 auto;
  display: flex;
  align-items: center;
`;

const LeftInnerText = styled.span`
  color: #4a4a4a;
  font-size: 24rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 500;
  line-height: 150%;
  margin: 0 auto;
  text-align: center;
`;

const Arrow = styled.img`
  width: 64rem;
  @media screen and (max-width: 1400px) {
    transform: rotate(90deg);
    width: 48rem;
  }
`;

const RightDiv = styled.div`
  width: 650rem;
  height: 400rem;
  border-radius: 10rem;
  background: #fff;
  display: flex;
  align-items: center;
  @media screen and (max-width: 1400px) {
    width: 600rem;
    height: 180rem;
    margin-top: 20rem;
  }
`;

const RightInnerDiv = styled.div`
  width: 600rem;
  height: 350rem;
  border-radius: 10rem;
  border: 5rem solid var(--main-blue, #4b81bf);
  background: #fff;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  @media screen and (max-width: 1400px) {
    height: 180rem;
  }
`;

const CheckImg = styled.img`
  width: 80rem;
  margin-top: 85rem;
  margin-bottom: 60rem;
  @media screen and (max-width: 1400px) {
    margin-top: 25rem;
    margin-bottom: 20rem;
  }
`;

const RightText = styled.span`
  color: #4a4a4a;
  font-size: 24rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 600;
  line-height: normal;
`;

const NextButton = styled.div`
  width: 320rem;
  height: 60rem;
  border-radius: 15rem;
  border: 5rem solid var(--main-blue, #4b81bf);
  background: var(--main-blue, #4b81bf);
  color: #fff;
  text-align: center;
  font-size: 26rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  cursor: pointer;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
`;

// Mobile

const MobileHeaderText = styled.span`
  display: flex;
  align-items: center;
  justify-content: center;
  color: #000;
  font-size: 20rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 700;
  line-height: 200%;
  margin: 0 auto;
  margin-top: 20rem;
  margin-bottom: 40rem;
`;

const MobileCenterDiv = styled.div`
  width: 100%;
  height: 57vh;
  background: var(--sub-sky, #d3eaff);
  display: flex;
  align-items: center;
`;

const MobileCenterInnerDiv = styled.div`
  margin: 0 auto;
  width: 100%;
  height: 35vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
`;

const MobileLeftInnerDiv = styled.div`
  width: 80%;
  height: 90rem;
  border-radius: 10rem;
  border: 5rem solid var(--light-gray, #eee);
  background: #fff;
  margin: 0 auto;
  display: flex;
  align-items: center;
`;

const MobileLeftInnerText = styled.span`
  color: #4a4a4a;
  font-size: 18rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 500;
  line-height: 150%;
  margin-left: 20rem;
  // text-align: left;
  margin: 0 auto;
  text-align: center;
`;

const MobileArrow = styled.img`
  width: 42rem;
  transform: rotate(90deg);
`;

const MobileRightInnerDiv = styled.div`
  width: 90%;
  height: 120rem;
  border-radius: 10rem;
  border: 5rem solid var(--main-blue, #4b81bf);
  background: #fff;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const MobileCheckImg = styled.img`
  width: 50rem;
  margin-top: 15rem;
  margin-bottom: 20rem;
`;

const MobileRightText = styled.span`
  color: #4a4a4a;
  font-size: 16rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 600;
  line-height: normal;
`;

const MobileNextButton = styled.div`
  width: 70%;
  height: 50rem;
  border-radius: 15rem;
  border: 5rem solid var(--main-blue, #4b81bf);
  background: var(--main-blue, #4b81bf);
  color: #fff;
  text-align: center;
  font-size: 22rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  cursor: pointer;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const MobileCenterWrapper = styled.div`
  width: 100%;
  // height: 555rem;
  height: 47vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
`;
