import { HeaderLogoBlock } from "@blocks/HeaderLogo";
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import styled, { css } from "styled-components";
import * as KF from "@styles/keyframes";
import { useRecoilState } from "recoil";
import { userState } from "atoms/userState";

const isMobile = function () {
  const match = window.matchMedia("(pointer:coarse)");
  return match && match.matches;
};

const SearchPage = () => {
  const [inputs, setInput] = useState("");
  const [stage, setStage] = useState("name");
  const [end, setEnd] = useState(0);
  const [userProd, setProd] = useState("");
  const [, setUserInput] = useRecoilState(userState);

  const navigate = useNavigate();
  const ButtonHandler = () => {
    if (inputs.length !== 0) {
      if (stage === "name") {
        setInput("");
        setStage("option");
        setProd(inputs);
        localStorage.setItem("product", inputs);
      } else {
        const userInput = {
          production: userProd,
          query: inputs,
        };
        localStorage.setItem("query", inputs);
        setUserInput(userInput);
        setEnd(1);
        setTimeout(() => navigate("/result"), 1600);
      }
    }
  };

  const EnterHandler = (e: { key: string }) => {
    if (e.key === "Enter") {
      ButtonHandler();
    }
  };

  const ChangeHandler = (e: {
    target: { value: React.SetStateAction<string> };
  }) => {
    setInput(e.target.value);
  };

  return (
    <>
      <HeaderLogoBlock />
      <CenterWrapper>
        <CenterText isend={end}>
          {stage === "name"
            ? "찾고자 하는 상품을 알려주세요."
            : "원하는 조건을 알려주세요."}
        </CenterText>
        <CenterInput
          isend={end}
          onChange={ChangeHandler}
          value={inputs}
          onKeyPress={EnterHandler}
          placeholder={
            stage === "name"
              ? "ex. 떡볶이, 삼겹살, 토마토소스"
              : "ex. 맵고 양 많은 것"
          }
        />
        <NextButton
          isend={end}
          onClick={ButtonHandler}
          inputlength={inputs.length}
        >
          {stage === "name" ? "다음" : "찾아보기"}
        </NextButton>
      </CenterWrapper>
    </>
  );
};

export default SearchPage;

interface endType {
  isend: number;
}

const CenterWrapper = styled.div`
  margin: 0 auto;
  margin-top: 230rem;
  height: 440rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  ${isMobile() && "margin-top: 120rem;"}
`;

const CenterText = styled.span<endType>`
  color: #4a4a4a;
  text-align: center;
  font-size: 36rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 700;
  line-height: normal;

  ${isMobile() && "font-size: 24rem;"}
  ${css`
    animation: ${KF.start} 0.8s 0.2s 1 both;
  `}
  ${(props) =>
    props.isend === 1 &&
    css`
      animation: ${KF.end} 0.8s 0.8s 1 both;
    `}
`;

const CenterInput = styled.input<endType>`
  height: 150rem;
  border-radius: 20rem;
  width: 700rem;
  background: #fff;
  box-shadow: 0rem 0rem 17rem 0rem rgba(0, 0, 0, 0.25);
  border: 0rem;
  text-align: center;
  color: #000;
  text-align: center;
  font-size: 28rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 400;
  line-height: normal;
  word-break: keep-all;

  &:focus {
    outline: none;
  }

  ${isMobile() && "width: 300rem; height: 120rem; font-size: 24rem;"}
  ${css`
    animation: ${KF.start} 0.8s 0.4s 1 both;
  `}
  ${(props) =>
    props.isend === 1 &&
    css`
      animation: ${KF.end} 0.8s 0.4s 1 both;
    `}
`;

interface buttonType {
  isend: number;
  inputlength: number;
}

const NextButton = styled.div<buttonType>`
  width: 200rem;
  height: 70rem;
  flex-shrink: 0;
  border-radius: 15rem;
  background: ${(props) => (props.inputlength === 0 ? "#eee" : "#4b81bf")};
  color: ${(props) => (props.inputlength === 0 ? "#ffffff" : "#ffffff")};
  text-align: center;
  font-size: 24rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  ${(props) => props.inputlength !== 0 && "cursor: pointer"};
  display: flex;
  align-items: center;
  justify-content: center;

  ${isMobile() && "font-size: 20rem; width: 160rem; height: 60rem;"}
  ${css`
    animation: ${KF.start} 1s 0.8s 1 both;
  `}
  ${(props) =>
    props.isend === 1 &&
    css`
      animation: ${KF.end} 0.8s 0s 1 both;
    `}
`;
