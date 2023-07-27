import React, { useEffect, useState } from "react";
import styled from "styled-components";

const isMobile = function () {
  const match = window.matchMedia("(pointer:coarse)");
  return match && match.matches;
};

const TextSlider: React.FC = () => {
  const textList = [
    "DB에 데이터가 있다면 10초 이내로 결과를 볼 수 있어요",
    "결과페이지에서 가장 마음에 드는 상품을 선택해주세요",
    "여러분들의 리뷰로 더 나은 서비스로 개선될 거예요",
    "조금만 기다려주세요",
    "DB에 데이터가 없으면 실시간으로 크롤링해요",
    "크롤링 진행 시 총 1분 가량 소요됩니다",
    "서버 4대를 동원한 프로젝트입니다",
    "크롤링, DPR, 요약, 텍스트 필터링 등이 사용되었어요",
    "인프라 짜느라 너무 힘들었습니다,,",
    "안심하세요 에러가 난 것이 아닙니다",
    "다만 조금 오래 걸릴 뿐입니다",
    "에러가 발생하면 에러 페이지로 이동합니다",
    "가끔 사용자가 몰리면 더 늦어지기도 합니다",
    "실시간으로 수집, 분석, 요약이 진행되니 양해부탁드려요🙏",
    "이 쯤 되면 결과가 나올 법도 한데요",
    "2분 이상 같은 현상이 지속되면 처음부터 진행해주세요🥲",
  ];
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    // 5초에 한 번씩 currentIndex를 업데이트하여 텍스트를 변경합니다.
    const interval = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % textList.length);
    }, 7000);

    // 컴포넌트가 언마운트될 때 interval을 정리합니다.
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div>
      <WaitText>{textList[currentIndex]}</WaitText>
    </div>
  );
};

export default TextSlider;

const WaitText = styled.span`
  color: #4a4a4a;
  text-align: center;
  font-size: 24rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 500;
  line-height: normal;
  ${isMobile() && "font-size: 15rem;"}
`;
