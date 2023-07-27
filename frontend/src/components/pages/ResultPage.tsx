import { HeaderLogoBlock } from "@blocks/HeaderLogo";
import { useEffect, useState } from "react";
import styled, { css } from "styled-components";
import check from "@assets/images/check.png";
import * as KF from "@styles/keyframes";
import axios from "axios";
import Spinner from "@blocks/Spinner";
import EmptyHeart from "@assets/images/empty_heart.png";
import FillHeart from "@assets/images/fill_heart.png";
import TextSlider from "@blocks/TextSlider";

const isMobile = function () {
  const match = window.matchMedia("(pointer:coarse)");
  return match && match.matches;
};

type DPRReviewType = {
  review: [
    { prod_id: string; context: string; prod_name: string },
    { prod_id: string; context: string; prod_name: string },
    { prod_id: string; context: string; prod_name: string }
  ];
};

type ProductType = {
  product_id: string;
  prod_name: string;
  price: string;
  url: string;
  summary: string;
  product_img_url: string;
};

interface LoadingType {
  isdone: number;
}

const ResultPage = () => {
  const [db_loaded, setDBLoaded] = useState(false);
  const [retrieve_loaded, setRetrieveLoaded] = useState(false);
  const [summary_loaded, setSummaryLoaded] = useState(false);
  const [prodIDList, setIDList] = useState<number[]>([0, 0, 0]);
  const [isLoaded, setLoaded] = useState(false);
  const [showResult, setResult] = useState(false);
  const [summaryList, setSummaryList] = useState<string[]>(["", "", ""]);
  const [imgURLs, setURLs] = useState<string[]>(["", "", ""]);
  const [imgLinks, setLinks] = useState<string[]>(["", "", ""]);
  const [prodNames, setProdNames] = useState<string[]>(["", "", ""]);
  const [isTextFeedbackModalOn, setTextFeedbackModal] = useState(false);
  const [errorModalOn, setErrorModalOn] = useState(false);
  const [FirstHeart, setFirstHeart] = useState(false);
  const [SecondHeart, setSecondHeart] = useState(false);
  const [ThirdHeart, setThirdHeart] = useState(false);
  const [isProdSelected, setSelected] = useState(false);
  const heartPushed = [FirstHeart, SecondHeart, ThirdHeart];
  const heartPushHandler = [setFirstHeart, setSecondHeart, setThirdHeart];
  const [summaryResponse, setResponse] = useState<ProductType[] | null>(null);
  const [dataSource, setSource] = useState<string>("crawl");
  const [feedbackID, setFeedbackID] = useState<number | null>(32);
  const [isDescModalOn, setDescModal] = useState(true);
  const [inputs, setInput] = useState("");

  useEffect(() => {
    const FetchData = async () => {
      await axios({
        method: "get",
        url: "/api/reviews/search/prod_name/" + localStorage.getItem("product"),
      })
        .then((response) => {
          setSource(response.data.source);
          setDBLoaded(true);
          if (response.data.source === "crawl") {
            axios({
              method: "post",
              url: process.env.REACT_APP_DPR_ENDPOINT + "/dpr/split_v3",
              headers: {
                Accept: "*/*",
                "Content-Type": "application/json",
              },
              data: {
                query: localStorage.getItem("query"),
                reviews: response.data.reviews,
              },
            })
              .then(async (dprResponse) => {
                setRetrieveLoaded(true);
                const dprList: DPRReviewType = dprResponse.data;
                setIDList([
                  parseInt(dprList.review[0].prod_id),
                  parseInt(dprList.review[1].prod_id),
                  parseInt(dprList.review[2].prod_id),
                ]);

                setProdNames([
                  dprList.review[0].prod_name,
                  dprList.review[1].prod_name,
                  dprList.review[2].prod_name,
                ]);

                await axios({
                  method: "post",
                  url: process.env.REACT_APP_SUMMARY_ENDPOINT + "/summary_v3",
                  headers: {
                    Accept: "*/*",
                    "Content-Type": "application/json",
                  },
                  data: [
                    dprList.review[0].context,
                    dprList.review[1].context,
                    dprList.review[2].context,
                  ],
                })
                  .then(async (summary_response) => {
                    setSummaryList(summary_response.data);
                    await axios({
                      method: "post",
                      url: "/api/products/url",
                      headers: {
                        Accept: "*/*",
                        "Content-Type": "application/json",
                      },
                      data: {
                        product_id: [
                          parseInt(dprList.review[0].prod_id),
                          parseInt(dprList.review[1].prod_id),
                          parseInt(dprList.review[2].prod_id),
                        ],
                      },
                    })
                      .then((response) => {
                        setURLs([
                          response.data.prod_id1.product_img_url,
                          response.data.prod_id2.product_img_url,
                          response.data.prod_id3.product_img_url,
                        ]);
                        setLinks([
                          response.data.prod_id1.url,
                          response.data.prod_id2.url,
                          response.data.prod_id3.url,
                        ]);
                        setSummaryLoaded(true);
                        setTimeout(() => setLoaded(true), 2000);
                        setTimeout(() => setResult(true), 6200);
                      })
                      .catch((error) => {
                        console.log(error);
                        setErrorModalOn(true);
                      });
                  })
                  .catch((error) => {
                    console.log(error);
                    setErrorModalOn(true);
                  });
              })
              .catch((error) => {
                console.log(error);
                setErrorModalOn(true);
              });
          } else {
            // db에서 가져옴
            axios({
              method: "post",
              url: process.env.REACT_APP_DPR_ENDPOINT + "/dpr/concat_v3",
              headers: {
                Accept: "*/*",
                "Content-Type": "application/json",
              },
              data: {
                query: localStorage.getItem("query"),
                products: response.data.products,
              },
            })
              .then((response) => {
                setResponse(response.data.product);
                setRetrieveLoaded(true);
                setSummaryLoaded(true);
                setTimeout(() => setLoaded(true), 2000);
                setTimeout(() => setResult(true), 6200);
              })
              .catch((error) => {
                console.log(error);
                setErrorModalOn(true);
              });
          }
        })
        .catch((error) => {
          console.log(error);
          setErrorModalOn(true);
        });
    };
    localStorage.getItem("product") && FetchData();

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const HeartHandler = async (idx: number) => {
    if (!isProdSelected && !isDescModalOn) {
      if (dataSource === "db") {
        heartPushHandler[idx](true);
        setSelected(true);
        setTextFeedbackModal(true);
        const curData = {
          query: localStorage.getItem("query"),
          recommendations: JSON.stringify([
            parseInt(summaryResponse![0].product_id),
            parseInt(summaryResponse![1].product_id),
            parseInt(summaryResponse![2].product_id),
          ]),
          best: summaryResponse![idx].product_id.toString(),
          review: null,
        };
        await axios({
          method: "post",
          url: "/api/feedback",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          data: curData,
        })
          .then((response) => {
            setFeedbackID(response.data.feedback_id);
          })
          .catch((error) => {
            console.log(error);
          });
      } else {
        heartPushHandler[idx](true);
        setSelected(true);
        setTextFeedbackModal(true);
        const curData = {
          query: localStorage.getItem("query"),
          recommendations: JSON.stringify(prodIDList),
          best: prodIDList[idx].toString(),
          review: null,
        };
        await axios({
          method: "post",
          url: "/api/feedback",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
          data: curData,
        })
          .then((response) => {
            setFeedbackID(response.data.feedback_id);
          })
          .catch((error) => {
            console.log(error);
          });
      }
    }
  };

  const ChangeHandler = (e: {
    target: { value: React.SetStateAction<string> };
  }) => {
    setInput(e.target.value);
  };

  const SubmitHandler = async () => {
    if (inputs.length === 0) {
      setTextFeedbackModal(false);
    } else {
      await axios({
        method: "put",
        url: "/api/feedback/feedback_id/" + feedbackID!.toString(),
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        data: {
          review: inputs,
        },
      }).catch((error) => {
        console.log(error);
      });
      setTextFeedbackModal(false);
    }
  };

  const TextToBullet = (input: string): string[] => {
    const result: string[] = [];
    let startIndex = input.indexOf("<");
    let endIndex = input.indexOf(">");

    while (startIndex !== -1 && endIndex !== -1) {
      const prefix = input.substring(0, startIndex);
      const suffix = input.substring(endIndex + 1);
      result.push(prefix.trim());
      input = suffix;
      startIndex = input.indexOf("<");
      endIndex = input.indexOf(">");
    }

    if (input.trim().length > 0) {
      result.push(input.trim());
    }

    return result;
  };

  const RemoveFirstWord = (input: string): string => {
    const words = input.split(" ");
    words.shift(); // 첫 번째 단어를 제거
    return words.join(" ");
  };

  const AddCommasToNumber = (amount: string): string => {
    return amount.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  };

  if (errorModalOn) {
    return (
      <>
        <HeaderLogoBlock />
        <ErrorModal>
          <ErrorHeightBox />
          <DescText>
            이런 🥲 에러가 발생했어요
            <br />
            <br />
            <DescSmallText>
              다른 상품 혹은 조건으로 입력해주세요.
              <br />
              <br />
            </DescSmallText>
            <DescSmallText>
              만약 에러가 계속 발생한다면, <br />
              관리자에게 문의해주세요🙏
            </DescSmallText>
          </DescText>
        </ErrorModal>
      </>
    );
  }
  if (!showResult && !isLoaded) {
    return (
      <>
        <HeaderLogoBlock />
        <CenterWrapper>
          <LodingDiv isdone={summary_loaded ? 1 : 0}>
            <LodingText>
              {!db_loaded
                ? "리뷰 데이터를 수집하고 있어요"
                : !retrieve_loaded
                ? "리뷰 데이터를 분석하고 있어요"
                : dataSource === "crawl"
                ? "리뷰 데이터를 요약하고 있어요"
                : "리뷰 데이터를 분석하고 있어요"}
            </LodingText>
            <TextSlider />
            <Spinner />
          </LodingDiv>
        </CenterWrapper>
      </>
    );
  }
  if (!showResult && isLoaded) {
    return (
      <>
        <HeaderLogoBlock />
        <CenterWrapper>
          <CenterText>조건에 맞는 상품을 찾았어요!</CenterText>
          <CheckImg src={check} alt="" />
        </CenterWrapper>
      </>
    );
  }

  return (
    <MainWrapper>
      <HeaderLogoBlock />
      <SummaryWrapper>
        {dataSource === "db"
          ? summaryResponse!.map((product, index) => (
              <SummaryDiv key={index}>
                <ItemDiv>
                  <ProductImg
                    src={product.product_img_url}
                    alt=""
                    onClick={() => window.open(product.url)}
                  />
                  <ProductTitleText>
                    {RemoveFirstWord(product.prod_name)}
                  </ProductTitleText>
                  <PriceText>{AddCommasToNumber(product.price)}원</PriceText>
                  <FeedBackDiv>
                    <HeartImg
                      src={heartPushed[index] ? FillHeart : EmptyHeart}
                      alt=""
                      onClick={() => HeartHandler(index)}
                    />
                  </FeedBackDiv>
                </ItemDiv>
                <TextDiv>
                  <Description>
                    {TextToBullet(product.summary.slice(1, -1)).map(
                      (text) =>
                        text.length >= 5 &&
                        text.length <= 40 && <SummaryLine>{text}</SummaryLine>
                    )}
                  </Description>
                </TextDiv>
              </SummaryDiv>
            ))
          : [0, 1, 2].map((idx) => (
              <SummaryDiv key={idx}>
                <ItemDiv>
                  <ProductImg
                    src={imgURLs[idx]}
                    alt=""
                    onClick={() => window.open(imgLinks[idx])}
                  />
                  <ProductTitleText>
                    {RemoveFirstWord(prodNames[idx])}
                  </ProductTitleText>
                  <FeedBackDiv>
                    <HeartImg
                      src={heartPushed[idx] ? FillHeart : EmptyHeart}
                      alt=""
                      onClick={() => HeartHandler(idx)}
                    />
                  </FeedBackDiv>
                </ItemDiv>
                <TextDiv>
                  <Description>
                    {TextToBullet(summaryList[idx]).map(
                      (text) =>
                        text.length >= 5 &&
                        text.length <= 40 && <SummaryLine>{text}</SummaryLine>
                    )}
                  </Description>
                </TextDiv>
              </SummaryDiv>
            ))}
      </SummaryWrapper>
      {isDescModalOn && (
        <DescModal>
          <HeightBox />
          <DescText>
            추천 받은 상품들 중 <br />
            가장 마음에 드는 상품을 골라주세요! <br />
            <DescSmallText>
              (상품 카드를 클릭하면 상품 페이지로 이동합니다)
            </DescSmallText>
            <br />
            <br />
            <DescLightText>
              한 번만 고를 수 있으니 <br />
              신중히 골라주세요 ☺️
            </DescLightText>
            <br />
          </DescText>
          <DescButton onClick={() => setDescModal(false)}>
            확인했어요
          </DescButton>
        </DescModal>
      )}
      {isTextFeedbackModalOn && (
        <ReviewModal>
          <FeedbackHeightBox />
          <DescText>
            리뷰에 참여해주셔서 감사해요! 😙
            <br />
            <br />
            <ReviewLightText>
              개선할 점이 있다면 편하게 말씀해주세요 😊
            </ReviewLightText>
          </DescText>
          <ReviewInputBox
            placeholder="이런 점을 보완하면 좋을 것 같아요"
            onChange={ChangeHandler}
          />
          <DescButton onClick={SubmitHandler}>
            {inputs.length === 0 ? "건너뛰기" : "제출하기"}
          </DescButton>
        </ReviewModal>
      )}
    </MainWrapper>
  );
};

export default ResultPage;

const CenterWrapper = styled.div`
  margin: 0 auto;
  margin-top: 230rem;
  width: 700rem;
  height: 274rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: space-between;
  ${isMobile() && "width: 100%; height: 170rem;"}
`;

const CenterText = styled.span`
  color: #4a4a4a;
  text-align: center;
  font-size: 36rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 700;
  line-height: normal;

  ${css`
    animation: ${KF.startEnd} 4s 0s 1 both;
  `}

  ${isMobile() && "font-size: 24rem;"}
`;

const CheckImg = styled.img`
  width: 100rem;
  height: 100rem;
  margin: 0 auto;
  margin-top: 110rem;

  animation: ${KF.startEnd} 4s 0.2s 1 both;

  ${isMobile() && "width: 80rem; height: 80rem; margin-top: 30rem;"}
`;

const ItemDiv = styled.div`
  margin: 0 auto;
  margin-top: 90rem;
  margin-bottom: 50rem;
  width: 417rem;
  height: 570rem;
  border-radius: 20rem;
  background: #fff;
  box-shadow: 0rem 0rem 17rem 0rem rgba(0, 0, 0, 0.25);

  display: flex;
  flex-direction: column;
  align-items: center;
  cursor: pointer;

  ${css`
    animation: ${KF.start2} 0.8s 0.2s 1 both;
  `}
  ${isMobile() &&
  "width: 300rem; min-height: 400rem; margin-top: 30rem; height: 460rem; margin-bottom: 30rem;"}
`;

const Description = styled.span`
  margin: 0 auto;
  color: #4a4a4a;
  text-align: left;
  font-size: 28rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 500;
  line-height: 150%;
  width: 200rem;
  word-break: keep-all;

  ${isMobile() && "font-size: 20rem;"}
  ${css`
    animation: ${KF.start} 0.8s 0.4s 1 both;
  `};
`;

const LodingText = styled.span`
  color: #4a4a4a;
  text-align: center;
  font-size: 36rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  ${isMobile() && "font-size: 24rem;"}
`;

const LodingDiv = styled.div<LoadingType>`
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: center;
  height: 200rem;
  ${css`
    animation: ${KF.start} 0.8s 0.6s 1 both;
  `}

  ${(props) =>
    props.isdone === 1 &&
    css`
      animation: ${KF.end} 0.8s 1s 1 both;
    `}
    ${isMobile() && "height: 180rem;"}
`;

const TextDiv = styled.div`
  width: 417rem;
  margin: 0 auto;
  margin-bottom: 30rem;
  ${isMobile() && "width: 300rem;"}
`;

const SummaryWrapper = styled.div`
  display: flex;
  width: 80vw;
  justify-content: center;
  @media screen and (max-width: 1400px) {
    align-items: center;
    flex-direction: column;
  }
`;

const SummaryDiv = styled.div`
  display: flex;
  flex-direction: column;
  margin: 0 50rem;
`;

const MainWrapper = styled.div`
  display: flex;
  flex-direction: column;
  width: 100%;
  align-items: center;
`;

const ProductImg = styled.img`
  width: 350rem;
  height: 350rem;
  margin-top: 30rem;
  border-radius: 20rem;
  ${isMobile() && "width: 270rem; height: 270rem; margin-top: 20rem;"}
`;

const ProductTitleText = styled.div`
  height: 70rem;
  font-size: 24rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  width: 90%;
  margin-top: 30rem;
  word-break: keep-all;
  display: flex;
  align-items: center;
  justify-content: center;
  ${isMobile() && "font-size: 18rem; margin-top: 20rem;"}
`;

const FeedBackDiv = styled.div`
  height: 10rem;
  display: flex;
  justify-content: flex-end;
  width: 90%;
  margin-top: 10rem;
  ${isMobile() && "margin-top: 5rem;"}
`;

const HeartImg = styled.img`
  width: 30rem;
  height: 30rem;
`;

const DescModal = styled.div`
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);

  width: 400rem;
  height: 400rem;
  background: white;

  // border: 5rem solid #f4f4f4;
  border-radius: 20rem;
  box-shadow: 0rem 0rem 17rem 0rem rgba(0, 0, 0, 0.25);

  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-direction: column;

  ${css`
    animation: ${KF.opacity_change} 0.8s 0s 1 both;
  `};
  ${isMobile() && "width: 300rem; height:300rem;"}
`;

const DescText = styled.span`
  font-size: 24rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 600;
  line-height: normal;
  width: 90%;
  word-break: keep-all;
  ${css`
    animation: ${KF.opacity_change} 0.8s 0s 1 both;
  `};
  ${isMobile() && "font-size: 18rem;"}
`;

const DescSmallText = styled.span`
  font-weight: 500;
  font-size: 18rem;
  ${isMobile() && "font-size: 14rem;"}
`;

const DescLightText = styled.span`
  font-weight: 500;
`;

const DescButton = styled.div`
  width: 200rem;
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
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 30rem;

  ${isMobile() &&
  "font-size: 18rem; width: 150rem; height: 45rem; margin-bottom: 20rem;"}
`;

const HeightBox = styled.div`
  height: 50rem;
  ${isMobile() && "height: 55rem;"}
`;

const ReviewModal = styled.div`
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);

  width: 400rem;
  height: 400rem;
  background: white;

  border-radius: 20rem;
  box-shadow: 0rem 0rem 17rem 0rem rgba(0, 0, 0, 0.25);

  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;

  ${css`
    animation: ${KF.opacity_change} 0.8s 0s 1 both;
    // animation: ${KF.opacity_end} 0.8s 2s 1 both;
  `};
  ${isMobile() && "width: 300rem; height:300rem;"}
`;

const ReviewInputBox = styled.textarea`
  margin: 30rem auto;
  padding: 10rem;
  width: 300rem;
  // height: 80rem;
  border-radius: 20rem;
  background: #fff;
  box-shadow: 0rem 0rem 17rem 0rem rgba(0, 0, 0, 0.25);
  border: 0.5rem solid;
  text-align: center;
  font-size: 20rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 400;
  line-height: normal;
  &:focus {
    outline: none;
  }
  word-break: keep-all;
  white-space: wrap;
  resize: none;
  ${isMobile() && "width:80%; font-size: 16rem; margin: 20rem auto;"}
`;

const ReviewLightText = styled.span`
  font-weight: 500;
  font-size: 20rem;
  ${isMobile() && "font-size: 14rem;"}
`;

const FeedbackHeightBox = styled.div`
  height: 100rem;
  ${isMobile() && "height: 50rem;"}
`;

const ErrorModal = styled.div`
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);

  width: 400rem;
  height: 400rem;
  background: white;

  // border: 5rem solid #f4f4f4;
  border-radius: 20rem;
  box-shadow: 0rem 0rem 17rem 0rem rgba(0, 0, 0, 0.25);

  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;

  ${css`
    animation: ${KF.opacity_change} 0.8s 0s 1 both;
  `};
  ${isMobile() && "width: 300rem; height:300rem;"}
`;

const ErrorHeightBox = styled.div`
  height: 30rem;
  ${isMobile() && "height: 55rem;"}
`;

const SummaryLine = styled.li`
  list-style-position: inside;
  text-indent: -30rem;
  margin-left: 40rem;
  font-size: 24rem;
  ${isMobile() && "font-size: 18rem; text-indent: -20rem; margin-left:25rem;"}
`;

const PriceText = styled.div`
  width: 90%;
  text-align: right;
  font-size: 24rem;
  font-family: Pretendard;
  font-style: normal;
  font-weight: 500;
  line-height: normal;
  margin-top: 10rem;
  ${isMobile() && "font-size: 18rem;;"}
`;
