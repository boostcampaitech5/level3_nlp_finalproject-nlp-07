import styled from "styled-components";
import logo_img from "@assets/images/logo.png";
import { useNavigate } from "react-router-dom";

export const HeaderLogoBlock = () => {
  const navigate = useNavigate();
  return (
    <LogoDiv onClick={() => navigate("/")}>
      <Logo src={logo_img} alt="" />
    </LogoDiv>
  );
};

const LogoDiv = styled.div`
  width: 100%;
  height: 90rem;
  cursor: pointer;
`;
const Logo = styled.img`
  margin: 20rem; auto;
  height: 50rem;
`;
