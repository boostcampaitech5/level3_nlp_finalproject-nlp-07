import { keyframes } from "styled-components";

export const start2 = keyframes`
0%{
  transform: translate(0,20%);
  opacity: 0;
}
100%{
  transform: translate(0%,0);
  opacity: 1;
}
`;

export const startEnd = keyframes`
0%{
  transform: translate(0,80%);
  opacity: 0;
}
20%{
  transform: translate(0,0%);
  opacity: 1;
}
80%{
  transform: translate(0,0%);
  opacity: 1;
}
100%{
  transform: translate(0%,80%);
  opacity: 0;
}
`;

export const start = keyframes`
0%{
  transform: translate(0,80%);
  opacity: 0;
}
100%{
  transform: translate(0%,0);
  opacity: 1;
}
`;

export const end = keyframes`
0%{
  transform: translate(0,0);
  opacity: 1;
}
100%{
  transform: translate(0%,80%);
  opacity: 0;
}
`;

export const opacity_change = keyframes`
  0%{
    opacity: 0;
  }
  100%{
    opacity: 1;
  }
`;

export const opacity_end = keyframes`
0%{
  opacity: 1;
}
100%{
  opacity: 0;
}
`;
