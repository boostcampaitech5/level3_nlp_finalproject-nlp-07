import { atom } from "recoil";

export const userState = atom({
  key: "userInput",
  default: { production: "", query: "" },
});

export const recomState = atom({
  key: "recommendations",
  default: [0, 0, 0, 0, 0],
});
