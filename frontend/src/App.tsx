import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./App.css";
import StartPage from "@pages/StartPage";
import SearchPage from "@pages/SearchPage";
import ResultPage from "@pages/ResultPage";
import { RecoilRoot } from "recoil";

function App() {
  return (
    <RecoilRoot>
      <div className="App">
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<StartPage />}></Route>
            <Route path="/search" element={<SearchPage />}></Route>
            <Route path="/result" element={<ResultPage />}></Route>
          </Routes>
        </BrowserRouter>
      </div>
    </RecoilRoot>
  );
}

export default App;
