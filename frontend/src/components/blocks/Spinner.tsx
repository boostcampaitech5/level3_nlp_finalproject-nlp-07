import React from "react";
import "./Spinner.css";

const Spinner = () => {
  return (
    <div className="sk-chase" style={{ marginTop: "20rem" }}>
      <div className="sk-chase-dot"></div>
      <div className="sk-chase-dot"></div>
      <div className="sk-chase-dot"></div>
      <div className="sk-chase-dot"></div>
      <div className="sk-chase-dot"></div>
      <div className="sk-chase-dot"></div>
    </div>
    // <div className="spinner">
    //   <div className="bounce1"></div>
    //   <div className="bounce2"></div>
    //   <div className="bounce3"></div>
    // </div>
  );
};

export default Spinner;
