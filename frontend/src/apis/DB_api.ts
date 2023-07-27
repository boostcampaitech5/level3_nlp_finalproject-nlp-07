import axios from "axios";

// export type DBDataType = {
//   reviews: {
//     prod_id: number;
//     prod_name: string;
//     rating: string;
//     title: string;
//     context: string;
//     answer: string;
//     review_url: string | null;
//   }[];
// };
export type DBDataType = {
  reviews: string[];
};

const FindFromDB = async (
  setData: React.Dispatch<React.SetStateAction<DBDataType | null>>
) => {
  var product = localStorage.getItem("itemName");
  await axios({
    method: "get",
    url: "http://localhost:8000/reviews/search/prod_name/" + product,
  }).then((response) => {
    setData(response.data);
    // console.log(response);
  });
};

export default FindFromDB;
