import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

from utils.prompter import Prompter
from utils.preprocess import clean_text


def inference(
    model,
    tokenizer,
    dataset: list,
    prompt_template_path="../templates/summary_v1.0_infer.json",
):
    prompt = Prompter(prompt_template_path)
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    def ask(input_text):
        start = time.time()
        ans = pipe(
            input_text,
            do_sample=True,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
            eos_token_id=2,
        )
        end = time.time()
        time_span = end - start
        result = ans[0]["generated_text"]
        return result, time_span

    # test_data = [{id, review, summary, keywords}]

    preds = []
    times = []

    # 컬럼명
    REVIEW = "review"

    for data in tqdm(dataset, desc="Inferencing", total=len(dataset)):
        clean_review = clean_text(data[REVIEW])
        input_text = prompt.generate_prompt(review=clean_review)
        generated_text, time_span = ask(input_text)
        generated_text = clean_text(generated_text, remove_tag=False)

        preds.append(generated_text)
        times.append(time_span)

    return preds, times


if __name__ == "__main__":
    test_data = [
        {
            "id": 1,
            "review": "가격도 싸고 월드콘 구구콘 두 가지 콘 아이스크림 먹을 수 있어 재구매 하였습니다. 맛도 달콤하고 예전부터 있던 상품으로 변함없는 맛이 좋습니다. 구입 기준 제조일자는 3주 전입니다. 두 가지 맛을 동시에 가격은 더 낮게 구입할 수 있어 만족하며 달콤하고 씹는 식감도 좋아 가족 모두가 잘 먹었습니다. 구입 기준 제조일자는 2주 전입니다. 맛은 짱 잘못 사면 콘이 누그러져서 맛없는데 와 이건 완전 바삭바삭 다 먹고 재구매해야겠네요. 최근에 만든 거 보내주셨네요. 오랜만에 아이스크림 주문했어요. 월드콘은 항상 맛있고 구구콘은 달콤한 게 댕길 때 더 맛남 평상시엔 좀 달게 느껴지는데  맛있게 잘 먹을게요. 부드럽고 달콤한 맛 뽀도독 씹히는 견과류의 고소한 맛은 옛날 어릴 적 추억여행을 하게 해줍니다. 언제 먹어도 맛난 아이스크림 진짜 맛있어요 더운 여름에 없어서는 안될 아이스크림콘도 바삭바삭 너무 많이네요  몇 번째 여기서만 주문해서 콘이 바삭바삭 너무 맛있어요. 여름이라 집에 항상 쟁여놓고 있네요. 전에 한번 구매해서 맛 좋아서 재구매합니다. 땡땡 잘 얼어서 배달됐네요. 콘도 바삭바삭 녹지 않고 맛있어요. 아이가 갑자기 아이스크림을 찾길래 주문해 봤어요. 하나도 안 녹고 진짜 잘 도착했어요. 월드콘이 줄었어요. 같이 온 구구는 그대로인데 월드콘 양이 많이 줄었네요. 얼었다 녹은 것도 아니고 월드콘만 다 저 크기예요 마트보다 쿠팡이 쌀 때가 많은데 수량이 많아 못 샀어요. 이번엔 반반씩 나와서 바로 샀어요. 여름엔 다양하게 사두는 편인데 쿠팡은 가격만 좋고 구성은 마트가 더 좋아요 큼직한 드라이아이스 3개로 밀봉포장되옴. 다 녹아오지 않을까 괜한 걱정이었음. 프레시 인정 많이 녹아 있었어요. 그래도 월드콘과 구구콘은 맛있지만요 오랫동안 먹어왔던 콘 아이스크림 집에서 받아서 먹으니 더 맛있다. 월드콘이랑 헷갈렸었나  맛있음. 근데 좀 녹아서 와서 까면 윗부분에 초코가 뚜껑에 붙어서 떼짐. 구구콘과 월드콘의 조합 좋아유 정말 좋아요. 만족도 백퍼입니다 10시 되기 전 주문하면 오후에 받을 수 있어서 좋았어요",
            "summary": "<가격> 싸고 <맛> 달콤하고 변함없는 맛 <맛> 두 가지 맛을 동시에 <가격> 더 낮게 <맛> 달콤하고 <식감> 씹는 식감도 좋아 <맛>완전 바삭바삭 부드럽고 달콤한 맛 뽀도독 씹히는 견과류의 고소한 맛 옛날 어릴 적 추억여행을 하게 해줍니다 <배달> 땡땡 잘 얼어서 <양> 월드콘 양이 많이 줄었네요 수량이 많아 이번엔 반반씩 나와서 <맛> 집에서 받아서 먹으니 더 맛있다",
        }
    ]

    MODEL = "boostcamp-5th-nlp07/koalpaca-polyglot-5.8b-summary-v1.0"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device=f"cuda", non_blocking=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    preds, times = inference(model, tokenizer, test_data, prompt_template_path="v1.0")
    print(preds)
    print(times)
