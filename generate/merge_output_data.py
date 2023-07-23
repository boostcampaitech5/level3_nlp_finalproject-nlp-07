import json

# 미리 생성한 데이터셋 경로
DATASET_PATH = "/opt/ml/input/data/v3.5/summary_v3.5_2000.json"
# 생성한 요약문이 들어있는 파일
SUMMARY_PATH = "/opt/ml/input/data/v3.5/prompt_2000_0.json"

print("Summary file:", SUMMARY_PATH)
print("Dataset file:", DATASET_PATH)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data_list = json.load(f)
    
with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    result_list = json.load(f)
    
# 생성한 요약문을 데이터셋에 넣기
for i in range(len(result_list)):
    data_list[i]["summary"] = result_list[i]["text"]
    
with open(DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False)