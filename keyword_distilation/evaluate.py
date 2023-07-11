from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, ndcg_score
import numpy as np

def compute_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def compute_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def compute_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)

def compute_map_score(y_true, y_pred):
    return average_precision_score(y_true, y_pred)

def compute_ndcg_score(true_scores, predicted_scores):
    return ndcg_score(true_scores, predicted_scores)

def compute_mrr_score(y_true, y_pred):
    scores = 0
    for t, p in zip(y_true, y_pred):
        if p in t:
            scores += 1 / (t.index(p) + 1)
    return scores / len(y_true)


if __name__ == "__main__":

    # 사용 예시
    y_true_binary = [1, 1, 0, 1, 0, 0, 1, 0, 0, 1]
    y_pred_binary = [0, 1, 1, 1, 1, 0, 1, 0, 0, 0]

    print(f"Precision: {compute_precision(y_true_binary, y_pred_binary)}")
    print(f"Recall: {compute_recall(y_true_binary, y_pred_binary)}")
    print(f"F1 Score: {compute_f1_score(y_true_binary, y_pred_binary)}")
    print(f"MAP: {compute_map_score(y_true_binary, y_pred_binary)}")

    true_scores_rank = np.array([[3, 2, 3, 0, 1, 2]])
    predicted_scores_rank = np.array([[3, 2, 1, 0, 1, 2]])
    print(f"NDCG: {compute_ndcg_score(true_scores_rank, predicted_scores_rank)}")

    y_true_rank = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    y_pred_rank = [2, 1, 3]
    print(f"MRR: {compute_mrr_score(y_true_rank, y_pred_rank)}")
