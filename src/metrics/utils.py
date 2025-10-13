import editdistance


def calc_cer(target_text: str, predicted_text: str) -> float:
    target_len = len(target_text)
    predicted_len = len(predicted_text)

    if target_len == 0 and predicted_len == 0:
        return 0.0
    elif target_len == 0:
        return float("inf")
    elif predicted_text == "":
        return 1.0

    return editdistance.eval(target_text, predicted_text) / target_len


def calc_wer(target_text: str, predicted_text: str) -> float:
    target_words = target_text.split()
    predicted_words = predicted_text.split()
    
    predicted_len = len(predicted_words)
    target_len = len(target_words)

    if target_len == 0 and predicted_len == 0:
        return 0.0
    elif target_len == 0:
        return float("inf")
    elif predicted_len == 0:
        return 1.0

    return editdistance.eval(target_words, predicted_words) / target_len
