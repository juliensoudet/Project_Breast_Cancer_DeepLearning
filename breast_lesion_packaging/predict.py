def test(result):
    benign_prob = round(result[0], 2)
    malignant_prob = round(result[1], 2)

    if benign_prob >= 0.7:
        return f'The predicted probability of the lesion being benign is {benign_prob}'
    elif malignant_prob >= 0.7:
        return f'The predicted probability of the lesion being malignant is {malignant_prob}'
    else:
        return f'Uncertainty warning! The predicted probability of the lesion being benign is only: {benign_prob}, and being malignant: {malignant_prob}'
