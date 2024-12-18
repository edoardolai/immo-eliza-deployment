import torch


def predict_price(input, model, numeric_cols, cat_cols, device):
    numeric_data = torch.tensor(input[numeric_cols].values, dtype=torch.float32).to(
        device
    )
    categorical_data = torch.tensor(input[cat_cols].values, dtype=torch.long).to(device)

    dist_data = categorical_data[:, cat_cols.index("district_id")]
    prop_data = categorical_data[:, cat_cols.index("property_sub_type_id")]

    with torch.no_grad():
        prediction = model(numeric_data, categorical_data, dist_data, prop_data)

    predicted_price = prediction.cpu().numpy()[0]
    return predicted_price
