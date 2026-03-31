from src.data_loader import load_data


def test_load_data():
    data = load_data("data/russian_credit_data.csv")
    assert data.shape[0] > 0
    assert "target" in data.columns