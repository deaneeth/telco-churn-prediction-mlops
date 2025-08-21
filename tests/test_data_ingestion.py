from src.data_ingestion import DataIngestion
import pandas as pd

def test_load_csv(tmp_path):
    p = tmp_path / "small.csv"
    df = pd.DataFrame({"a":[1,2,3]})
    df.to_csv(p, index=False)
    di = DataIngestion(str(p))
    loaded = di.load_csv()
    assert list(loaded.columns) == ["a"]
    assert loaded.shape == (3,1)
