from typing import List
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import TransformerMixin
try:
    import category_encoders as ce
except Exception:
    ce = None

class EncoderFactory:
    @staticmethod
    def get_encoder(strategy: str = "onehot", handle_unknown: str = "ignore"):
        if strategy == "onehot":
            return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
        if strategy == "ordinal":
            return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        if strategy == "target":
            if ce is None:
                raise ImportError("category_encoders not installed")
            return ce.TargetEncoder()
        raise ValueError(f"Unknown encoding strategy: {strategy}")
