from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class ScalerFactory:
    @staticmethod
    def get_scaler(name: str = "standard"):
        if name == "standard":
            return StandardScaler()
        if name == "minmax":
            return MinMaxScaler()
        if name == "robust":
            return RobustScaler()
        raise ValueError("Unknown scaler")
