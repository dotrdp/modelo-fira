"""XGBoost preentrenado con estadística municipal del SIAP y clima de AgERA5, ajustado a las 138 parcelas: media de 20
semillas ya entrenadas."""
import numpy as np
import pandas as pd
import xgboost as xgb

from .datos import RAIZ


def predecir(ids_pred):
    X = pd.read_csv(RAIZ / "data/xgb_entradas_59.csv", index_col="id").astype("float32").loc[list(ids_pred)]
    archivos = sorted((RAIZ / "modelos/xgb").glob("semilla_*.json"))
    assert len(archivos) == 20, "faltan modelos de XGBoost"
    m = xgb.DMatrix(X)
    return pd.Series(np.mean([xgb.Booster(model_file=str(a)).predict(m) for a in archivos], axis=0), index=X.index)
