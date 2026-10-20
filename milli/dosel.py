"""Modelo de dosel: corrección por atención (mezcla de expertos) sobre una base MMST-ViT + Daily DeepCropNet.

Para cada una de las 3 semillas de preentrenamiento y cada uno de los 5 pliegues externos se incluyen las cabezas ya
entrenadas (3 cabezas base y la cabeza de corrección elegida) y las salidas congeladas de los codificadores (latentes de
las 197 parcelas). La predicción es la media de los 5 pliegues y de las 3 semillas."""
import numpy as np
import pandas as pd
import torch

from .cabezas import fusion_legacy_attention as base
from .cabezas import thermal_compact_attention as correccion
from .datos import RAIZ

SEMILLAS_BASE = (20261012, 20261013, 20261014)


def _pliegue(carpeta):
    with np.load(carpeta / "latentes.npz", allow_pickle=False) as z:
        lat = {k: z[k] for k in z.files}
    viejos = {"original_mmst": lat["original_mmst"], "original_mmst_prediction": lat["original_mmst_prediction"],
              "daily": lat["daily"], "daily_prediction": lat["daily_prediction"]}
    cargar = lambda p: torch.load(p, weights_only=True, map_location="cpu")  # noqa: E731
    ancla = np.mean([base.predict_head(cargar(carpeta / f"base_{s}.pt"), viejos) for s in SEMILLAS_BASE], axis=0)
    valores = {"municipal_mmst": lat["municipal_mmst"], "daily": lat["daily_extendido"],
               "daily_delta": lat["daily_extendido"] - lat["daily"], "baseline_anchor": ancla[:, None]}
    cp = cargar(carpeta / "correccion.pt")
    pred = np.mean([correccion.predict_head(m, valores) for m in cp["members"]], axis=0)
    return pd.Series(pred, index=lat["ids"].astype(str))


def predecir(ids_pred):
    semillas = sorted((RAIZ / "modelos/dosel").glob("semilla_*"))
    assert len(semillas) == 3, "faltan modelos del dosel"
    por_semilla = [pd.concat([_pliegue(c) for c in sorted(s.glob("pliegue_*"))], axis=1).mean(axis=1)
                   for s in semillas]
    return pd.concat(por_semilla, axis=1).mean(axis=1).loc[list(ids_pred)]
