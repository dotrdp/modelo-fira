"""Proceso gaussiano espacial (kriging universal) con hiperparámetros ya ajustados.

    y = X beta + u_municipio + u_estado + f(xy) + e
    K = tm^2 [mismo municipio] + ts^2 [mismo estado] + sf^2 exp(-d/ell) + sn^2 I

beta se estima por mínimos cuadrados generalizados con las parcelas de entrenamiento de cada modelo. La predicción
para las 59 parcelas es la media de los 5 modelos de pliegue (media: intercepto y log-área) más la corrección de las
columnas N2 y N3, calculada con el modelo de las 138 parcelas.
"""
import json

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, solve_triangular

from .datos import RAIZ, parcelas


class Kriging:
    def __init__(self):
        p = parcelas()
        self.ids = p.index.to_numpy(str)
        self.pos = {k: i for i, k in enumerate(self.ids)}
        x, y = p.easting_m.to_numpy(), p.northing_m.to_numpy()
        self.D = np.hypot(np.subtract.outer(x, x), np.subtract.outer(y, y))
        muni, edo = p.municipality.to_numpy(str), p.state.to_numpy(str)
        self.SM = (muni[:, None] == muni[None]).astype(float)
        self.SS = (edo[:, None] == edo[None]).astype(float)
        self.logA = np.log(p.area_ha.to_numpy())
        self.Z = pd.read_csv(RAIZ / "data/gp_columnas_media.csv", index_col="id").reindex(self.ids).to_numpy(float)
        self.cfg = json.loads((RAIZ / "modelos/gp.json").read_text())
        self.pliegues = json.loads((RAIZ / "modelos/pliegues_canonicos.json").read_text())

    def _diseno(self, idx, tr, Z):
        cols = [np.ones(len(idx)), (self.logA[idx] - self.logA[tr].mean()) / self.logA[tr].std()]
        if Z is not None:
            m, s = Z[tr].mean(0), Z[tr].std(0)
            s[s == 0] = 1
            cols += list(((Z[idx] - m) / s).T)
        return np.column_stack(cols)

    def _predecir(self, h, tr, te, y, Z=None):
        B = lambda M, a, b: M[np.ix_(a, b)]  # noqa: E731
        k = lambda a, b: (h["tm"] ** 2 * B(self.SM, a, b) + h["ts"] ** 2 * B(self.SS, a, b)  # noqa: E731
                          + h["sf"] ** 2 * np.exp(-B(self.D, a, b) / h["ell"]))
        L = np.linalg.cholesky(k(tr, tr) + h["sn"] ** 2 * np.eye(len(tr)))
        Xtr, Xte = self._diseno(tr, tr, Z), self._diseno(te, tr, Z)
        beta = np.linalg.lstsq(solve_triangular(L, Xtr, lower=True), solve_triangular(L, y, lower=True), rcond=None)[0]
        return Xte @ beta + cho_solve((L, True), k(te, tr).T).T @ (y - Xtr @ beta)

    def predecir(self, y, ids_pred):
        """y: rendimientos de las 138 parcelas (Serie indexada por id). Devuelve una Serie para ids_pred."""
        te = np.array([self.pos[i] for i in ids_pred])
        folds = []
        for f, h in zip(self.pliegues, self.cfg["pliegues"]):
            assert f["pliegue"] == h["pliegue"]
            tr = np.array([self.pos[i] for i in f["ids_entrenamiento"]])
            folds.append(self._predecir(h, tr, te, y.loc[f["ids_entrenamiento"]].to_numpy()))
        tr = np.array([self.pos[i] for i in y.index])
        h = {k: self.cfg[k] for k in ("tm", "ts", "sf", "ell", "sn")}
        delta = self._predecir(h, tr, te, y.to_numpy(), self.Z) - self._predecir(h, tr, te, y.to_numpy())
        return pd.Series(np.mean(folds, axis=0) + delta, index=list(ids_pred))
