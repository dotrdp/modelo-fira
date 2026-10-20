#!/usr/bin/env python3
"""Predicción del rendimiento de cebada (t/ha) de las 59 parcelas del conjunto PREDICCION del Reto AgroCebada 2026.

    python predecir.py --reto <carpeta del dataset del reto> [--salida predicciones.csv] [--dispositivo auto|cuda|cpu]
    python predecir.py --reto <carpeta> --verificar      # compara con las predicciones de referencia del proyecto

Modelo (pesos fijos):
    E = (GP + Kumo + Dosel) / 3        prediccion = (0.65 E + 0.10 XGB) / 0.75
y luego la regla R1: en parcelas cuyo campo Cultivo es 'Cebada' (fuente que reporta en múltiplos de 0.5 t/ha), la
predicción se redondea al múltiplo de 0.5 más cercano si está a menos de 0.10 t/ha de él.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from milli import datos, dosel, kumo, xgb_miembro  # noqa: E402
from milli.gp import Kriging  # noqa: E402

W_E, W_X = 0.65, 0.10


def regla_r1(p, cultivo):
    g = 0.5 * np.round(p / 0.5)
    snap = (cultivo.reindex(p.index) == "Cebada") & ((p - g).abs() <= 0.10)
    return p.where(~snap, g), int(snap.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reto", required=True, type=Path,
                    help="carpeta con DATASET_RETO_AGRO_2026 (rendimientos .csv y shapefile de parcelas)")
    ap.add_argument("--salida", type=Path, default=Path("predicciones.csv"))
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cuda", "cpu"], help="para Kumo-small")
    ap.add_argument("--verificar", action="store_true", help="compara con referencia/miembros_59.csv")
    a = ap.parse_args()

    y = datos.rendimientos(a.reto)
    cultivo = datos.cultivo(a.reto)
    p = datos.parcelas()
    ids = [i for i in p.index if i not in y.index]
    assert len(ids) == 59 and set(cultivo.index) == set(p.index), "las parcelas del reto no coinciden con data/parcelas.csv"

    print("proceso gaussiano...", flush=True)
    gp = Kriging().predecir(y, ids)
    print("Kumo-small...", flush=True)
    ku = kumo.predecir(y, ids, a.dispositivo)
    print("modelo de dosel...", flush=True)
    do = dosel.predecir(ids)
    print("XGBoost...", flush=True)
    xg = xgb_miembro.predecir(ids)

    e = (gp + ku + do) / 3
    bruta = (W_E * e + W_X * xg) / (W_E + W_X)
    final, n = regla_r1(bruta, cultivo)
    out = pd.DataFrame({"municipio": p.municipality.loc[ids], "prediccion_t_ha": final.round(4),
                        "prediccion_sin_r1": bruta.round(4), "gp": gp.round(4), "kumo": ku.round(4),
                        "dosel": do.round(4), "xgb": xg.round(4)}).rename_axis("id")
    out.to_csv(a.salida)
    print(f"{len(out)} predicciones en {a.salida} (R1 redondeó {n}); media {final.mean():.2f} t/ha, "
          f"rango {final.min():.2f}-{final.max():.2f}")

    if a.verificar:
        ref = pd.read_csv(Path(__file__).resolve().parent / "referencia/miembros_59.csv", index_col="id").loc[ids]
        for nombre, serie, col in (("gp", gp, "gp"), ("kumo", ku, "kumo"), ("dosel", do, "dosel"), ("xgb", xg, "xgb"),
                                   ("final sin R1", bruta, "prediccion_sin_r1"), ("final", final, "prediccion")):
            print(f"  {nombre:13s} max |diferencia| = {float((serie - ref[col]).abs().max()):.2e}")


if __name__ == "__main__":
    main()
