"""Kumo-small: NVIDIA Kumo-Tabular (modelo tabular fundacional, aprendizaje en contexto), adaptado con 1,791 registros
municipales del SIAP y ajustado 10 pasos en cada uno de los 5 pliegues canónicos. Aquí solo se cargan los 5 modelos ya
ajustados y se predice: el contexto de cada modelo son las parcelas de entrenamiento de su pliegue con su rendimiento.
Lectura: mediana de los 999 cuantiles, recortada en 0; resultado: media de los 5 pliegues.

En GPU (CUDA) se usa la misma precisión mixta fp16 que en el entrenamiento, y el resultado coincide con el original.
En CPU se usa fp32 y el generador aleatorio de la CPU: Kumo difiere hasta ~0.02 t/ha y la predicción final hasta
~0.005 t/ha."""
import sys

import numpy as np
import pandas as pd

from .datos import RAIZ

sys.path.insert(0, str(RAIZ / "third_party"))
SEMILLA_PLAN, ESTIMADORES = 20261013, 8


def _tabla(sdm, d, y, entrenamiento, dispositivo):
    vals = np.full(len(d["ids"]), np.nan, dtype=np.float32)
    vals[entrenamiento] = y
    x, cats = d["features"], d["categorical_features"]
    num = [f"n{i:04d}" for i in range(x.shape[1])]
    cat = [f"c{i:04d}" for i in range(cats.shape[1])]
    frame = pd.DataFrame(x, columns=num)
    stypes = {n: "numerical" for n in num}
    for i, n in enumerate(cat):
        frame[n] = cats[:, i].astype(str)
        stypes[n] = "categorical"
    frame["yield_t_ha"] = vals
    stypes["yield_t_ha"] = "numerical"
    return sdm.TableTensor.from_pandas(frame, stypes=stypes, device=dispositivo)


def predecir(y, ids_pred, dispositivo="auto"):
    import json

    import torch
    import sdm

    if dispositivo == "auto":
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dispositivo)
    torch.set_num_threads(2)
    with np.load(RAIZ / "data/kumo_entradas.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    ids = list(d["ids"].astype(str))
    pos = {i: n for n, i in enumerate(ids)}
    pliegues = json.loads((RAIZ / "modelos/pliegues_canonicos.json").read_text())
    query = np.array([pos[i] for i in ids_pred])
    preds = []
    for f in pliegues:
        k = f["pliegue"]
        tr = np.sort(np.array([pos[i] for i in f["ids_entrenamiento"]]))
        tabla = _tabla(sdm, d, y.loc[[ids[i] for i in tr]].to_numpy(np.float32), tr, dev)
        # Cada modelo está guardado en dos partes (GitHub no acepta archivos de más de 100 MB).
        pesos = {}
        for parte in (1, 2):
            pesos.update(torch.load(RAIZ / f"modelos/kumo/pliegue_{k}_parte{parte}.pt", weights_only=True,
                                    map_location="cpu")["state_dict"])
        modelo = sdm.models.KumoTabular(task="regression", size="small", pretrained=False, device=dev)
        modelo.load_state_dict(pesos, strict=True)
        modelo.eval()
        semilla = SEMILLA_PLAN + 100 * k
        torch.manual_seed(semilla)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(semilla)
        # El orden de consulta es el del entrenamiento original: todas las parcelas fuera del entrenamiento del pliegue.
        q_all = np.setdiff1d(np.arange(len(ids)), tr)
        c = tabla[torch.as_tensor(tr, device=dev)]
        q = tabla[torch.as_tensor(q_all, device=dev)]
        amp = torch.amp.autocast("cuda", dtype=torch.float16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
        with torch.inference_mode(), amp:
            out = modelo(x_context=c.drop_columns("yield_t_ha"), y_context=c["yield_t_ha"],
                         x_query=q.drop_columns("yield_t_ha"), num_estimators=ESTIMADORES, estimator_batch_size=1,
                         generator=torch.Generator(device=dev).manual_seed(semilla))
        cuantiles = out.numerical.detach().cpu().numpy().astype(np.float64)
        med = np.maximum(0, cuantiles[:, 499])
        qpos = {qi: n for n, qi in enumerate(q_all)}
        preds.append(np.array([med[qpos[i]] for i in query]))
        del modelo, tabla
    return pd.Series(np.mean(preds, axis=0), index=list(ids_pred))
