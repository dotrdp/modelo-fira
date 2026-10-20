"""Lectura de los archivos del Reto AgroCebada 2026 que necesita la inferencia: rendimientos y cultivo de cada parcela."""
import struct
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RENDIMIENTOS = "ID_area_rendimiento_70_30_Reto_AgroCebada.csv"
PARCELAS_DBF = "Parcelas_Reto_AGC_CONJUNTO.dbf"


def buscar(carpeta, nombre):
    """Ruta de `nombre` dentro de la carpeta del reto (busca en subcarpetas)."""
    hallados = sorted(Path(carpeta).rglob(nombre))
    if not hallados:
        raise FileNotFoundError(f"No se encontró {nombre} dentro de {carpeta}")
    return hallados[0]


def rendimientos(carpeta):
    """Serie id -> rendimiento (t/ha) de las 138 parcelas de ENTRENAMIENTO.

    Los valores se redondean a float32, como en el entrenamiento original de los modelos."""
    df = pd.read_csv(buscar(carpeta, RENDIMIENTOS), encoding="utf-8-sig").set_index("ID_POLIGONO")
    y = df.RENDIMIENTO_T_HA.dropna().astype(np.float32).astype(float)
    if len(y) != 138:
        raise ValueError(f"Se esperaban 138 rendimientos y hay {len(y)}")
    return y.sort_index()


def leer_dbf(ruta):
    """Lector mínimo de .dbf (dBase III) sin dependencias externas: lista de dicts con texto."""
    with open(ruta, "rb") as f:
        cab = f.read(32)
        n = struct.unpack("<I", cab[4:8])[0]
        largo_cab, largo_reg = struct.unpack("<HH", cab[8:12])
        campos = []
        while True:
            d = f.read(32)
            if d[0] == 0x0D:
                break
            campos.append((d[:11].split(b"\0")[0].decode("latin1"), d[16]))
        f.seek(largo_cab)
        filas = []
        for _ in range(n):
            r = f.read(largo_reg)
            pos, fila = 1, {}
            for nombre, largo in campos:
                fila[nombre] = r[pos:pos + largo].decode("utf-8", errors="replace").strip()
                pos += largo
            filas.append(fila)
    return filas


def cultivo(carpeta):
    """Serie id -> valor del campo 'Cultivo' del shapefile de parcelas ('Cebada' o 'CEBADA')."""
    filas = leer_dbf(buscar(carpeta, PARCELAS_DBF))
    return pd.Series({f["ID_POLIGON"]: f["Cultivo"] for f in filas}).sort_index()


def parcelas():
    """Tabla de las 197 parcelas incluida en el repositorio (municipio, estado, coordenadas UTM 14N y área)."""
    return pd.read_csv(RAIZ / "data/parcelas.csv").set_index("id").sort_index()
