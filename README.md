# Milli: rendimiento de cebada de temporal (Reto AgroCebada FIRA 2026)

Modelo ya ajustado que estima el rendimiento de cebada (t/ha) del ciclo abril–octubre de 2025 en las 59 parcelas del
conjunto PREDICCION del reto, en Puebla, Hidalgo y Tlaxcala.

```
E = (GP + Kumo + Dosel) / 3          predicción = (0.65·E + 0.10·XGB) / 0.75
```

| Componente | Qué es | Qué incluye este repositorio |
|---|---|---|
| GP | Proceso gaussiano espacial (kriging universal): efectos de municipio y estado, covarianza exponencial con la distancia, log-área y dos columnas fenológicas (N2, N3) | Hiperparámetros de los 5 modelos de pliegue y del modelo de las 138 parcelas (`modelos/gp.json`) |
| Kumo | NVIDIA Kumo-Tabular *small*, adaptado con estadística municipal del SIAP y ajustado a las parcelas | 5 modelos ajustados, uno por pliegue (`modelos/kumo/`) |
| Dosel | Corrección por atención sobre MMST-ViT + Daily DeepCropNet con series de Sentinel-2 | Cabezas entrenadas y salidas congeladas de los codificadores, 3 semillas × 5 pliegues (`modelos/dosel/`) |
| XGB | XGBoost preentrenado con estadística municipal del SIAP y clima AgERA5, ajustado a las parcelas | 20 modelos, uno por semilla (`modelos/xgb/`) |

Después se aplica la regla R1: en las parcelas cuyo campo `Cultivo` es `Cebada` (fuente que reporta en múltiplos de
0.5 t/ha), la predicción se redondea al múltiplo de 0.5 más cercano si está a menos de 0.10 t/ha de él.

En validación cruzada de 5 pliegues repetida 20 veces, el modelo obtiene RMSE 0.472 t/ha, MAE 0.323 t/ha y R² 0.690.

## Instalación

Python 3.13. La GPU (CUDA) es opcional.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

### Para la inferencia

Solo se necesita el dataset del reto, `DATASET_RETO_AGRO_2026`. FIRA lo entrega a los equipos registrados en la
plataforma del reto (FIRA, 2026a). La inferencia lee dos archivos:

- `ID_area_rendimiento_70_30_Reto_AgroCebada.csv`: rendimiento de las 138 parcelas de entrenamiento. El GP y Kumo los
  usan como contexto al predecir.
- `Parcelas_Reto_AGC_CONJUNTO.dbf`, del shapefile de parcelas: el campo `Cultivo` que usa la regla R1.

### Con los que se construyeron las entradas y se entrenó el modelo

Ya vienen procesados en `data/` y en los pesos de `modelos/`, así que no hace falta descargarlos para predecir. Para
reconstruirlos:

| Datos | Uso en el modelo | Descarga |
|---|---|---|
| Dataset del reto: parcelas, índices de Sentinel-2, Landsat y PlanetScope, precipitación CHIRPS, temperatura CHIRTS-ERA5, elevación y pendiente del CEM de INEGI (FIRA, 2026b) | Todas las entradas por parcela | Plataforma del reto |
| Sentinel-2 MSI nivel 2A (Copernicus Sentinel-2, 2021; Drusch et al., 2012) | Series del modelo de dosel; columnas N2 y N3 del GP | [ESA, colección 1 nivel 2A](https://doi.org/10.5270/S2_-znk9xsj) |
| AgERA5 (Copernicus Climate Change Service, 2020) | Clima de XGB; preentrenamiento municipal del dosel y de Kumo | [Climate Data Store](https://doi.org/10.24381/cds.6c68c9bb) |
| CHIRPS v2.0 diario, 1991–2025 (Funk et al., 2015) | Preentrenamiento municipal del dosel | [Climate Hazards Center](https://www.chc.ucsb.edu/data/chirps) |
| Producción agrícola municipal SIACON–SIAP 2003–2024 (Zepeda Villarreal et al., 2026) | Preentrenamiento de XGB (3,437 registros), de Kumo y del dosel (1,791 registros de Hidalgo, Puebla y Tlaxcala) | [figshare](https://doi.org/10.6084/m9.figshare.32065674.v3) |
| AlphaEarth Satellite Embedding V1 (Brown et al., 2025) | Variables de la adaptación municipal de Kumo | [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) |
| Pesos de Kumo-Tabular (NVIDIA, 2026) | Punto de partida de Kumo | [Hugging Face](https://huggingface.co/nvidia/Kumo-Tabular) |

## Inferencia

```bash
python predecir.py --reto /ruta/a/DATASET_RETO_AGRO_2026
```

Escribe `predicciones.csv` con una fila por parcela:

| Columna | Contenido |
|---|---|
| `id`, `municipio` | Parcela |
| `prediccion_t_ha` | Predicción final (t/ha) |
| `prediccion_sin_r1` | Predicción antes de la regla R1 |
| `gp`, `kumo`, `dosel`, `xgb` | Predicción de cada componente |

Opciones:

- `--salida archivo.csv` cambia el archivo de salida.
- `--dispositivo cuda|cpu` elige el dispositivo de Kumo. El valor por defecto, `auto`, usa la GPU si existe.
- `--verificar` compara cada componente con `referencia/miembros_59.csv`, las predicciones originales del proyecto.

Tarda menos de 10 segundos, en GPU o en CPU. En GPU el resultado coincide con la referencia (diferencias
< 1e-6 t/ha). En CPU, Kumo usa fp32 y otro generador aleatorio: difiere hasta 0.02 t/ha, y la predicción final hasta
0.005 t/ha.

El modelo cubre las 197 parcelas del reto, porque sus entradas ya están calculadas en `data/`. Para predecir otras
parcelas habría que reconstruir esas entradas a partir de los datos de la tabla anterior.

## Estructura

```
predecir.py             inferencia
milli/                  código de cada componente (gp.py, kumo.py, dosel.py, xgb_miembro.py, datos.py, cabezas/)
modelos/                pesos e hiperparámetros ya ajustados; pliegues_canonicos.json (ids de entrenamiento de cada pliegue)
data/                   entradas por parcela ya calculadas (sin rendimientos)
referencia/             predicciones originales de cada componente, para --verificar
third_party/sdm/        biblioteca de NVIDIA para ejecutar Kumo-Tabular
```

## Licencias

- `modelos/kumo/` son pesos derivados de NVIDIA Kumo-Tabular, bajo la licencia OpenMDW-1.1 (`modelos/kumo/LICENSE`).
- `third_party/sdm/` es la biblioteca *structured-data-models* de NVIDIA, bajo Apache-2.0 (`third_party/sdm/LICENSE`).

## Referencias

Brown, C. F., Kazmierski, M. R., Pasquarella, V. J., Rucklidge, W. J., Samsikova, M., Zhang, C., Shelhamer, E., Lahera,
E., Wiles, O., Ilyushchenko, S., Gorelick, N., Zhang, L. L., Alj, S., Schechter, E., Askay, S., Guinan, O., Moore, R.,
Boukouvalas, A., y Kohli, P. (2025). *AlphaEarth Foundations: An embedding field model for accurate and efficient global
mapping from sparse label data* (arXiv:2507.22291). arXiv. https://doi.org/10.48550/arXiv.2507.22291

Copernicus Climate Change Service. (2020). *Agrometeorological indicators from 1979 to present derived from reanalysis*
[Conjunto de datos]. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). https://doi.org/10.24381/cds.6c68c9bb

Copernicus Sentinel-2 (processed by ESA). (2021). *MSI Level-2A BOA reflectance product. Collection 1* [Conjunto de
datos]. European Space Agency. https://doi.org/10.5270/S2_-znk9xsj

Drusch, M., Del Bello, U., Carlier, S., Colin, O., Fernandez, V., Gascon, F., Hoersch, B., Isola, C., Laberinti, P.,
Martimort, P., Meygret, A., Spoto, F., Sy, O., Marchese, F., y Bargellini, P. (2012). Sentinel-2: ESA's optical
high-resolution mission for GMES operational services. *Remote Sensing of Environment, 120*, 25–36.
https://doi.org/10.1016/j.rse.2011.11.026

Fideicomisos Instituidos en Relación con la Agricultura [FIRA]. (2026a). *Reto AgroCebada FIRA 2026: Bases de
participación*. https://www.fira.gob.mx/Nd/Reto%20Agrocebada%20Bases.pdf

Fideicomisos Instituidos en Relación con la Agricultura [FIRA]. (2026b). *Dataset Reto AgroCebada 2026* [Conjunto de
datos entregado a los equipos participantes].

Funk, C., Peterson, P., Landsfeld, M., Pedreros, D., Verdin, J., Shukla, S., Husak, G., Rowland, J., Harrison, L.,
Hoell, A., y Michaelsen, J. (2015). The climate hazards infrared precipitation with stations—a new environmental record
for monitoring extremes. *Scientific Data, 2*, 150066. https://doi.org/10.1038/sdata.2015.66

NVIDIA. (2026). *Kumo-Tabular* [Modelo preentrenado]. Hugging Face. https://huggingface.co/nvidia/Kumo-Tabular

Zepeda Villarreal, E. A., Lopez-Ridaura, S., y Sonder, K. (2026). *Municipal Agricultural Production Database (Mexico,
2003–2024): SIACON–SIAP Harmonized Dataset with Agrobiodiversity Indicators* (Versión 3) [Conjunto de datos]. figshare.
https://doi.org/10.6084/m9.figshare.32065674.v3
