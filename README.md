# Milli · modelo de rendimiento de cebada

Modelo para el reto **AgroCebada de FIRA 2026**: estima el rendimiento (t/ha) de cebada de temporal en Hidalgo, Puebla y Tlaxcala, ciclo abril–octubre de 2025, para las 59 parcelas del conjunto de predicción. El modelo ya viene ajustado: solo hay que correr la inferencia.

![Observado frente a predicho y error de cada componente del modelo](imagenes/resultados.png)

*Figura 1. a) Rendimiento observado frente a predicho; b) RMSE de cada componente y del modelo completo en 20 repeticiones de validación cruzada.*

## Cómo funciona

Promedia cuatro modelos con pesos fijos: `predicción = (0.65·(GP + Kumo + Dosel)/3 + 0.10·XGBoost) / 0.75`.

- **GP**: proceso gaussiano espacial (kriging), porque las parcelas cercanas rinden parecido (Figura 2).
- **Kumo**: modelo tabular preentrenado de NVIDIA (Figura 4), adaptado con estadística municipal del SIAP.
- **Dosel**: red con atención sobre series de Sentinel-2, basada en MMST-ViT (Figura 5) y Daily DeepCropNet.
- **XGBoost**: preentrenado con estadística municipal del SIAP y clima de AgERA5.

Los índices satelitales por sí solos explican poco del rendimiento (Figura 3); la ubicación y el municipio explican mucho más.

En validación cruzada obtiene **RMSE 0.472 t/ha, MAE 0.323 t/ha y R² 0.690** (Figura 1).

![I de Moran del rendimiento](imagenes/moran.png)

*Figura 2. I de Moran del rendimiento (Moran, 1950): cada parcela frente al promedio de sus 8 vecinas.*

![Análisis de componentes principales](imagenes/pca.png)

*Figura 3. Componentes principales (Jolliffe y Cadima, 2016) de 31 variables de temporada; el rendimiento casi no se alinea con ellos.*

![Arquitectura de Kumo-Tabular](imagenes/kumo.png)

*Figura 4. Arquitectura de Kumo-Tabular: embebido de celdas, codificador de tablas y aprendizaje en contexto. Tomada de Qu et al. (2026).*

![Arquitectura de MMST-ViT](imagenes/mmst_vit.png)

*Figura 5. Arquitectura de MMST-ViT: transformadores multimodal, espacial y temporal sobre imágenes satelitales y clima. Tomada de Lin et al. (2023).*

## Uso

```sh
pip install -r requirements.txt
python predecir.py --reto ruta/a/DATASET_RETO_AGRO_2026
```

Escribe `predicciones.csv` con las 59 predicciones. Con `--verificar` las compara con las originales. Funciona en CPU o GPU y tarda unos segundos.

## Datos

Para predecir solo necesitas el **dataset del reto** (lo entrega FIRA a los equipos). El resto ya viene procesado en `data/` y `modelos/`:

- Sentinel-2 nivel 2A (ESA Copernicus).
- AgERA5 (Copernicus Climate Change Service).
- CHIRPS (Climate Hazards Center).
- Producción agrícola municipal SIACON–SIAP 2003–2024.
- AlphaEarth Satellite Embedding (Google).
- Pesos de Kumo-Tabular (NVIDIA).

## Créditos

- **Desarrollo**: [dotrdp](https://github.com/dotrdp).
- **Datos del reto**: FIRA (Fideicomisos Instituidos en Relación con la Agricultura).
- **Figuras 4 y 5**: NVIDIA (Qu et al., 2026) y Lin et al. (2023).
- **Licencias**: los pesos de `modelos/kumo/` derivan de NVIDIA Kumo-Tabular (OpenMDW-1.1); `third_party/sdm/` es de NVIDIA (Apache-2.0).

## Citas

1. Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1/2), 17–23. <https://doi.org/10.2307/2332142>
2. Jolliffe, I. T., y Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202. <https://doi.org/10.1098/rsta.2015.0202>
3. Qu, J., Hudovernik, V., Jurkovic, M., Lorenz, C., Nitta, A., Gordeev, D., Lopez, F., Bendias, R., Titericz, G., Jr., Sokolovski, A. S., Hulseman, I., Leskovec, J., y Fey, M. (2026, 29 de septiembre). *NVIDIA Kumo Tabular sets a new accuracy-efficiency frontier for tabular prediction*. Hugging Face Blog. <https://huggingface.co/blog/nvidia/kumo-tabular>
4. Lin, F., Crawford, S., Guillot, K., Zhang, Y., Chen, Y., Yuan, X., Chen, L., Williams, S., Minvielle, R., Xiao, X., Gholson, D., Ashwell, N., Setiyono, T., Tubana, B., Peng, L., Bayoumi, M., y Tzeng, N.-F. (2023). MMST-ViT: Climate change-aware crop yield prediction via multi-modal spatial-temporal vision transformer. En *2023 IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 5751–5761). IEEE. <https://doi.org/10.1109/ICCV51070.2023.00531>
5. Xiong, X., Zhong, R., Tian, Q., Huang, J., Zhu, L., Yang, Y., y Lin, T. (2024). Daily DeepCropNet: A hierarchical deep learning approach with daily time series of vegetation indices and climatic variables for corn yield estimation. *ISPRS Journal of Photogrammetry and Remote Sensing*, 209, 249–264. <https://doi.org/10.1016/j.isprsjprs.2024.02.008>
6. NVIDIA. (2026). *Kumo-Tabular*. Hugging Face. <https://huggingface.co/nvidia/Kumo-Tabular>
7. Copernicus Sentinel-2 (processed by ESA). (2021). *MSI Level-2A BOA reflectance product. Collection 1*. <https://doi.org/10.5270/S2_-znk9xsj>
8. Copernicus Climate Change Service. (2020). *Agrometeorological indicators from 1979 to present derived from reanalysis*. <https://doi.org/10.24381/cds.6c68c9bb>
9. Funk, C., Peterson, P., Landsfeld, M., *et al.* (2015). The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. *Scientific Data*, 2, 150066. <https://doi.org/10.1038/sdata.2015.66>
10. Zepeda Villarreal, E. A., Lopez-Ridaura, S., y Sonder, K. (2026). *Municipal Agricultural Production Database (Mexico, 2003–2024): SIACON–SIAP Harmonized Dataset with Agrobiodiversity Indicators* (v3). figshare. <https://doi.org/10.6084/m9.figshare.32065674.v3>
11. Brown, C. F., Kazmierski, M. R., Pasquarella, V. J., *et al.* (2025). *AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data*. arXiv:2507.22291. <https://doi.org/10.48550/arXiv.2507.22291>
12. FIRA. (2026). *Reto AgroCebada FIRA 2026: Bases de participación*. <https://www.fira.gob.mx/Nd/Reto%20Agrocebada%20Bases.pdf>
