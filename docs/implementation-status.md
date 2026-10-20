# ClimateApp: objetivo y verificación

Solicitud: aplicación personal pública, coste incremental estricto 0 EUR,
misma URL y rama Oracle. Conservar histórico e incorporar previsión de
WeatherNext 3 y fuentes gratuitas, ubicaciones ampliables, archivo automático,
comparación climatológica, evaluación contra estaciones y consenso ponderado.

El objetivo NO está terminado hasta verificar WeatherNext 3 con datos reales.
El propietario debe solicitar la autorización de Google. No activar billing,
pruebas de pago, recursos OCI ni APIs comerciales.

## Implementación prevista

- Streamlit/Python y Docker existentes; SQLite en el volumen `/app/data`.
- Recolector independiente de sesiones web, reintentos limitados y cuotas.
- ECMWF, DWD ICON y NOAA GFS vía Open-Meteo; MET Norway como segunda API.
- WeatherNext 3 mediante BigQuery sin facturación, con comprobación de billing,
  dry run, límites por consulta y presupuesto mensual de bytes.
- Observaciones Meteostat con `model=False`, estación explícita y distancias.
- Evaluación por ubicación, variable y horizonte; datos disponibles antes del
  instante de predicción, mínimo de muestras y pesos regularizados.
- Incertidumbre nativa separada de desacuerdo entre modelos; sin probabilidades
  inventadas ni verificación contra un modelo presentado como observación.
- Snapshot consistente SQLite para backup y restauración aislada.

## Evidencia y pendientes

- 2026-10-20: checkout inicial limpio, `oracle-production` = `4ae8cd1` remoto.
- VPS accesible por Tailscale; Climate App inicial healthy en ese commit.
- Volumen ya incluido en backup integral y monitor público existente.
- Rama de desarrollo `codex/forecast-comparison`.
- Implementados: SQLite, cuatro fuentes gratuitas, recolector independiente,
  observaciones reales, referencia de cinco años, evaluación, consenso y login.
- Prueba real aislada: cuatro modelos y unas 43.000 observaciones por cada uno
  de los dos lugares, con datos recientes y backup SQLite consistente.
- 23 pruebas pasan, incluyendo login/logout, aislamiento de WeatherNext, histórico,
  errores, consenso, unidades, filtrado de modelos, cuotas y restauración SQLite.
- APIs reales verificadas: ECMWF, ICON, GFS y MET Norway. GFS ofrece probabilidad
  de precipitación; no se inventa en fuentes que no la suministran.
- Pendiente: verificación final y visual, HTTPS, despliegue y restauración en VPS.
- El propietario rechazó Tailscale para acceder a la app: se implementa login en
  la web HTTPS. Tailscale queda reservado a administración de la VPS.
- Pendiente externo: acceso WeatherNext aprobado y credenciales de runtime
  configuradas de forma privada; no se han recibido ni solicitado secretos.
