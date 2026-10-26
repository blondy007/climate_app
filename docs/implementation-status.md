# ClimateApp: objetivo y verificación

Solicitud: aplicación personal pública, coste incremental estricto 0 EUR,
misma URL y rama Oracle. Conservar histórico e incorporar previsión de
WeatherNext 3 y fuentes gratuitas, ubicaciones ampliables, archivo automático,
comparación climatológica, evaluación contra estaciones y consenso ponderado.

El objetivo NO está terminado hasta verificar WeatherNext 3 con datos reales.
La autorización de Google está aprobada por correo (revisión 26/10/2026);
faltan el proyecto/dataset sin facturación y la autenticación privada del runtime.
El adaptador no se considera verificado por recibir la aprobación. No activar billing,
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
- 32 pruebas pasan, incluyendo login/logout, aislamiento de WeatherNext, histórico,
  errores, consenso, unidades, filtrado de modelos, cuotas, límites de BigQuery y
  restauración SQLite con procedencia por hora y compatibilidad de rollback.
- APIs reales verificadas: ECMWF, ICON, GFS y MET Norway. GFS ofrece probabilidad
  de precipitación; no se inventa en fuentes que no la suministran.
- Preliminar desplegada en Oracle: `720e9a4`, build ARM64, HTTPS válido y redirección
  desde HTTP, cuatro fuentes operativas y 86.846 observaciones reales persistentes.
- Restauración aislada real: SQLite íntegro (2 lugares, 10 runs, 2.680 puntos,
  86.846 observaciones), CSV original de 4.611.430 bytes conservado.
- GitHub Actions valida las 23 pruebas, pero el trigger de la API Coolify devuelve
  403. Se utilizó el fallback manual autorizado (`bg5ukmw5y35t4h0hcgax3wa7`).
- Preliminar posterior `5788d89` verificada en móvil de 390 px, con leyendas y
  tablas adaptadas. Monitor HTTPS Up; infraestructura documentada y publicada.
- Cuenta del propietario configurada y verificada sin mostrar contraseña ni hash.
- La gestión de estaciones y umbrales conserva la autorización pública previa
  del propietario. WeatherNext y el consenso completo exigen la sesión personal.
- La revisión conserva la pasada y celda nativas por cada hora de
  WeatherNext y evita replicar metadatos JSON al evaluar meses de archivo.
- Rediseño completo de UI/UX con referencias Apple Weather, CARROT y Windy:
  portada, iconos SVG locales, horas desplazables, rangos diarios, navegación
  inferior móvil, ajustes propios y análisis avanzado organizado en vistas.
  Verificado en 320, 390 y 1280 px, con datos sintéticos locales para la previsualización.
- El propietario rechazó Tailscale para acceder a la app: se implementa login en
  la web HTTPS. Tailscale queda reservado a administración de la VPS.
- Pendiente externo: acceso WeatherNext aprobado y credenciales de runtime
  configuradas de forma privada; no se han recibido ni solicitado secretos.
