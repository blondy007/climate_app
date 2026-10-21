# Continuidad · 21 de octubre de 2026

## Estado útil

- Producción actual: `11cf633a1a3c8bf9f38c232ca1da32e3789874a5`, publicada por petición
  de Iván desde `codex/compact-place-controls`. Actions 39653861713: 84 Python + 16 Node
  correctas y disparador API correcto. Imagen exacta running/healthy, HTTPS/health 200,
  logs de arranque limpios, Kuma 5 Up, volumen y binding local conservados. Tus lugares
  usa lápiz y aspa por fila, alias en línea con guardar/cancelar y confirmación antes
  de quitar. Lugares quitados usa recuperar/aspa; borrado definitivo mantiene sesión
  personal y confirmación. 9 pruebas enfocadas correctas (alias, cancelación, duplicados,
  selección, recuperación y autorización). Capturas locales con datos ficticios en
  `test-results/compact-place-controls-mobile.png` y `compact-place-confirmation-mobile.png`.
  Editor comprobado en producción a 390 px sin guardar cambios ni quitar lugares.
  Evidencia: `test-results/compact-place-controls-production.png`. Registro canónico
  publicado como `8ea732d` en master de oracle-vps-infra (apps/climate-app.md e
  inventory/applications.yaml). Sin tareas pendientes de este despliegue.
- Versión anterior / rollback: `083c25c6dba25827baccdf5a6a10a0f437bc4ad0`.
  Rama anterior `codex/manual-weather-refresh`. Despliegue y registro documental
  aprobados expresamente por Iván. Actions 39647808142: 83 Python + 16 Node correctas
  y disparador API correcto. Despliegue completamente automático, sin fallback.
- Actualizar previsión es un icono circular dentro del widget de lugares guardados.
  Consulta las cuatro fuentes abiertas, conserva cuota gratuita, respeta Expires de
  MET Norway y recalcula el consenso público con el método existente. Límite por lugar
  compartido de quince minutos en SQLite; fallos conservan los datos anteriores.
  Funciona también sin previsión previa. WeatherNext y lluvia experimental intactos.
- Pruebas aisladas: cuota, caché, concurrencia, duplicados, fallos parciales/totales,
  selector sin None al cambiar opciones y recuperación del componente montado tarde.
  Clic local a 390 px cambió datos ficticios de 18 a 21 °C. Prueba real de Los Narejos
  autorizada aparte tras rechazo de auto-review: éxito a las 19:57 CEST, hora actualizada
  e icono desactivado. Captura pública: `test-results/manual-refresh-production.png`.
- Imagen exacta running/healthy, HTTPS/health 200, sesión anónima no-store, logs limpios,
  binding 127.0.0.1:8501 y volumen original conservados. Kuma 5 Funcional desde 19:56
  tras el reemplazo (404 breve a las 19:55). No se leyó/exportó SQLite ni credenciales
  de producción para verificar la interfaz; solo se ejecutó la actualización aprobada.
- Coolify 403 resuelto: su lista solo admitía Tailscale, pero Nginx ve la pasarela
  Docker 10.0.2.1. Excepción /32 autorizada, persistida y verificada por SSH; API admite
  la pasarela y rechaza una IP externa. Actions real confirma el arreglo. Se mantienen
  token, panel privado y puertos. Rollback: retirar únicamente 10.0.2.1/32 de la lista.
- Selección recordada por navegador, nombre aproximado de ubicación automática y
  borrado definitivo de lugares quitados ya publicados. El borrado exige sesión personal
  y segunda confirmación; elimina solo el ID elegido. No se borraron lugares reales.
  Los rediseños de portada, histórico y Ajustes están resueltos: no reabrirlos.
- Registros de infraestructura `2a442d3` y cierre `01ad94c` publicados en master de
  oracle-vps-infra: versión activa, modo automático y aceptación real documentados.
  Sin tareas pendientes de esta corrección ni otro despliegue de aplicación.

## Preferencias y límites

- Respuestas breves y verificaciones proporcionadas. No repetir comprobaciones sin motivo.
- No desplegar ni hacer push a oracle-production sin petición explícita nueva. Los commits
  locales de continuidad no deben provocar otro despliegue.
- App pública desde móvil sin Tailscale; administración privada con Tailscale. Coste 0 EUR.
- Login personal persistente implementado; WeatherNext espera aprobación real de Google.
- Después de cerrar este trabajo, Iván prevé separar front y back para web de escritorio
  y móvil, Android muy probable e iPhone posible. No se ha iniciado la migración.

## Consultar solo si aplica

- Interfaz y actualización: climate_app/ui/forecast_dashboard.py, design.py, weather.css;
  climate_app/forecast/refresh.py; criterios en docs/design.md y docs/operations.md.
- Operación: skill oracle-vps-ops y repo local Oracle_cloud/oracle-vps-infra.
- Radar separado: research/nowcast/README.md e informe results/2026-10-20.md. Captura
  congelada como máximo hasta 27/10/2026 20:00:15 UTC; no ampliar ni publicar.
  Heartbeat validar-lluvia-de-climateapp en el chat anterior: no duplicarlo.
- Adjuntos sin versionar conservados. test-results es local e ignorado; no cargar su
  cronología completa. Las capturas y fixtures no contienen credenciales reales.

Este resumen es el punto de partida del chat nuevo; no es una orden de iniciar trabajos.
