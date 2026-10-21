# Continuidad · 21 de octubre de 2026

## Estado útil

- Producción verificada: `0ce4b74d9ae44a4ebdcb7db465fe2917ff1055b5`.
  Corrección de Ajustes aprobada y despliegue solicitado por Iván el 21/10; publicado mediante
  el fallback manual de Coolify. CI: 71 pruebas Python + 12 Node correctas.
- Histórico: tabla corregida en claro/oscuro; comprobación pública a 320/390 px,
  sin scroll horizontal de página. Cabecera y selección de lugares también resueltas.
- Servidor healthy, HTTPS 200, Kuma Funcional. El trigger automático de Coolify
  devuelve 403; el despliegue manual sí funciona. No investigar ese fallo salvo petición.
- Rediseño del widget publicado desde `codex/current-weather-redesign`.
  Temperatura/sensación a la izquierda,
  ilustración/condición a la derecha y rangos en una franja inferior. Actualizar
  ubicación es un icono circular nativo dentro de la cabecera; reintentos conservados.
  21 pruebas focalizadas + 1 integración de actualización/recuperación correctas.
  Vista local ficticia comprobada a 320/390 px en claro/oscuro y escritorio.
  Aceptación pública móvil a 390 px; captura: `test-results/current-widget-production.png`.
  Diseño en `docs/design.md`. No hay cambios pendientes de este rediseño.
- Corrección de Ajustes publicada desde `codex/settings-desktop-layout`:
  columnas alineadas en PC, modelos en cuadrícula, filas de lugares sin bordes
  duplicados, fuentes bajo su título y formularios sin solapamientos. Cuatro
  pruebas focalizadas correctas; revisión local ficticia en escritorio 1024/1440 px
  y móvil 320 px oscuro, sin desbordamiento. Aceptación pública de PC a 1440 px,
  imagen exacta healthy, HTTPS/health 200, logs de arranque sin errores y Kuma 5
  Funcional. Captura: `test-results/settings-production-desktop.png`.
  Registro de infraestructura `1fbc973` publicado en `master` de `oracle-vps-infra`
  con autorización explícita de Iván. Sin tareas pendientes de esta corrección.

## Preferencias y límites

- Iván quiere contexto reducido, respuestas breves y verificaciones proporcionadas.
- No desplegar automáticamente: preparar correcciones e instrucciones cuando se solicite.
- App pública desde móvil sin Tailscale; administración privada de VPS con Tailscale.
- Login personal con persistencia ya implementado. WeatherNext pendiente de Google.
- No volver a probar credenciales ni consultar la base de producción para comprobar UI.

## Consultar solo si aplica

- Interfaz: `climate_app/ui/weather.css`, `design.py`, `dashboard.py`,
  `forecast_dashboard.py`; criterios en `docs/design.md`.
- Operación: `docs/operations.md` y skill `oracle-vps-ops`.
- Radar: `research/nowcast/README.md` e informe `results/2026-10-20.md`.
  Captura congelada hasta 27/10/2026 20:00:15 UTC como máximo; no pública.
  El heartbeat `validar-lluvia-de-climateapp` sigue en el chat anterior: no duplicarlo.
- Los adjuntos de usuario sin versionar se conservan. `test-results/` es local e ignorado;
  no cargar su historial completo. Evidencia última: `history-table-public-dark.png`.

Este resumen es el punto de partida del chat nuevo; no es una orden de iniciar trabajos.
