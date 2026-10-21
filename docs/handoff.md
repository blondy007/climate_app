# Continuidad · 21 de octubre de 2026

## Estado útil

- Producción verificada: `29632a78d605d76c7b4c3b14abda2e3b201568e5`.
  Iván hizo cherry-pick y redespliegue manual. 70 pruebas Python + 12 Node correctas.
- Histórico: tabla corregida en claro/oscuro; comprobación pública a 320/390 px,
  sin scroll horizontal de página. Cabecera y selección de lugares también resueltas.
- Servidor healthy, HTTPS 200, Kuma Funcional. El trigger automático de Coolify
  devuelve 403; el despliegue manual sí funciona. No investigar ese fallo salvo petición.
- No hay otra corrección pendiente de esta conversación. Esperar los siguientes cambios.

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
