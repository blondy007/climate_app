# Continuidad · 21 de octubre de 2026

## Estado útil

- Publicado `a876f18bd8cc516f762c1bf7d063ca6a13ab31e7` con petición explícita de Iván.
  Rama de trabajo: `codex/location-memory-and-deletion`. El navegador recuerda
  el lugar guardado elegido; la ubicación automática muestra el nombre resuelto
  con BigDataCloud desde coordenadas aproximadas recién obtenidas. Lugares quitados
  permite borrado definitivo con sesión personal y segunda confirmación; transacción
  por ID, protección frente a respuestas tardías y favoritos iniciales sin resurrección.
  77 pruebas Python y 15 Node correctas. Revisión local ficticia: selección conservada
  en pestaña nueva y aviso de borrado en móvil. No se han leído ni eliminado datos de
  producción. CI correcto, API 403 y fallback manual de Coolify; imagen exacta
  running/healthy, HTTPS/health 200, binding/volumen conservados, logs limpios y Kuma 5
  Funcional. La UI pública muestra Borrar definitivamente para el duplicado quitado.
- Pendiente: `ac1232e` corrige el aviso de valor por defecto duplicado de Streamlit
  al restaurar la ubicación. 13 pruebas focalizadas correctas. Auto-review rechazó su
  push a producción por exigir autorización explícita nueva; pregunta enviada a Iván.
  No publicar sin esa respuesta. Se observó también un aviso de carga del componente
  en una pestaña pública de Brave: reaparece tras esperar incluso con tema explícito,
  aunque el HTML del componente responde 200. Pendiente de acotar; no darlo por resuelto.
  El registro de infraestructura `ee02d5e` se publicó en `master` con autorización
  expresa de Iván tras el bloqueo inicial de auto-review.
- Consulta posterior: el 403 exacto corresponde a `ApiAllowed` de Coolify 4.1.2,
  filtro de IP permitidas (código oficial); falta identificar la IP que ve Coolify.
  No se han cambiado permisos, tokens ni configuración. Los lugares guardados no
  tienen actualización meteorológica manual: Actualizar vista solo relee SQLite;
  recopilación cada tres horas. Iván preguntó por ello, sin petición de implementación aún.
- Producción anterior verificada: `0ce4b74d9ae44a4ebdcb7db465fe2917ff1055b5`.
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
