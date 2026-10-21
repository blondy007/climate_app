# ClimateApp

App personal de meteorología, Python/Streamlit, desplegada en Oracle con Coolify.

## Trabajo y contexto

- Responder en español y con brevedad. Resolver preguntas simples sin auditorías.
- Al retomar en un chat nuevo, leer `docs/handoff.md` y solo los archivos necesarios
  para la petición actual. No cargar conversaciones completas ni el registro largo
  de `test-results/continuation.md` salvo que falte un dato concreto.
- No reabrir tareas ya resueltas, revisar servicios ni ejecutar pruebas completas
  por una pregunta informativa. Para cambios, verificar lo proporcional al riesgo;
  una vez comprobados, no repetir la misma validación sin motivo.
- Usar búsquedas acotadas; excluir entornos virtuales y dependencias de resultados.
- Actualizar el resumen de continuidad en lugar de acumular cronologías duplicadas.

## Límites y producto

- No desplegar ni hacer push a `oracle-production` sin nueva petición explícita.
  Esa rama dispara despliegues. Mantener cambios nuevos en una rama `codex/`.
- Para operar la VPS, aplicar `oracle-vps-ops`. Coste incremental estricto: 0 EUR.
- No leer/exportar SQLite de producción ni credenciales durante pruebas de UI.
- Priorizar móvil, contraste claro/oscuro y diseño sin exceso de tarjetas.
  Conservar funciones existentes, accesibilidad y sensación térmica en previsión.
- WeatherNext espera aprobación de Google. No simular acceso ni datos.
- Investigación de lluvia separada y no pública. No modificar el algoritmo,
  predicciones emitidas, plazos o recursos sin autorización. No promover investigación
  a producción junto con una corrección de interfaz.
