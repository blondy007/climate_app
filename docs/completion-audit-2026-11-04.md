# Auditoría del objetivo original · 04/11/2026

**Cierre global todavía no acreditado: falta comprobar la actualidad de las
previsiones persistidas en producción. No falta solicitar de nuevo acceso a Google.**

Esta auditoría conserva el objetivo completo: WeatherNext 3 operativo junto a
previsión, climatología, archivo automático, proveedores gratuitos, evaluación
observacional y consenso ponderado, manteniendo coste adicional 0 EUR y lugares
ampliables. No sustituye WeatherNext por una alternativa ni reabre el experimento
de lluvia, cuyo fallo y corrección están cerrados en otra rama.

## Evidencia actual inspeccionada

- Código de aplicación revisado: `e09d65fa47b6842ce7eeec53ce6cd7ec84ac0065`;
  documentación de cierre original `cca6517`. La rama de esta auditoría solo
  añade documentación y una herramienta opcional de lectura agregada.
- VPS: imagen exacta e09d65f, **Up 6 days (healthy)** el 04/11.
- HTTPS externo: `/_stcore/health` devuelve **200** y `/_climate/auth/session`
  devuelve **200**, `authenticated=false`, `Cache-Control: no-store`.
- Validación funcional local del código de la aplicación: **90 pruebas Python
  correctas**, 47,759 s; **16 pruebas Node correctas**. Las pruebas usan fixtures
  temporales, sin consultar proveedores ni bases de producción.
- Integración real documentada el 29/10 en `docs/weathernext-setup.md` y el registro
  canónico de infraestructura: Earth Engine no comercial, facturación deshabilitada,
  credencial privada montada en lectura, 348 horas WeatherNext futuras por cada
  uno de tres lugares en aquel momento y consenso privado persistido.
  Esto acredita la entrega del 29/10; no demuestra por sí solo actualidad el 04/11.

## Comprobación por requisito

| Requisito original | Evidencia inspeccionada | Resultado |
|---|---|---|
| 1. App personal accesible públicamente | HTTPS externo 200, sesión anónima no-store; configuración de acceso público y login personal | Acreditado para acceso público; WeatherNext reservado al propietario |
| 2. WeatherNext 3 real | Adaptadores `weathernext.py`/`weathernext_ee.py`, malla/tiempos/unidades/cuota/billing probados, aceptación real del 29/10 y misma imagen activa | Implementado y entregado; pendiente comprobar actualidad de las previsiones guardadas |
| 2. Pantalla de previsión y comparación climatológica | `forecast_dashboard.py`, `analysis.climatology`, pruebas AppTest de previsión, fechas y condiciones ausentes; interfaz ya aceptada en despliegue | Implementado y pruebas correctas |
| 2. Almacenamiento automático | Proceso `worker.run_forever`, archivo inmutable por pasada, procedencia por hora y snapshots; pruebas de inmutabilidad y restauración | Implementado; recuentos actuales pendientes del permiso indicado abajo |
| 2. Otros proveedores gratuitos | ECMWF, ICON, GFS y MET Norway, parsers y cuotas; recopilación real de entrega documentada | Implementados y probados; el informe agregado preparado incluye su actualidad |
| 2. Evaluación de aciertos | `scores` empareja predicciones anteriores al resultado con observaciones de estación, excluye sustitutos de modelo y duplicados | Implementado y probado; no se promete que un proveedor sea mejor sin muestras suficientes |
| 2. Consenso ponderado | `combine` y `rebuild_consensus`: pesos por variable/horizonte, mínimo 30 muestras/3 días, regularización, separación pública/privada | Implementado y probado con evidencia controlada y entrega real del 29/10 |
| 3. Explicar y obtener acceso Google | Solicitud documentada, aprobación y registro no comercial confirmados, identidad autorizada y montaje verificados el 29/10 | Resuelto; no se pide otra cuenta ni otra solicitud |
| 4. Límite inmutable de 0 EUR | Lecturas bloqueadas si billing no es explícitamente false o registro no comercial no se confirma; cuota diaria persistente 240, BigQuery fuera de uso | Política implementada y probada; en esta auditoría no se crean recursos ni consultas de proveedor |
| 5. Preliminar gratuita y WeatherNext necesario para finalizar | Preliminar anterior documentada, WeatherNext entregado el 29/10, Docker/HTTPS actuales correctos | Cierre pendiente únicamente de acreditar actualidad, sin redefinir el requisito |
| 6. Añadir otros lugares | Gestión manual/búsqueda, `Store.add_location`, alias, duplicados, quitar/recuperar y archivo conservado; pruebas AppTest/storage | Implementado y pruebas correctas |
| 7. Conservar lo existente | `render_history`, CSV histórico y umbrales conservados, restauraciones históricas y compatibilidad de almacenamiento probadas | Sin cambios en los datos existentes durante esta auditoría |

## Comprobación bloqueada y propuesta concreta

La revisión automática rechazó ejecutar una comprobación agregada sobre
`forecasts.sqlite3` de producción por la prohibición previa de leer SQLite en
producción. **La consulta no se ejecutó.** No se ha usado otra herramienta,
endpoint o copia para eludir esa decisión ni se ha leído ninguna credencial.

Se prepara para revisión [scripts/audit_forecast_storage.py](../scripts/audit_forecast_storage.py).
Requiere aprobación explícita separada antes de usarlo en producción. Abre SQLite
con `mode=ro`, activa `query_only`, limita el trabajo a 15 segundos y emite solo:

- cantidad de lugares activos, representados con números locales;
- proveedor y fecha de descarga de su última pasada guardada;
- cantidad de horas futuras y primera/última hora válida.

No devuelve nombres/IDs de lugares, coordenadas, valores meteorológicos, cuentas,
tokens ni credenciales. No exporta SQLite, no modifica datos, no consulta Google
y no fuerza una recopilación. Dos pruebas con base sintética verifican recuentos,
exclusión de lugares inactivos/pasadas antiguas, anonimización, contenido de la base
inalterado y ausencia de creación de una base inexistente. La herramienta no se
despliega ni ejecuta en producción como parte de esta preparación.

La salud Docker es evidencia del proceso activo y su comprobación compuesta; no
se presenta como prueba de que un proveedor concreto siga entregando datos nuevos.
Hasta obtener esa evidencia, el objetivo global permanece abierto.
