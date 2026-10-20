# ClimateApp · Histórico, previsión y aciertos

Aplicación personal en español con previsión de ECMWF, ICON, GFS y MET Norway,
comparación climatológica, archivo automático, evaluación de aciertos y consenso
ponderado por observaciones reales. Conserva el histórico Meteostat existente.

WeatherNext 3 incorpora adaptador y acceso personal con login HTTPS. Su validación
real sigue pendiente de aprobación de Google y credenciales privadas.

Interfaz adaptada a móvil y escritorio: portada meteorológica, iconos SVG propios,
franja horaria desplazable, días con rangos de temperatura y navegación Tiempo,
Aciertos, Histórico y Ajustes. El análisis avanzado se conserva en vistas dedicadas.
En Ajustes puedes elegir tema claro, oscuro o automático (según el dispositivo),
recordado por navegador, y quitar o recuperar lugares sin perder su historial.
[Referencias y decisiones de diseño](docs/design.md).

**Coste incremental estricto: 0 EUR.** No activar billing, pruebas con cargo,
APIs de pago ni recursos Oracle adicionales.

- [Solicitar y configurar WeatherNext](docs/weathernext-setup.md)
- [Login, colector, datos, backup y despliegue](docs/operations.md)

## Entorno virtual

### Windows
1. `python -m venv .venv`
2. `.\.venv\Scripts\activate`
3. `pip install -r requirements.txt`

### Unix/macOS
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

## Ejecución
- `python -m climate_app.runtime`: web y recopilación automática.
- `streamlit run app.py`: solo interfaz sobre datos previamente recopilados.
- `python -m unittest discover -s tests -v`: pruebas sin APIs ni credenciales reales.

## Docker

La imagen escucha en el puerto `8501` y expone el healthcheck de Streamlit en
`/_stcore/health`.

El directorio `/app/data` contiene el CSV maestro y los umbrales modificables.
Debe montarse como volumen persistente. Cuando el volumen está vacío, el
entrypoint copia los datos iniciales incluidos en la imagen; nunca sobrescribe
datos existentes durante un redeploy.

## Funcionalidades
- Carga automática del CSV Meteostat por defecto y soporte para archivos subidos con el mismo esquema.
- Filtros por fechas, meses, horas, ciudades y escenarios, con opción "Todos" para mantener todos los registros.
- Ajuste de umbrales de viento por ciudad directamente en la barra lateral, aplicados en tiempo real.
- Agregaciones mensuales con mediana, media, percentiles 25 y 75, mínimo y máximo para temperatura y sensación térmica.
- Gráfico mensual comparativo por ciudad, boxplots opcionales de temperatura y sensación térmica, y mensajes de ausencia de datos por ciudad/mes.
- Exportaciones en CSV de la tabla filtrada, del agregado mensual y del dataset de distribución cuando está habilitado.

## Pruebas manuales
1. Caso 1: Hora única 14:00, escenario “Calma” → Torrejón suele tener datos, Los Alcázares puede no tener; el gráfico debe mostrar Torrejón y avisos “sin datos” para Alcázares en meses correspondientes.
2. Caso 2: Varias horas (00:00 y 07:00), escenario “Ventoso” → deben aparecer ambas ciudades en varios meses.
3. Caso 3: Cambiar el umbral “Ventoso” para Los Alcázares a 20 km/h y comprobar que reclasifica al vuelo (la curva cambia).
