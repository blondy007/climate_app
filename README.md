# Clima Meteostat · Comparativa

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
- `streamlit run app.py`

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
