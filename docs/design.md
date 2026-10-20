# Diseño de ClimateApp

Referencias visuales consultadas el 20 de octubre de 2026:

- [Apple Weather](https://apps.apple.com/us/app/weather/id1069513131/): temperatura
  protagonista, previsión horaria horizontal y días con rangos de temperatura.
- [CARROT Weather](https://www.meetcarrot.com/weather/): ilustración meteorológica,
  lectura rápida y datos agrupados en módulos.
- [Windy](https://www.windy.com/): selección de fuentes y comparación de modelos,
  con detalle disponible sin saturar la consulta principal.

La implementación conserva Streamlit y los servicios existentes. Usa una paleta
costera de azul petróleo, fondos claros y acentos cálidos, tipografía del sistema,
tarjetas adaptables y SVG originales locales. No añade bibliotecas, cuotas,
recursos remotos de diseño ni suscripciones.

## Navegación

- Tiempo: lugar, próxima hora, franja horaria, cuatro indicadores y días con rangos.
- Aciertos: evaluación observada por variable y antelación; estado vacío explicativo
  mientras se acumulan observaciones, sin mostrar métricas ficticias.
- Histórico: análisis original, con filtros, estaciones, umbrales y carga de CSV
  accesibles desde un panel plegable en la propia página, también en móvil.
- Ajustes: cuenta personal, gestión de lugares, estado de fuentes y licencias.

En móvil la navegación queda al alcance del pulgar en la parte inferior. Los
controles conservan interacción de teclado y foco visible. Las horas se presentan
en la zona del lugar. Los textos del usuario se escapan antes de generar HTML.

Dentro de Tiempo se mantienen las vistas Evolución, Comparar modelos y Climatología,
la selección de fuentes y horizontes, las tablas completas y las descargas previas.
Los datos ausentes aparecen como una raya; los iconos se derivan de las variables
disponibles y no crean probabilidades ni observaciones. La temperatura protagonista
siempre se identifica como previsión de la próxima hora disponible.

WeatherNext y el consenso privado conservan el control de acceso existente. La
gestión pública de estaciones y umbrales mantiene la autorización del propietario.
