# Diseño de ClimateApp

Referencias visuales consultadas el 20 de octubre de 2026:

- [Apple Weather](https://apps.apple.com/us/app/weather/id1069513131/): temperatura
  protagonista, previsión horaria horizontal y días con rangos de temperatura.
- [CARROT Weather](https://www.meetcarrot.com/weather/): ilustración meteorológica,
  lectura rápida y datos agrupados en módulos.
- [Windy](https://www.windy.com/): selección de fuentes y comparación de modelos,
  con detalle disponible sin saturar la consulta principal.

La implementación conserva Streamlit y los servicios existentes. Usa una paleta
costera de azul petróleo, fondos claros u oscuros y acentos cálidos, tipografía del sistema,
tarjetas adaptables y SVG originales locales. No añade bibliotecas, cuotas,
recursos remotos de diseño ni suscripciones.

## Navegación

- Tiempo: lugar, próxima hora, franja horaria, cuatro indicadores y días con rangos.
- Aciertos: evaluación observada por variable y antelación; estado vacío explicativo
  mientras se acumulan observaciones, sin mostrar métricas ficticias.
- Histórico: análisis original, con filtros, estaciones, umbrales y carga de CSV
  accesibles desde un panel plegable en la propia página, también en móvil.
- Ajustes: apariencia, gestión de lugares, cuenta personal, fuentes y licencias.

El selector Automático / Claro / Oscuro recuerda la elección por navegador.
El modo automático sigue el dispositivo; tarjetas, formularios, navegación y
gráficos comparten la paleta. La gestión muestra cada lugar con su botón Quitar,
y un apartado de lugares quitados permite recuperarlo conservando su archivo.
El aviso de retirada ofrece Deshacer. Los alias distinguen sitios con nombres
iguales; mientras no tengan alias, el selector añade coordenadas para diferenciarlos.
La cabecera mantiene selector y botón de gestión en una fila también en móvil;
el botón desplaza directamente a Tus lugares.

Las filas diarias son botones accesibles con teclado y área táctil completa.
El contenedor Markdown de cada fila elimina el margen negativo de Streamlit
para que contenido, botón, fondo y borde tengan la misma altura. El resaltado
seleccionado y el foco se dibujan hacia dentro y no invaden los días contiguos.
Al elegir una se resaltan y se abre su franja horaria, con vuelta a las próximas
24 horas. Solo aparecen las horas disponibles del día y horizonte elegidos.
Los cambios de lugar o modelo reinician la selección; un horizonte que ya no
incluye el día también la reinicia. Los días de 25 horas conservan ambas horas
repetidas con su zona, y la portada sigue describiendo la próxima hora prevista.

Streamlit se actualiza de 1.37.0 a 1.38.0 para corregir el fallo de su árbol de
bloques que podía dejar la pantalla en blanco al cambiar la vista
([corrección oficial](https://github.com/streamlit/streamlit/pull/9276)).

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
