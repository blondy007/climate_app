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
secciones abiertas y SVG originales locales. No añade bibliotecas, cuotas,
recursos remotos de diseño ni suscripciones.

## Navegación

- Tiempo: lugar, próxima hora, franja horaria, cuatro indicadores y días con rangos.
- Aciertos: evaluación observada por variable y antelación; estado vacío explicativo
  mientras se acumulan observaciones, sin mostrar métricas ficticias.
- Histórico: análisis original, con filtros, estaciones, umbrales y carga de CSV
  accesibles desde un panel plegable en la propia página, también en móvil.
- Ajustes: apariencia, modelo principal, gestión de lugares, cuenta personal, fuentes y licencias.

El selector Automático / Claro / Oscuro recuerda la elección por navegador.
El modo automático sigue el dispositivo; secciones, formularios, navegación y
gráficos comparten la paleta. La gestión muestra cada lugar con su botón Quitar,
y un apartado de lugares quitados permite recuperarlo conservando su archivo.
El aviso de retirada ofrece Deshacer. Los alias distinguen sitios con nombres
iguales; mientras no tengan alias, el selector añade coordenadas para diferenciarlos.
La cabecera ofrece lugares en una franja horizontal desplazable, sin desplegable,
con selección por toque y teclado. Ubicación actual solicita el permiso nativo;
los errores ofrecen reintentar o elegir un lugar guardado. El botón de gestión
desplaza directamente a Tus lugares. El modelo principal se elige mediante una
lista de opciones en Ajustes y se recuerda por navegador; la portada identifica
siempre la fuente efectiva y avisa si debe utilizar otra disponible.
Los indicadores se centran, con cuatro columnas en escritorio y dos en móvil.

Las filas diarias son botones accesibles con teclado y área táctil completa.
El contenedor Markdown de cada fila elimina el margen negativo de Streamlit
para que contenido, botón y fondo tengan la misma altura. El resaltado seleccionado
usa un fondo tenue y una marca lateral. El foco de teclado se dibuja hacia dentro
y no invade los días contiguos.
Al elegir una se resaltan y se abre su franja horaria, con vuelta a las próximas
24 horas. Solo aparecen las horas disponibles del día y horizonte elegidos.
Los cambios de lugar o modelo reinician la selección; un horizonte que ya no
incluye el día también la reinicia. Los días de 25 horas conservan ambas horas
repetidas con su zona, y la portada sigue describiendo la próxima hora prevista.

La previsión muestra temperatura y sensación térmica juntas en portada, horas y
rangos diarios; Evolución muestra ambas curvas. Comparar modelos y Climatología
permiten elegir la sensación, que también figura en tablas, CSV y Aciertos.
Se calcula por hora con temperatura, humedad y viento de la misma fuente (o las
variables del consenso), usando [Steadman a la sombra](https://www.bom.gov.au/info/thermal_stress/#atapproximation).
El viento a 10 m se convierte de km/h a m/s; no se añade radiación solar. Se
requieren los tres datos finitos, humedad de 0 a 100 %, viento no negativo y
temperatura meteorológica plausible (-100 a 70 °C). No se rellena con temperatura
cuando falta un dato. Los rangos diarios parten de las sensaciones horarias y
las tablas indican su cobertura. La banda P10–P90 sigue siendo solo de temperatura.
Climatología y Aciertos aplican el mismo método a las observaciones completas;
la sensación no se presenta como una medición directa. Se deriva al consultar,
sin migraciones ni cambios de datos archivados, cuotas o consultas externas.
El análisis histórico original conserva sus índices de frío/calor existentes.

Streamlit se actualiza de 1.37.0 a 1.38.0 para corregir el fallo de su árbol de
bloques que podía dejar la pantalla en blanco al cambiar la vista
([corrección oficial](https://github.com/streamlit/streamlit/pull/9276)).

La previsión usa un lienzo continuo: horas y días no tienen marco, y viento,
humedad y lluvia se leen como una lista de datos con separadores finos. Los
gráficos, indicadores del histórico y conclusiones también prescinden de cajas
decorativas. La tarjeta principal conserva la ilustración y el color del cielo;
los campos de formulario mantienen su contorno para reconocerlos como controles.
La primera hora no lleva un resaltado que pueda confundirse con una selección.

En móvil la navegación queda al alcance del pulgar en una barra inferior continua,
con icono y texto activos en color, sin tarjetas dentro de una tarjeta. Los
controles conservan interacción de teclado y foco visible. Las horas se presentan
en la zona del lugar. Los textos del usuario se escapan antes de generar HTML.

Dentro de Tiempo se mantienen las vistas Evolución, Comparar modelos y Climatología,
la selección de fuentes y horizontes, las tablas completas y las descargas previas.
Los datos ausentes aparecen como una raya; los iconos se derivan de las variables
disponibles y no crean probabilidades ni observaciones. La temperatura protagonista
siempre se identifica como previsión de la próxima hora disponible.

WeatherNext y el consenso privado conservan el control de acceso existente. La
gestión pública de estaciones y umbrales mantiene la autorización del propietario.
