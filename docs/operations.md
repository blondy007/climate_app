# Operación de ClimateApp

## Runtime y datos

`python -m climate_app.runtime` supervisa Streamlit y un recolector independiente
de las visitas. Reinicia el recolector si termina inesperadamente y propaga señales
de parada. Un bloqueo de proceso evita dos recolectores sobre el mismo volumen.
La web arranca con `python -m climate_app.web run app.py`, que registra las rutas
de sesión en el mismo servidor y puerto. `streamlit run app.py` directo no incorpora
esas rutas y no permite recordar el login. La integración está probada con la
versión fijada de Streamlit 1.38.0; revisar el registro de rutas al actualizarla.

El volumen existente `/app/data` conserva el CSV y umbrales originales, además de:

- `forecasts.sqlite3`: lugares, snapshots inmutables, observaciones, estados y cuotas.
- `forecasts-backup.sqlite3`: copia online consistente, renovada después de cada
  ciclo y validada con `PRAGMA integrity_check`.
- `auth-owner.json`: usuario y hash del propietario si se configura mediante CLI.
- Cachés prescindibles de Meteostat, que no sustituyen a los datos archivados.

El despliegue no sobrescribe el volumen. El esquema es aditivo y no hay purgas
automáticas de previsiones ni de observaciones.

## Login del propietario

El formulario está en **Ajustes → Tu cuenta → Acceso personal · WeatherNext**.
La navegación inferior permanece accesible en móvil. La gestión de lugares y el
estado de fuentes también están en Ajustes; los filtros y estaciones históricas
se abren dentro de Histórico, sin depender de una barra lateral oculta.

## Apariencia y lugares

**Ajustes → Apariencia** ofrece Automático, Claro y Oscuro. La preferencia se
guarda en el navegador, separada de la cuenta y de la base de datos compartida.
Automático sigue el esquema de color del dispositivo, incluidos sus cambios.
Si el navegador bloquea el almacenamiento local, la elección dura la sesión.

El botón **Mis lugares**, junto al selector de Tiempo, abre directamente Tus lugares
en Ajustes. El icono de lápiz permite identificar cada ubicación (máximo 60 caracteres);
abre un campo en línea con una marca para guardar y un aspa para cancelar.
dejarlo vacío recupera su nombre original. Los alias están en una tabla aditiva
`location_aliases`, dentro del mismo SQLite y de sus snapshots, sin cambiar IDs,
coordenadas ni la tabla original de lugares; el esquema sigue admitiendo rollback.

En **Tus lugares**, cada ubicación tiene
un aspa que pide confirmar **Sí, quitar**: desactiva su recopilación sin borrar previsiones ni
observaciones. **Lugares quitados → Recuperar** reactiva la misma ubicación con
su archivo anterior, respetando el máximo de diez lugares activos. Se permite
quitar todos los lugares; las ubicaciones iniciales no se vuelven a añadir solas.
El aviso **Deshacer** recupera la última retirada de la sesión, incluido su alias
y selección previa. La recuperación respeta el mismo límite de diez lugares.

En Tiempo, tocar un día abre sus horas locales disponibles; **Ver próximas 24 horas**
vuelve a la franja general. No cambia la temperatura de próxima hora de la portada.

Los lugares se eligen en una franja horizontal. **Ubicación actual** solicita el
permiso del navegador y actualiza la posición como máximo cada cinco minutos,
solo con la vista visible. Las coordenadas se redondean a dos decimales antes
de salir del navegador y se utilizan únicamente durante la visita; no se añaden
lugares, previsiones ni posiciones al archivo compartido. La previsión consulta
las cuatro fuentes abiertas con caché en memoria de sesión (tres horas si están
todas; reintenta fuentes fallidas a los 15 minutos y fallos totales a los dos).
Comparte los límites gratuitos persistentes existentes. Este modo no consulta
WeatherNext ni atribuye aciertos o climatología de una estación no vinculada.
Los errores de permiso o de fuentes se muestran sin inventar datos.

**Ajustes → Tu previsión** ofrece el modelo principal mediante opciones directas.
Se recuerda junto al modo automático en almacenamiento local del navegador,
sin coordenadas ni credenciales. Si la fuente preferida no está disponible se
avisa y se utiliza una disponible; la portada siempre indica la fuente efectiva.

## Configuración privada del acceso

Después de publicar HTTPS, ejecuta `python -m climate_app.auth verify-https` en el
contenedor. Comprueba el certificado y el endpoint antes de habilitar el login.
También se admite `CLIMATE_HTTPS_VERIFIED=1` en Coolify tras una verificación externa.
La contraseña se elige privadamente; no debe enviarse al chat ni guardarse en Git.
En una terminal privada del contenedor:

```sh
python -m climate_app.auth configure
```

El asistente pide usuario y contraseña dos veces, con entrada oculta. Exige 12
caracteres y guarda únicamente PBKDF2-SHA256 con sal aleatoria y 600.000 iteraciones.
No imprime contraseñas ni hashes. Se admiten también secretos de runtime:
`CLIMATE_OWNER_USERNAME` y `CLIMATE_OWNER_PASSWORD_HASH`.

El formulario ofrece **Mantener sesión en este dispositivo**, activado inicialmente.
Recuerda el acceso durante 30 días mediante una cookie opaca `__Host-climateapp_owner`,
Secure, HttpOnly y SameSite=Strict. Solo se almacena su hash en la tabla aditiva
`owner_browser_tokens` del SQLite existente, incluida en sus snapshots y backup
integral. La restauración aislada se comprueba con sesiones sintéticas; no se
exporta la base ni las credenciales de producción para probarla. Los canjes y
pruebas entre navegador y sesión web son de un solo uso y caducan en 90/60 segundos.
Las rutas conservan XSRF y validan origen; no transmiten el token de cookie a JavaScript.
La recuperación inicial lee la cookie mediante `st.context.cookies` en el servidor.
Si el navegador deja el componente en `about:blank` tras recargar, un recuperador
acotado carga el mismo HTML local en ese iframe, conservando su sandbox; nunca
reemplaza un documento activo ni admite contenido de otro origen o ruta.
Cerrar sesión revoca el acceso recordado y avisa a las otras pestañas.
Sin marcar la opción, la sesión solo pertenece a la conexión web y caduca a las
24 horas. Cambiar la credencial invalida las sesiones existentes. Se permiten cinco intentos por
ventana de 15 minutos mediante un contador persistente y atómico.
No hay contraseña por defecto ni registro abierto. Sin cuenta configurada,
WeatherNext y el consenso privado permanecen ocultos.

La gestión pública de estaciones y umbrales conserva la autorización previa del
propietario. Se pueden mantener hasta diez lugares activos dentro de las cuotas.

## Frecuencia y coste

- Open-Meteo y MET Norway: cada 3 horas; WeatherNext: cada hora si está configurado.
- Observaciones: cada 6 horas, revisando 14 días por retrasos y correcciones.
- Climatología: cinco años, actualizada mensualmente.
- Fallos: reintento desde 45 minutos después, conservando el snapshot anterior.
- Máximo de 10 lugares activos y presupuesto persistente de 8.000 unidades
  Open-Meteo/día, reservando 32 por petición. El límite publicado es 10.000.
- Las visitas a lugares guardados leen SQLite; no disparan consultas meteorológicas.
  Ubicación actual consulta bajo demanda con caché de sesión y el mismo límite.
  La búsqueda de lugares tiene caché y comparte el presupuesto. No hay paso
  automático a planes de pago.

Actualizar previsión permite una consulta manual de un lugar guardado cada quince
minutos. El límite se reclama en una transacción SQLite mediante
`task:{id}:manual-refresh`, compartido por pestañas, usuarios y procesos, también
ante fallos. Consulta solo ECMWF, ICON, GFS y MET Norway, en paralelo; las llamadas
de Open-Meteo mantienen la reserva preventiva diaria de 8.000 unidades. La caché
interna de Open-Meteo se revalida, mientras MET Norway conserva su plazo HTTP
Expires (o la caducidad previa si el registro no contiene ese encabezado).
Una fuente servida desde caché no crea un snapshot nuevo. Los fallos conservan
el archivo; las fuentes actualizadas reconstruyen únicamente el consenso público
con el método existente. No consulta Google ni observaciones/climatología, no
modifica el heartbeat del recolector y mantiene el snapshot periódico existente.

Se respeta la [licencia gratuita no comercial de Open-Meteo](https://open-meteo.com/en/terms).
MET Norway recibe identificación User-Agent con el repositorio; se respetan sus
cabeceras de caché y atribución. Unidades: °C, km/h y mm. Almacenamiento temporal en UTC.

## Evaluación y consenso

Solo se validan previsiones descargadas antes de su hora objetivo, frente a una
estación real cuya identidad y distancia se muestran. Cada variable observada de
Meteostat requiere una fuente observacional conocida: los sustitutos de modelos
se descartan, incluso cuando otras variables del mismo registro son observadas.

La antelación se mide desde la descarga: varias APIs no publican la ejecución del
modelo. Se selecciona una muestra por modelo, hora objetivo y tramo de antelación.
La pantalla permite comparar solo horas comunes. Se muestran MAE, RMSE, sesgo,
número de muestras, Brier cuando hay probabilidades y cobertura P10–P90 cuando existe.

El consenso usa únicamente errores disponibles antes de crearlo, por ubicación,
variable y plazo. Hasta tener 30 muestras en tres días por cada contribuyente
aplica pesos iguales. Los pesos inversos al MAE están regularizados; los datos
ausentes se omiten. El consenso se archiva para evaluar también sus propios aciertos.
El desacuerdo entre modelos no se presenta como intervalo probabilístico.

WeatherNext conserva por cada hora la pasada nativa de superficie y temperatura,
y las coordenadas de sus celdas. Así se distinguen las ejecuciones horarias de
48 horas de las sinópticas que cubren hasta 15 días. La procedencia se guarda en
una tabla aditiva; la tabla de puntos original sigue siendo compatible con rollback.
La evaluación lee el archivo sin repetir el JSON de cada ejecución por cada hora.

## Backup, restauración y despliegue

La ubicación automática resuelve el nombre con la API gratuita de
[BigDataCloud](https://www.bigdatacloud.com/geocoding-apis/free-reverse-geocode-to-city-api),
sin clave, desde el navegador y solo para su posición recién obtenida por GPS,
redondeada a dos decimales. No se consultan coordenadas guardadas ni ubicaciones
por IP. Caché solo en memoria de la pestaña, máximo 16 entradas y un día; errores
durante cinco minutos. La consulta tiene un timeout de cinco segundos y su fallo
no impide consultar el tiempo.

El borrado definitivo solo acepta lugares inactivos y la interfaz exige sesión
del propietario y confirmación. Una transacción elimina lugar, alias, ejecuciones,
puntos, procedencia, observaciones y estado propio del recolector/HTTP. Las
respuestas tardías no pueden recrear esos datos. La inicialización se marca en
`state.locations_seeded`: borrar el último lugar no repone los favoritos iniciales.
Los backups anteriores siguen conteniendo el archivo previo al borrado; restaurar
uno de ellos recupera su estado de entonces.

El volumen ya se incluye en `climate-app-data.tar.gz` del backup integral.
Para recuperar SQLite utiliza `forecasts-backup.sqlite3`, la copia consistente
del archivo, y no la base viva que puede haberse copiado durante una escritura.

Prueba aislada: extraer en un directorio nuevo, copiar el snapshot como
`forecasts.sqlite3`, ejecutar `PRAGMA integrity_check` y comprobar recuentos de
lugares, runs, puntos y observaciones. Nunca probar sobre el volumen de producción.
La cuenta del propietario se recupera solo desde la copia privada de su fichero.

```sh
python -m unittest discover -s tests -v
python -m climate_app.health
```

El healthcheck de imagen comprueba la web y actividad del recolector en los últimos
15 minutos. La UI muestra separadamente el estado de las fuentes. Coolify y Kuma
conservan el endpoint Streamlit existente.

Solo `oracle-production` despliega Oracle. El workflow valida antes de activar
Coolify con el mecanismo existente. Tailscale se limita a la administración, no
al acceso móvil. Para rollback, elegir el último commit saludable en Coolify y
conservar el volumen: no se borran ni transforman los datos originales.
