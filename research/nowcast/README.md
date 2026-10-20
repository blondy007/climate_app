# Evaluación de lluvia a corto plazo

Investigación separada de ClimateApp. **No está aprobada para uso público** y no
se importa desde la aplicación. No requiere Google, claves, cuentas de pago ni
modificaciones en `requirements.txt` de la app. El objetivo es comprobar si puede
existir una previsión local creíble antes de construir su interfaz.

## Datos y protocolo congelado

- Centros públicos de Los Narejos / Los Alcázares (37.744, -0.850) y Torrejón
  (40.456, -3.475). Nunca se lee la base de lugares ni GPS del usuario.
- EUMETNET OPERA DBZH: reflectividad máxima, 1 km / 5 minutos. RATE: estimación
  de lluvia superficial, 2 km / 15 minutos. ACRR: acumulación previa de una hora,
  usada únicamente para localizar episodios. Licencia de compuestos CC BY 4.0.
- Se conservan ventanas de 256 km, unidades físicas, máscara de cobertura,
  `undetect`, fecha interna y periodo de observación del HDF5, geometría,
  primera descarga, fecha de publicación y hashes del original y del recorte.
- `opera-lk-rate-v1`: movimiento Lucas–Kanade de tres DBZH consecutivas mediante
  pySTEPS, transporte semilagrangiano de RATE y conversión física entre mallas.
  Parámetros originales, umbral de lluvia 0,1 mm/h, antelaciones 15/30/60 minutos.
  No se ajustó el método tras examinar el día reservado.
- En replay, un fichero solo está disponible después de su `LastModified` S3
  más dos minutos. Se incluye toda la antigüedad de RATE al extrapolar, no solo
  la antelación solicitada. En directo se usa la primera descarga efectiva y
  jamás se retrofecha una predicción.
- Se rechazan observaciones viejas, cobertura local insuficiente y movimientos
  extremos. Un rechazo o un píxel ausente nunca se interpreta como tiempo seco.
- Referencia sencilla: persistir la última intensidad conocida. CSI = aciertos /
  (aciertos + fallos + falsas alarmas); los periodos secos no inflan este índice.
  Comparar sobre pares idénticos y publicar también disponibilidad y rechazos.
- Desarrollo: 17–18 octubre de 2026. Comprobación reservada: 19 octubre, hasta
  las 22:45 UTC de emisión. Ver [resultado inicial](results/2026-10-20.md).

El radar posterior es una referencia dependiente, **no un pluviómetro**. Tres
días y predicciones solapadas no son episodios independientes. La cadencia de
RATE no valida un inicio al minuto ni una precisión de cinco minutos.

## Ejecución reproducible

El motor usa dependencias científicas propias y se ha verificado en Linux ARM64.
En Windows, pySTEPS necesita compilación nativa; no instalar compiladores ni
mezclar dependencias con las de la app para evitarlo.

Desde la raíz del repositorio:

```sh
docker build -f research/nowcast/Dockerfile -t climate-radar-lab:research .
docker run --rm --network none climate-radar-lab:research python -m unittest discover -s research/nowcast/tests -v
```

Para descargar un estudio de hasta cuatro días, montar un directorio de datos
vacío, privado y escribible por UID 1000, con los mismos límites de recursos del
colector. El subcomando es:

```sh
python -m research.nowcast.backfill --root /data --start 2026-10-16T23:30 --end 2026-10-20T00:00
python -m research.nowcast.replay --root /data --start 2026-10-17T00:00 --end 2026-10-19T23:00 --validation-start 2026-10-19T00:00
```

No ejecutar dos escritores del mismo backfill. El colector tiene bloqueo de
escritor propio. El replay puede ejecutarse sin red sobre recortes ya guardados;
solo modifica sus tres ficheros `retrospective-*.json`.

## Experimento activo en Oracle

- Contenedor: `climate-radar-capture-20261020`.
- Dependencias: imagen `climate-radar-lab:c852a5f125e3`.
- Código de captura congelado: `/home/ubuntu/climate-radar-lab/code-b107f760a60b/research`,
  huella `b107f760a60b802d4de998895a91c151c25180dc1185a2e455fb21b24ff9aa0b`.
- Hash del motor: `5e8e4205924afd7c9514b9de5d851cd63ed53e90ec87b9046867ac68eacd952e`.
  La imagen aporta dependencias; el código indicado está montado en lectura.
  El lector de pluviómetro se añadió después, sin cambiar el motor ni la captura.
- Datos: subdirectorio `radar-lab` del volumen existente
  `xk53thwe7wzp3wah1qxvttvg-climate-app-data`. Único directorio de datos montado;
  el contenedor no puede acceder a SQLite, la cuenta ni otros archivos de la app.
- Inicio: 20/10/2026 20:00 UTC. Fin absoluto: **27/10/2026 20:00:15 UTC**,
  o antes si alcanza un límite. Reiniciar no renueva el plazo ni el presupuesto.
- Ciclo de 5 minutos, captura de las últimas 2 horas. 0,5 CPU, 768 MiB, 128 procesos,
  raíz en lectura, `/tmp` de 64 MiB, sin capacidades ni privilegios nuevos, UID 1000.
- Sin puertos, ruta web ni cuenta nueva. Solo solicitudes HTTPS a datos abiertos.
  No es una nueva aplicación ni un panel que añadir al catálogo público.
- Máximo de 750 MiB conservados, sin purgas automáticas. Presupuesto de 12 GB
  para ficheros de radar, reservando de antemano sus tres posibles intentos;
  puede agotarse antes de siete días. Los catálogos pequeños son adicionales.
- Logs rotados: 2 × 5 MiB. `on-failure:5`; no reanudar indefinidamente después
  de vencer. Un reinicio del host puede requerir reanudar el contenedor existente;
  comprobar antes su plazo. No reconstruir una sesión con presupuesto vacío.
- Healthcheck de progreso real, cada cinco minutos. `live-status.json` conserva
  el último ciclo y los errores. `prospective/*.json` son predicciones originales
  con creación exclusiva, hora real de emisión y de escritura. `prospective-report.json`
  se recalcula después con RATE posterior, sin interpolar la referencia.
- Las salidas cada 5 minutos son experimentales; solo los objetivos que coinciden
  con un RATE cada 15 minutos tienen verificación directa. Las métricas del informe
  de directo se deben volver a comprobar sobre pares idénticos antes de concluir.
- Heartbeat de Codex `validar-lluvia-de-climateapp`: dos revisiones diarias,
  14 ejecuciones como máximo; silencioso sin novedades relevantes. La captura
  funciona en la VPS aunque el escritorio esté cerrado; la revisión local depende
  de que el entorno de Codex esté disponible.

Para comprobar sin modificar la captura:

```sh
sudo docker inspect --format '{{json .State}}' climate-radar-capture-20261020
sudo docker logs --since 15m --tail 10 climate-radar-capture-20261020
sudo docker exec climate-radar-capture-20261020 python -m research.nowcast.health --root /data
```

`deployment.json` guarda imagen, huellas, ubicación del código y plazo exactos.
Consultar solo estos archivos de investigación; nunca exportar el volumen entero.
No cambiar el modelo conservando el mismo identificador ni sustituir predicciones
ya emitidas. Cualquier versión nueva necesita otro identificador y episodios futuros.

## Pluviómetro independiente

`gauge.py` recupera únicamente la estación pública CHS `06A18P01`, desembocadura
de la Rambla del Albujón (37.716015, -0.861255), a unos 3,3 km del centro estudiado.
Conserva HTML original, hash, registros cada 5 minutos, calidad, procedencia y
ausencias. Valida milímetros, tipo de serie, regularidad y suma contra el acumulado
oficial. La gráfica codifica el reloj local como epoch aparente: se convierte desde
Europe/Madrid y se comprueba contra los epochs internos reales. Las horas ambiguas
por cambio de hora no se convierten silenciosamente.

Los datos son provisionales. No atribuir sus acumulados al domicilio o al centro
urbano. Antes de puntuar inicio/fin, confirmar convención del intervalo, resolución
del pluviómetro y comparar predicción en **la coordenada del sensor**. Por ahora
solo confirma episodios de lluvia cercanos; no se ha usado para aprobar el método.
No se recopila automáticamente con el contenedor congelado: el heartbeat recupera
el archivo reciente en peticiones de hasta cuatro días y conserva las capturas.

## Persistencia, recuperación y decisión

El backup integral ya archiva todo el volumen de ClimateApp en
`climate-app-data.tar.gz`, incluido `radar-lab`. Se probó una restauración aislada
del subconjunto de investigación, verificando hashes de recortes y de la captura
CHS. Es una prueba parcial; no sustituye la recuperación completa de la plataforma.
El código y las dependencias se reconstruyen desde Git; la carpeta de construcción
del host no contiene estado científico único. No borrar los datos al finalizar.

La siguiente evaluación debe comprobar disponibilidad durante lluvia, falsos
avisos, detección de episodios independientes y comparación de inicio/fin con
observaciones en tierra. Si falla un lugar u horizonte, no publicarlo allí. Si no
se logra fiabilidad suficiente con datos gratuitos, descartar esta versión.

## Fuentes

- [EUMETNET: acceso a datos abiertos](https://eumetnet.github.io/openradardata-documentation/2-ORD-API-discovering-and-accessing-data/).
- [EUMETNET OPERA: productos](https://www.eumetnet.eu/observations/opera-radar-animation/).
- [pySTEPS: extrapolación semilagrangiana](https://pysteps.readthedocs.io/en/stable/generated/pysteps.extrapolation.semilagrangian.extrapolate.html).
- [Confederación Hidrográfica del Segura: condiciones y carácter provisional](https://www.chsegura.es/es/cuenca/redes-de-control/saih/ivisor/).
- [CHS: estación 06A18P01](https://saihweb.chsegura.es/apps/iVisor/visor_variable.php?punto=06A18P01).
