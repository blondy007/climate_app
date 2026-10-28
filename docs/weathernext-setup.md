# WeatherNext 3 a coste cero

El adaptador está implementado. No se considera finalizado hasta comprobar datos
reales en Oracle con una identidad autorizada por Google y sin facturación.

## Estado revisado · 28/10/2026

Google ha confirmado por correo la aprobación de la solicitud del propietario.
La aprobación se aplica a la cuenta indicada en el formulario y a las tres vías
de acceso (BigQuery, Earth Engine y Cloud Storage); no entrega una clave API ni
autoriza automáticamente otras identidades. No es necesario repetir la solicitud.

El propietario creó `climateapp-weathernext` sin facturación, enlazó el dataset
`weathernext_3` en US y autenticó ADC con la cuenta aprobada. Se comprobó
`billingEnabled == false` por API y el esquema requerido de las dos tablas. Solo
se habilitaron Cloud Billing API (consulta del estado) y Earth Engine API; ninguna
cuenta de facturación se vinculó.

La comprobación real detectó un bloqueo del adaptador existente: las tablas
enlazadas son vistas y el dry run estima 6 542 787 139 200 bytes; restringir ambas
a una sola inicialización todavía estima 2 665 811 088 000 bytes. No se ejecutaron
estas consultas ni se amplió el límite de 2 GiB. En tablas agrupadas el estimador
puede ser una cota superior, pero
[BigQuery comprueba esa cota contra maximum_bytes_billed antes de ejecutar](https://docs.cloud.google.com/bigquery/docs/best-practices-costs).

La cuenta también accede a los metadatos del bucket GCS statistics sin Requester
Pays. Sin embargo, los chunks reales contienen un mapa mundial completo por
variable/hora (1801 × 3600 y 3601 × 7200 floats). Seleccionar un punto no evita
descargar el chunk entero; no se descargaron datos meteorológicos por esta vía.

Siguiente vía: [Earth Engine](https://developers.google.com/weathernext/guides/earth-engine),
con reducción en el servidor al punto solicitado. API habilitada, pero devuelve
que el proyecto no está registrado. El propietario debe completar el
[registro de uso no comercial](https://console.cloud.google.com/earth-engine/configuration?project=climateapp-weathernext)
con su uso real. El nivel Community no exige billing; no seleccionar planes de pago.
Después hay que implementar/verificar el adaptador; todavía no se ha validado
ninguna previsión real.

El propietario autorizó copiar ADC a Oracle, montaje exclusivo de ClimateApp,
configuración y reinicio para probarlo. Archivo preparado en el host bajo
`/data/coolify/applications/xk53thwe7wzp3wah1qxvttvg/secrets/google-adc.json`, root
0600/directorio 0700, cubierto por el backup integral de `/data/coolify`. Copia y
restauración aislada comprobadas sin mostrar contenido. Aún no está montado ni
activado en runtime; el contenedor no se reinició para activar una vía bloqueada.

Ruta que aprovecha el adaptador existente:

1. Usar la cuenta aprobada para acceder al proyecto dedicado de BigQuery sandbox.
2. Enlazar el dataset de [Analytics Hub desde la guía oficial de BigQuery](https://developers.google.com/weathernext/guides/bigquery).
3. Preparar autenticación privada para el recolector con una identidad aprobada;
   revisar cualquier autorización OAuth o nueva cuenta de servicio antes de concederla.
4. Configurar las variables indicadas abajo y comprobar billing deshabilitado,
   acceso a ambas tablas y dry run. Solo entonces probar una consulta acotada con
   datos reales y habilitar la recopilación; la vista sigue siendo personal.

Alternativa documentada si el enlace de BigQuery no funciona sin billing:
el [bucket de estadísticas Zarr](https://developers.google.com/weathernext/guides/gcs)
`weathernext3_statistics_spatial` tiene Requester Pays desactivado y no requiere
cabeceras de facturación. Esta vía necesitaría un adaptador adicional y comprobar
acceso y volumen de descarga antes de activarla. No usar el bucket de ensemble
completo `weathernext3_spatial`, que sí tiene Requester Pays activado.

La aprobación no habilita WeatherNext en la aplicación por sí sola. Mantener el
adaptador desactivado hasta comprobar una vía viable y una previsión real.

## Solicitud del propietario

1. Abre el [formulario oficial](https://docs.google.com/forms/d/e/1FAIpQLSeCf1JY8G78UDWzbm0ly9kJxfSjUIJT5WyMR_HiNqCm-IHIBg/viewform).
2. Introduce tu nombre y la cuenta Google que utilizarás en Google Cloud.
3. País: España. Organización: `N/A` para uso individual; selecciona también
   la opción de individuo sin entidad en el país de la organización.
4. Selecciona **WeatherNext 3 on BigQuery**. La aprobación incluye las demás superficies.
5. Si ya tienes una cuenta de servicio dedicada, solicita también esa identidad
   en el campo opcional adicional. No incluyas claves privadas.
6. Revisa personalmente los términos y envía el formulario. Google indica un
   plazo habitual de 5–7 días laborables y no garantiza aprobar toda solicitud.

Descripción sugerida si se solicita:

> Personal, non-commercial weather application for locations in Spain. I will
> compare WeatherNext 3 forecasts with other open forecasts and observed weather
> station measurements, keep an archive and evaluate accuracy by forecast horizon.
> Real-time WeatherNext data will be restricted to my authenticated personal access.

## Proyecto y credenciales

- Usa un proyecto dedicado de [BigQuery sandbox](https://docs.cloud.google.com/bigquery/docs/sandbox),
  **sin cuenta de facturación vinculada**. No actives tarjeta, créditos promocionales
  ni pruebas de pago para esta aplicación.
- Después de la aprobación, sigue la [guía de BigQuery](https://developers.google.com/weathernext/guides/bigquery)
  para enlazar las tablas de Analytics Hub en un dataset de ese proyecto.
- La identidad usada en Oracle debe estar autorizada: aprobar tu cuenta personal
  no aprueba automáticamente una cuenta de servicio.
- Monta Application Default Credentials como un fichero privado de runtime en
  Coolify. Nunca envíes su contenido al chat ni lo subas al repositorio.
- La identidad necesita permisos mínimos para leer las tablas, ejecutar consultas
  en el proyecto y consultar su estado de facturación.
- Si Google exige billing para esa vía, no se habilita; se conserva pendiente y
  se busca otra opción gratuita verificable.

Variables de runtime (solo nombres y ejemplos sin secretos):

```text
WEATHERNEXT_PROJECT=proyecto-sin-billing
WEATHERNEXT_DATASET=proyecto-sin-billing.dataset_enlazado
WEATHERNEXT_LOCATION=US
GOOGLE_APPLICATION_CREDENTIALS=/ruta/privada/credenciales.json
```

La región debe coincidir con el dataset. El cliente comprueba `billingEnabled == false`
antes de cada consulta, hace dry run y limita cada trabajo a 2 GiB. Reserva un
presupuesto máximo de 900 GiB/mes y devuelve los bytes no utilizados al terminar.
Las cuotas limitan uso; la garantía de no cobrar es mantener billing deshabilitado.
Los errores no muestran credenciales, rutas privadas ni identificadores de cuentas.

## Acceso y comprobación final

La web mantiene su dominio y utiliza HTTPS. El propietario inicia sesión desde
el móvil sin Tailscale. Los visitantes ven las fuentes abiertas y un consenso que
excluye WeatherNext; el propietario ve también WeatherNext y el consenso completo.
No hay registro abierto ni API pública de datos WeatherNext.
El login ofrece recordar este dispositivo durante 30 días, incluso tras reabrir
la web. Cerrar sesión revoca ese acceso. La ubicación automática utiliza solo
fuentes abiertas; WeatherNext permanece asociado a los lugares guardados.

Tras recibir el acceso hay que verificar: identidad autorizada, billing deshabilitado,
esquema real, celdas de ambas ubicaciones, percentiles, unidades y lluvia horaria,
ejecuciones recientes y cobertura de 15 días, persistencia de snapshots y aislamiento
entre sesiones públicas y autenticadas. Solo entonces se puede cerrar el objetivo.

Fuentes: [acceso](https://developers.google.com/weathernext/guides/access-forecast),
[variables](https://developers.google.com/weathernext/guides/models),
[términos](https://storage.googleapis.com/weathernext-public/terms-of-use.pdf).
