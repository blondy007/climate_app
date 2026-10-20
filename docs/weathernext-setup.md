# WeatherNext 3 a coste cero

El adaptador está implementado. No se considera finalizado hasta comprobar datos
reales en Oracle con una identidad autorizada por Google y sin facturación.

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

Tras recibir el acceso hay que verificar: identidad autorizada, billing deshabilitado,
esquema real, celdas de ambas ubicaciones, percentiles, unidades y lluvia horaria,
ejecuciones recientes y cobertura de 15 días, persistencia de snapshots y aislamiento
entre sesiones públicas y autenticadas. Solo entonces se puede cerrar el objetivo.

Fuentes: [acceso](https://developers.google.com/weathernext/guides/access-forecast),
[variables](https://developers.google.com/weathernext/guides/models),
[términos](https://storage.googleapis.com/weathernext-public/terms-of-use.pdf).
