# Revisión y contrato de percepción parcial

Revisión de base: commit `d4845d846a26cffa8b269f27d74d4993a494983e`.

## Hallazgos

1. `seguir_video.py` y `seguir_compensado.py` convierten `--video` a `Path`,
   requieren `is_file()` y reabren el archivo. No admitían RTSP directamente.
2. `main.py` es una prueba de imagen con YOLO26, no el pipeline de tesis con
   YOLO11, seguimiento y cola. La entrada nueva es `ejecutar_vision.py`.
3. `estado_cola.py` estima movimiento en píxeles/s normalizado por altura de
   caja. No convierte velocidad a m/s; `cola_visible_proxy` no equivale a
   `getLastStepHaltingNumber` de SUMO, cuyo umbral es 0.1 m/s.
4. El historial guarda fichas de todos los IDs y el ejecutor compensado acumula
   listas de tiempos/rendimientos para el resumen. Adecuado para clips finitos,
   pero crecería durante una cámara permanente.
5. Los JSON de zonas tienen resolución/encuadre específicos. El archivo
   `zonas_compensadas_cigarral.json`, por ejemplo, corresponde a 1280x720 y
   contiene una zona Trinidad→Boyera; la Tapo observada emitía 2304x1296.
6. `ultimo_frame.json` se escribe al finalizar los scripts anteriores, no era
   una interfaz periódica para un consumidor concurrente.
7. Hay observaciones experimentales de ~26 MB versionadas. Se mantienen para
   reproducibilidad; los nuevos resultados se ignoran y el historial rota.

## Implementación añadida

- Detector YOLO único; inferencia alternada, no dos modelos cargados en GPU.
- Captura concurrente por fuente; cola RTSP de un frame, archivo con backpressure.
- ByteTrack independiente; IDs identificados por sesión + fuente + época + ID.
  Se protege el contador global en versiones antiguas al reiniciar otro tracker.
- Reutilización de asignación geométrica, compensación y `EstimadorCola` existentes.
- Historias acotadas; reinicio por reconexión, tiempo no creciente o hueco >1 s.
- Estado atómico, caducidad, retención limitada y separación vivo/reproducción.
- Editor reutilizado para calibrar polígonos sobre cada fuente.

## Contrato `traffic-vision/1`

Un estado se publica cada 1 s por defecto. No es una llamada al agente ni una
orden de fase. La unión con RL queda pendiente del contrato de la política.

| Campo | Semántica |
|---|---|
| `sesion_id`, `generado_utc` | Identidad de corrida y frescura del publicador |
| `fuentes.<id>.tiempo_fuente_s` | Archivo: PTS con respaldo índice/FPS; RTSP: tiempo monotónico de recepción |
| `fuentes.<id>.edad_recepcion_s` | Tiempo desde recepción local del último frame analizado; no latencia completa de cámara |
| `fuentes.<id>.vigente` | Conexión/estado y edad dentro del umbral; no garantía de exactitud |
| `fuentes.<id>.dominio` | `en_vivo` o `reproduccion` |
| `accesos` | Solo mediciones en vivo de accesos configurados |
| `observados` | Detecciones presentes asignadas a la región, no flujo |
| `cola_visible_proxy` | Observados con ID y movimiento reducido persistente |
| `detenidos`, `lentos` | Clasificación con umbrales relativos a altura de caja |
| `indeterminados` | Falta de ID/historia suficiente; no equivale a movimiento |
| `cobertura_clasificacion` | Fracción con velocidad relativa calculable; no precisión ni confianza del detector |
| `fraccion_visible` | Parte geométrica del polígono dentro de imagen; no visibilidad tras oclusiones |
| `max_episodio_reducido_s` | Máximo episodio continuo observado entre vehículos en cola proxy; no espera total |
| `mascara_observacion` | Región con medida actual válida geométricamente; no valida todas sus métricas |
| `velocidad_media_m_s`, `longitud_cola_m`, `espera_total_s`, `flujo_veh_h` | `null`: no se estiman con este método |
| `listo_para_control` | `false`: falta integración/calibración con la política y estado semafórico |

Una región válida sin detecciones tiene `observados=0`. Una fuente caída,
alineación inválida o acceso sin cámara tiene `observados=null`. No rellenar los
nulos con cero sin una máscara explícita y un agente entrenado con esa convención.
Los ceros de cola con `indeterminados>0` son solo vehículos **confirmados** en
cola: no prueban ausencia de cola. Son una cota observada, no la cola real completa.

## Cobertura de esta tesis

Orden lógico fijo:

1. `boyera_trinidad`
2. `trinidad_boyera`
3. `cigarral`
4. `av1`
5. `av2`

Av. 1 y Av. 2 permanecen sin cámara (`null`, máscara falsa). Tampoco inferimos
sus vehículos a partir del paso por Cigarral: entradas/salidas y tiempos de
recorrido harían falta para una estimación distinta, que debería etiquetarse
como estimada. Ver un tramo principal no proporciona la cola completa de todas
las intersecciones del corredor ni los movimientos de giro por separado.

El agente puede controlar un corredor parcialmente observado, pero debe entrenarse
así: no puede recibir en simulación las colas perfectas de Av. 1/Av. 2 y esperar
encontrarlas al desplegar. Historial/recurrente, planes coordinados o un estimador
son decisiones del controlador, no información inventada por la cámara.
SUMO puede usar sus datos completos para recompensa/evaluación durante entrenamiento,
aunque la observación de la política se restrinja a lo medible en campo.

La fase actual, tiempo de verde/transición, solicitudes peatonales y restricciones
vienen del controlador. Un algoritmo de seguridad aplica verdes mínimos/máximos,
amarillo, todo rojo e incompatibilidades independientemente de la política.
Este repositorio publica percepción; no implementa todavía esas acciones.

Para comparar con SUMO hace falta definir regiones equivalentes, calibrar la
geometría (si se requieren metros/m/s), y medir el error de conteo/cola. Cambiar
el nombre de un campo no convierte un proxy en una medida equivalente.

## Fuentes técnicas

- https://docs.ultralytics.com/reference/trackers/byte_tracker/
- https://sumo.dlr.de/docs/TraCI/Lane_Value_Retrieval.html
- https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html

## Validación realizada en esta entrega

- 20 pruebas automatizadas: separación vivo/reproducción, cero vs desconocido,
  caducidad, desconexión, configuración, escritura atómica, captura con cola
  acotada, archivo AVI real y reconexión simulada.
- Integración con ByteTrack real en Ultralytics 8.4.197, detecciones controladas:
  dos historiales independientes, reinicio de una fuente, huecos temporales y dibujo.
- Prueba completa con YOLO11n real en CPU: dos archivos de 40 frames cada uno,
  procesados hasta el frame 39, observaciones exportadas y estado invalidado al salir.
  Se usó un clip sintético repetido de bus.jpg para verificar ejecución, no precisión.
- Entorno de prueba: Linux x86_64, Python 3.12, torch CPU 2.15.0, OpenCV headless
  5.0.0, Ultralytics 8.4.197. No es una medición del Jetson.
- Pendientes en equipo: Tapo RTSP real, vista gráfica, rendimiento y consumo
  sostenidos en Orin, calibración de polígonos y validación de colas contra conteo manual.
