# Paso 1: seguimiento de vehículos con YOLO11 y ByteTrack

## Qué entrega esta implementación

El objetivo de esta etapa es relacionar las observaciones de un vehículo a lo largo del video. El programa muestra un ID, una caja y una trayectoria reciente, y guarda los resultados para revisarlos. Esta memoria temporal es la base sobre la que construiremos la clasificación de movimiento y la cola visible.

No se entrena un modelo nuevo. YOLO11n sigue detectando los objetos; ByteTrack asocia esas detecciones entre fotogramas. Las clases incluidas en esta etapa son `car`, `motorcycle`, `bus` y `truck`. No se procesan personas todavía: estamos validando el seguimiento vehicular.

El resultado aún no es una cola, un flujo, una velocidad en km/h ni un número garantizado de vehículos únicos. Tampoco se compensa el movimiento de la cámara en esta versión. Estas son las siguientes implementaciones del plan.

## Archivos y primera ejecución

Añade `seguir_video.py` y `bytetrack_tesis.yaml` a la carpeta de tu proyecto. Deben estar junto a las versiones actuales de `contar_video.py` y `contar_multizonas.py`, que el programa importa para reutilizar el manejo del tiempo y las zonas. La guía puede guardarse donde prefieras.

Conserva el entorno virtual actual. La dependencia adicional de asociación se puede instalar así:

```powershell
.\.venv\Scripts\python.exe -m pip install "lap>=0.5.12"
```

`lap` resuelve el problema de asignación entre trayectorias y detecciones. No sustituye a PyTorch ni requiere cambiar nuestra instalación CUDA.

Primera prueba, con el mismo video de 31 segundos:

```powershell
.\.venv\Scripts\python.exe .\seguir_video.py --video .\video_prueba01.mp4 --imgsz 960 --conf 0.10 --guardar-video
```

Este comando sigue vehículos en toda la imagen. Las zonas son opcionales para que la comprobación de identidad no dependa de su desalineación en el video móvil.

Si quieres superponer las zonas que ya dibujaste, añade `--zonas` seguido del nombre real de tu JSON. Por ejemplo, si se llama `zonas_video.json`:

```powershell
.\.venv\Scripts\python.exe .\seguir_video.py --video .\video_prueba01.mp4 --zonas .\zonas_video.json --imgsz 960 --conf 0.10 --guardar-video
```

Ese nombre es un ejemplo: reutiliza el archivo de zonas de tu comando anterior. El nuevo programa no abre el editor ni modifica las zonas. Si se mueve la cámara, su asignación seguirá siendo aproximada hasta incorporar la compensación.

Pulsa Q o Esc para terminar antes del final. `--max-frames 300` limita el procesamiento a 300 fotogramas. `--sin-preview` oculta la ventana, pero mantiene la exportación. `--guardar-video` crea un MP4 anotado sin audio. La salida es una carpeta nueva dentro de `resultados_seguimiento`.

## Detección e identidad son cosas diferentes

Antes obteníamos, para cada imagen, una lista como: automóvil, confianza 0,82, caja entre dos esquinas. La posición en esa lista no era una identidad. El primer automóvil del fotograma siguiente podía ser otro.

Ahora cada caja publicada por el tracker tiene además `track_id`. Por ejemplo:

| Fotograma | Objeto observado | ID devuelto |
|---|---|---:|
| 100 | Automóvil gris | 17 |
| 101 | El mismo automóvil, algo más cerca | 17 |
| 102 | El mismo automóvil | 17 |

El ID 17 es una hipótesis mantenida por el algoritmo. No es la placa del carro ni una identificación persistente entre ejecuciones. Un mismo carro puede recibir otro ID si se pierde la trayectoria, y un ID puede cambiar accidentalmente de carro cuando se cruzan u ocultan. Por eso registramos identidades para analizar continuidad, sin presentar la cantidad de IDs como flujo real.

## Cómo funciona ByteTrack

ByteTrack utiliza la evolución geométrica de las cajas y una asociación en dos etapas. Primero intenta asociar detecciones de mayor confianza; después aprovecha detecciones débiles para mantener algunas trayectorias que quedaron sin correspondencia. Esto ayuda cuando un vehículo sigue visible parcialmente, aunque su detección sea menos segura. Una detección débil aislada no se usa para abrir una trayectoria nueva con nuestra configuración.

El tracker mantiene un estado de movimiento mediante un filtro de Kalman: predice dónde podría estar una caja y corrige esa predicción al obtener otra observación. Es una estimación geométrica en imagen; no conoce la velocidad real del carro ni la dinámica de la intersección. ByteTrack no utiliza aquí un modelo adicional para reconocer la apariencia de cada vehículo y no compensa la cámara.

Referencias: [artículo original de ByteTrack](https://arxiv.org/abs/2110.06864) y [documentación oficial del seguimiento](https://docs.ultralytics.com/modes/track/).

## Por qué ahora usamos conf=0.10

Hay dos decisiones diferentes: qué candidatos llegan al tracker y cuáles tienen suficiente confianza para iniciar una identidad.

| Parámetro | Valor inicial | Papel |
|---|---:|---|
| `--conf` | 0,10 | Umbral del detector para enviar candidatos al tracker |
| `track_low_thresh` | 0,10 | Límite inferior de las detecciones usadas en la segunda asociación |
| `track_high_thresh` | 0,25 | Umbral de candidatos para la primera asociación |
| `new_track_thresh` | 0,25 | Umbral para iniciar una trayectoria nueva sin correspondencia |
| `track_buffer` | 30 | Ventana en fotogramas/actualizaciones para conservar tracks perdidos |
| `match_thresh` | 0,80 | Umbral del coste de asociación; no significa exigir IoU de 80 % |
| `fuse_score` | true | Combinar confianza y geometría al asociar |

Ejemplo: un vehículo ya seguido con ID 17 tiene confianza 0,80. En otra imagen, una obstrucción hace que su confianza baje a 0,18. Con un filtro previo de 0,25 se perdería ese candidato antes de llegar a ByteTrack. Al aceptar candidatos desde 0,10, el tracker puede usarlo si la asociación resulta consistente. La recuperación no está garantizada.

No estamos diciendo que cualquier caja con confianza 0,10 sea un vehículo confirmado. El conteo del programa usa únicamente salidas del tracker con ID. Según el estado interno, una nueva trayectoria puede necesitar confirmación posterior antes de aparecer. Estos conteos no son directamente equivalentes a los de `contar_video.py`.

El YAML mantiene los valores iniciales del seguidor de referencia. Lo copiamos en cada carpeta de resultados para saber exactamente qué configuración produjo la ejecución. Véase [configuración oficial de ByteTrack](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml).

`track_buffer=30` representa aproximadamente un segundo de video si procesamos todos los fotogramas de una fuente a 30 FPS. No representa un segundo de tiempo de procesamiento en la computadora. Si cambia la cadencia o se saltan imágenes, esa interpretación cambia.

## El núcleo del código

La llamada principal está dentro del bucle que lee el video:

```python
result = model.track(
    frame,
    persist=True,
    tracker=str(tracker_local),
    **params,
)[0]
```

`frame` es la imagen recién decodificada. `params` contiene el dispositivo, tamaño de inferencia, umbral, clases y NMS. `tracker_local` señala la copia del YAML de esta ejecución. `persist=True` mantiene el estado del seguidor entre las llamadas correspondientes al mismo video.

Cargamos el modelo una sola vez. También hacemos una inferencia de calentamiento con `predict`, sin seguimiento, antes de comenzar. Así no enviamos dos veces el primer fotograma al tracker. Después reabrimos el video y procesamos todos sus fotogramas en orden, incluso los que no contienen vehículos. Esto permite que el seguidor actualice también sus ausencias.

Cada nueva ejecución crea un modelo y un historial nuevos. No reutilizamos identidades de otro video.

### Función extraer_objetos

Lee `boxes.id`, `boxes.xyxy`, `boxes.cls` y `boxes.conf` del resultado. Traslada las cajas a CPU y convierte sus datos a valores comunes de Python para exportarlos.

Solo acepta cajas con ID. Si la librería devuelve cajas sin identidad, registra cuántas hay como diagnóstico y no inventa IDs numerándolas. El contador `salidas_sin_id` no es el número total de candidatos rechazados: algunas salidas no confirmadas ya pueden haber sido ocultadas dentro de la librería.

Conservamos la clase actual que devuelve el tracker. Si un ID cambia de `car` a `truck`, no ocultamos ese cambio: queda en el CSV y se registra en el resumen por identidad. La clase más frecuente se calcula únicamente como descripción del historial; no altera el conteo actual.

Referencia de la interfaz: [Results/Boxes](https://docs.ultralytics.com/reference/engine/results/) y [adaptación de resultados de seguimiento](https://docs.ultralytics.com/reference/trackers/track/).

### Función asignar_y_contar

Reutiliza nuestra asignación exclusiva por cobertura de caja. Para cada vehículo decide una zona, `ambiguo` o `fuera`. Sin un archivo de zonas, usa `sin_zonas` y sigue toda la imagen.

La función comienza los conteos desde cero en cada fotograma. No suma el contenido histórico del tracker. Si el ID 17 está perdido temporalmente, no se cuenta como observado en la zona durante esa ausencia. Decidir si hay que estimar su presencia mientras está oculto será una política futura explícita, con su incertidumbre.

La geometría de zona sigue siendo la misma: por defecto se usa toda la caja, cobertura mínima 0,50, margen 0,10 y tolerancia 5 píxeles originales. Las cajas de seguimiento pueden estar ajustadas por el estado del tracker; no hay que asumir que sean idénticas a las cajas de `predict` en cada imagen.

### Clase Historial

Esta clase no hace la asociación. Recibe los IDs que ByteTrack ya asignó y almacena:

- Una lista corta de puntos para dibujar el recorrido reciente.
- La primera y última observación de cada ID.
- Cuántos fotogramas publicaron ese ID.
- Cuántas veces reapareció después de una ausencia.
- Cuántas veces cambió de clase entre observaciones.

Usamos `deque(maxlen=60)` para que cada trazo conserve como máximo 60 observaciones. Los puntos antiguos salen automáticamente. Los puntos de IDs inactivos también se retiran; permanece una ficha pequeña de estadísticas por ID para la auditoría de este clip.

El punto del recorrido es el centro inferior de la caja:

```python
punto_xy = [(x1 + x2) / 2, y2]
```

Es una referencia visual aproximadamente cercana al contacto del vehículo con la vía. No equivale a una coordenada medida en metros y puede variar por oclusión o cambios en la caja. Tampoco sustituye nuestra regla de cobertura para asignar la zona.

Si un ID desaparece y regresa, borramos su trazo anterior antes de dibujar el nuevo. No dibujamos una línea entre dos observaciones separadas por imágenes sin seguimiento. La identidad y la reaparición quedan registradas, pero las posiciones intermedias no se inventan.

### Función crear_vista

Dibuja cajas, IDs y puntos con un color determinista por ID. Los colores pueden repetirse; el número es la referencia. El panel muestra los IDs presentes y los distintos IDs publicados durante la ejecución, con una indicación de que este último valor no es flujo.

Solo se reduce el tamaño de la visualización. Las coordenadas exportadas permanecen en la resolución original. Las trayectorias están en coordenadas de imagen: se moverán con la cámara hasta implementar la compensación.

## Archivos de salida e interpretación

| Archivo | Contenido y utilidad |
|---|---|
| `seguimiento.mp4` | Video anotado para pausar y revisar identidades; se crea con `--guardar-video` |
| `tracks.csv` | Una fila por ID publicado y fotograma; incluye caja, punto, confianza y zona |
| `conteos.csv` | Una fila por fotograma, incluidos los que no tienen IDs |
| `ids.csv` | Una fila por ID, con observaciones, intervalo temporal, reapariciones y cambios de clase |
| `resumen.json` | Parámetros, versiones, rendimiento, motivo de fin y diagnósticos |
| `tracker_usado.yaml` | Copia de la configuración de ByteTrack |
| `ultimo_frame.jpg` | Última visualización |
| `ultimo_frame.json` | Objetos y asignaciones de ese último fotograma |

Ejemplo conceptual de `tracks.csv`:

| frame | tiempo_s | track_id | frames_ausentes_previos |
|---:|---:|---:|---:|
| 100 | 3,337 | 17 | 0 |
| 101 | 3,370 | 17 | 0 |
| 105 | 3,504 | 17 | 3 |

El ID 17 no fue publicado en los fotogramas 102, 103 y 104. No tiene filas en ellos; `conteos.csv` sí contiene esas imágenes. La reaparición es un evento del seguimiento, no prueba de que el algoritmo recuperó al carro correcto.

En `ids.csv`, `intervalo_primera_ultima_s` incluye posibles ausencias. **No es tiempo detenido ni tiempo de espera.** Del mismo modo, muchos IDs con pocas observaciones pueden señalar fragmentación, detecciones falsas o vehículos que apenas entraron en el encuadre; no permiten calcular precisión sin revisión.

## Tiempo de video y tiempo de computadora

Las trayectorias se registran con el tiempo de la fuente. Si un video de 31 segundos tarda 55 segundos en procesarse, el movimiento corresponde a 31 segundos. Medir velocidades o esperas usando los 55 segundos sería incorrecto.

Se reutiliza `tiempo_video`: intenta leer el timestamp del archivo y, si no es utilizable, estima con los FPS nominales, dejando indicada esa estimación. Se procesan todas las imágenes en orden.

El MP4 se escribe a los FPS nominales de la fuente, aunque la ejecución vaya más lenta. Para fuentes con cadencia variable, el MP4 es una visualización a cadencia constante; el CSV conserva los tiempos usados en el análisis. El MP4 no contiene audio.

`analisis_ms` incluye detector, tracker, traslado a CPU, zonas e historial. El rendimiento del bucle incorpora también lectura, dibujo, CSV y codificación del MP4. No debemos comparar esos FPS directamente con pruebas que no guardaban video. El primer frame puede incluir la inicialización del tracker.

## Prueba breve para terminar esta etapa

Ejecuta el clip completo una vez y abre `seguimiento.mp4`. Escoge tres vehículos distinguibles: uno cercano, otro a distancia intermedia y otro que se oculte parcialmente. Sigue sus IDs durante unos segundos.

Comprueba si conservan el número, si el número pasa a otro carro o si un carro recibe otro número mientras sigue visible. Registra el tiempo aproximado del problema. Revisa también que los trazos sigan la posición del vehículo y que las ausencias no mantengan un conteo ficticio.

No necesitamos alcanzar una identidad perfecta en todo el video para avanzar. El criterio práctico de esta primera prueba es confirmar que la tubería funciona y que hay trayectorias útiles durante varios segundos. Un fallo general de IDs nos haría corregir esta etapa; casos aislados quedan documentados para la evaluación posterior.

Para revisar juntos, comparte `resumen.json` e `ids.csv`, y describe un ejemplo: «este carro mantuvo el ID» o «cambió de 12 a 39 alrededor del segundo 8». Una captura aislada no demuestra continuidad; un fragmento del MP4 puede ayudar cuando exista un problema concreto.

## Qué se verificó al entregar el código

Se comprobó la sintaxis, la interfaz de argumentos y el manejo de resultados con una secuencia simulada que incluye IDs reordenados, ausencias, reapariciones, cambios de clase y cajas sin ID. Se probaron los conteos desde cero, los CSV, los timestamps y la escritura/lectura de un MP4 de 10 fotogramas con OpenCV.

En este entorno no se ejecutó YOLO ni ByteTrack real y no hay una GPU de prueba disponible. La primera ejecución en tu equipo verificará esa integración con tu versión instalada. La revisión visual sobre el video comprobará la calidad del seguimiento, que no se deduce de las pruebas de código.

## Próximas implementaciones

La siguiente etapa estimará el movimiento del fondo para mantener una referencia coherente: deberá transformar las zonas y los puntos usados para analizar movimiento, además de coordinar esa compensación con el tracker. Después clasificaremos movimiento lento/detenido durante una ventana temporal. Finalmente contaremos vehículos en espera en las zonas de aproximación y validaremos la cola visible por carril.

La compensación no es simplemente activar un parámetro que mágicamente fija nuestras zonas. Se diseñará y verificará como un módulo propio. Tampoco confundiremos la distancia recorrida en píxeles con metros: una misma distancia física ocupa tamaños distintos según la perspectiva.
