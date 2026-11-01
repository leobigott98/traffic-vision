# Compensación de cámara y observaciones sin ID

Esta etapa mantiene las zonas asociadas a la calle y expresa las posiciones de los vehículos en una referencia común. También conserva detecciones válidas que todavía no tienen identidad de seguimiento. El resultado sigue siendo un **conteo de vehículos visibles**, previo a la estimación de cola.

## 1. Un vehículo detenido sí puede tener seguimiento

YOLO detecta apariencia en cada imagen; no exige movimiento. ByteTrack asocia las detecciones entre fotogramas y puede mantener un ID aunque el vehículo permanezca aproximadamente en el mismo lugar. Una trayectoria dibujada puede verse como un punto o una línea muy corta: eso no significa que el vehículo carezca de seguimiento.

Conviene distinguir tres situaciones:

| Situación | Qué conocemos | Tratamiento actual |
|---|---|---|
| Detección con ID, desplazamiento pequeño | Posición actual e historial asociado | Se cuenta; su historial podrá servir para evaluar detención |
| Detección suficientemente confiable sin ID | Presencia en este fotograma | Se cuenta como observación sin ID; todavía no tiene tiempo individual de espera |
| Ninguna detección | No hay observación del vehículo en ese fotograma | No se inventa un vehículo ni se clasifica como detenido |

La ausencia de ID puede deberse a una incorporación reciente, asociaciones fallidas o detecciones intermitentes. No prueba que el vehículo esté detenido. La ausencia de detección también puede deberse a oclusión, tamaño pequeño o contraste insuficiente.

En esta versión, `--conf 0.10` conserva candidatos para las dos etapas de asociación de ByteTrack. `--conf-observacion 0.25` exige más confianza a una detección que el tracker no ha asociado. Una detección débil puede aparecer si quedó asociada a un ID; una detección débil sin asociación no se agrega al conteo.

La unión se hace por **índice de detección**, no sumando dos listas independientes: cada candidato aporta como máximo una observación, con ID o sin ID. Esto evita duplicarlo por el hecho de aparecer en ambos pasos. No garantiza que YOLO nunca produzca dos cajas para un mismo vehículo; la NMS sigue cumpliendo su función.

Estos totales pueden superar los del script anterior, que publicaba únicamente cajas con ID. Esa diferencia es esperable y queda separada en las salidas.

## 2. Referencia fija: qué estamos compensando

Elegimos el primer fotograma del archivo como referencia. Las zonas se dibujan allí una sola vez. Para cada fotograma posterior estimamos una transformación que lleva sus coordenadas a esa referencia.

Si `p_actual` es un punto de la imagen actual:

```text
p_referencia = A · p_actual + t
```

`A` es una matriz de 2 × 2 que representa una rotación y una escala uniforme; `t` contiene el desplazamiento horizontal y vertical. Juntas forman una matriz de 2 × 3. Este modelo, llamado transformación de similitud, tiene cuatro grados de libertad.

Ejemplo: un poste aparece inicialmente en x=500 y, después de moverse la cámara, en x=470. Si todo el cambio fuera una traslación horizontal, sumaríamos 30 píxeles a las posiciones actuales para expresarlas en la referencia. Un carro estacionado sufriría ese mismo desplazamiento aparente y recuperaría aproximadamente su posición original. Un carro que avanzó conservaría el desplazamiento adicional debido a su propio movimiento.

No generamos un video estabilizado para volver a detectar sobre él. YOLO recibe cada fotograma original. Después transformamos las detecciones y hacemos el seguimiento en coordenadas de referencia. Para dibujar las zonas y trayectorias sobre el fotograma actual aplicamos la transformación inversa.

**Por eso los polígonos se moverán en la pantalla cuando se mueva la cámara:** deben seguir las mismas partes de la calle, aunque esas partes cambien de posición en la imagen.

Cada imagen se compara directamente con el fotograma de referencia. Así evitamos acumular pequeños errores de transformaciones sucesivas. A cambio, necesitamos que siga siendo visible suficiente fondo de la referencia.

## 3. Cómo encontramos el movimiento de cámara

El módulo `compensacion_camara.py` aplica esta secuencia:

1. Selecciona una región de fondo en la referencia.
2. Excluye las cajas de vehículos detectados, con un pequeño margen.
3. Extrae puntos característicos y descriptores ORB.
4. Busca correspondencias entre el fotograma actual y la referencia.
5. Descarta emparejamientos poco distintivos y repeticiones de un mismo punto de referencia.
6. Estima la transformación con RANSAC.
7. Acepta o rechaza el resultado según su calidad geométrica.

Un punto característico es una ubicación de la imagen que resulta relativamente fácil de reconocer, por ejemplo una esquina con textura. El descriptor resume su entorno para poder buscar una correspondencia. No es un vehículo ni una identidad de seguimiento.

RANSAC ensaya transformaciones apoyadas en subconjuntos de correspondencias y busca una explicación coherente para muchas de ellas. Las correspondencias compatibles se denominan *inliers*. Esto permite tolerar emparejamientos equivocados, pero no garantiza que el movimiento dominante corresponda siempre al fondo correcto.

Para reducir el costo, ORB analiza una imagen de hasta 960 píxeles de ancho. Sus coordenadas se convierten nuevamente a la resolución original antes de estimar la transformación. Los errores registrados y la tolerancia de las zonas se expresan en píxeles de esa resolución original.

Por defecto se usa el 40 % superior del primer fotograma como región candidata de fondo. Es una elección inicial para este encuadre; **no es una regla universal para cualquier video**. La imagen `referencia_fondo.jpg` muestra iluminada esa región y oscurecido el resto. Las cajas de vehículos se excluyen internamente aunque no aparezcan tachadas en esa imagen.

Si el área incluye mucho tráfico, vegetación que se mueve o superficies sin textura, define polígonos de fondo con `--fondo` y `--editar-fondo`. Prefiere estructuras fijas distribuidas por la imagen: edificios, bordes de infraestructura o marcas de pavimento visibles. Evita seleccionar únicamente el poste que está ocultando vehículos: una región pequeña no ofrece suficiente soporte para todo el encuadre. Los vehículos que YOLO no detecta tampoco se excluyen automáticamente; seleccionar bien el fondo ayuda a reducir ese problema.

## 4. Controles de calidad y límites

Los valores iniciales del módulo exigen al menos 30 correspondencias y 25 inliers, una proporción de inliers de 0,45 o mayor y error mediano de ajuste no superior a 2 píxeles. También se revisa que los puntos estén suficientemente distribuidos, que el cambio de escala y rotación sea moderado y que permanezca en común al menos el 60 % del área de referencia.

Estos límites son heurísticos y se guardan en `resumen.json`; todavía no están calibrados para todas las cámaras. Un error pequeño en los puntos del fondo no demuestra que toda la calzada esté igualmente alineada.

La similitud sirve como primera aproximación para vibraciones y pequeños cambios de encuadre. No representa cualquier movimiento de cámara. Cambios importantes de perspectiva, desplazamientos laterales con paralaje, grandes giros de una cámara PTZ o una escena completamente distinta pueden requerir otro modelo o una nueva referencia. Aunque una transformación supere los controles, revisa que las zonas sigan coincidiendo con sus carriles.

Cuando se rechaza la alineación:

- Las detecciones suficientemente confiables siguen disponibles en la imagen actual, sin ID.
- El conteo por zona se marca `N/D`; se exporta como `null` en JSON y campo vacío en CSV.
- No se publican posiciones de referencia ni trayectorias basadas en una transformación dudosa.
- Se reinicia ByteTrack una vez al entrar en el fallo. Cuando se recupera la alineación, los nuevos IDs públicos no reutilizan los anteriores.

Ese reinicio sacrifica continuidad para no fabricar desplazamientos o tiempos de espera durante una alineación inválida. Puede aumentar los IDs distintos; estos nunca deben interpretarse como número de vehículos que atravesaron la escena.

Además se calcula qué fracción de cada zona sigue dentro del campo visual. El panel marca `PARCIAL` cuando es inferior a 0,95. El conteo corresponde a lo observado, aunque la zona esté parcialmente fuera de imagen. Ese aviso será importante al interpretar la cola.

## 5. Integración con ByteTrack y zonas

`seguir_compensado.py` usa `model.predict()` para obtener todas las detecciones candidatas, transforma sus cajas y llama directamente a `BYTETracker.update()`. Esto permite insertar la compensación antes de la asociación y conservar las detecciones no asociadas.

ByteTrack recibe la caja rectangular que envuelve las cuatro esquinas transformadas. Para asignar una zona usamos el polígono de esas cuatro esquinas, evitando que la envolvente rectangular aumente artificialmente el área cuando hay rotación. Se conservan las reglas previas: cobertura mínima, margen entre las mejores zonas, tolerancia y asignación exclusiva.

La trayectoria utiliza el centro inferior de la caja original transformado a la referencia. Es un punto de apoyo práctico, pero puede oscilar al cambiar la caja detectada. No es una medición exacta del contacto de los neumáticos con el suelo. Medir velocidad física requeriría además calibración; todavía trabajamos en píxeles.

La compensación puede ayudar a la asociación cuando se mueve la cámara. No reconstruye vehículos ocultos ni garantiza conservar el ID al pasar detrás del poste. Ese problema sigue siendo de oclusión y asociación temporal.

## 6. Archivos e instrucciones

Copia los dos archivos nuevos en la misma carpeta del proyecto:

- `compensacion_camara.py`: registro geométrico.
- `seguir_compensado.py`: detección, compensación, seguimiento, zonas y exportación.

Deben estar disponibles también los archivos anteriores `seguir_video.py`, `contar_video.py`, `contar_multizonas.py` y `bytetrack_tesis.yaml`. Conserva sus nombres. No hace falta cambiar las dependencias que ya instalamos para el seguimiento.

Primera ejecución, desde PowerShell en `tesis-vision`:

```powershell
.\.venv\Scripts\python.exe .\seguir_compensado.py --video .\video_prueba01.mp4 --zonas .\zonas_compensadas.json --editar-zonas --guardar-video
```

`--editar-zonas` abre el editor sobre el primer fotograma. Usa un archivo nuevo de zonas: un JSON dibujado sobre otro fotograma no se vuelve correcto por tener la misma resolución. Los polígonos admiten varios vértices; clic cerca del primero cierra el polígono. Completa nombre y acceso en la terminal. `S` guarda; botón derecho deshace, `R` reinicia el polígono actual, `U` elimina la última zona y `Esc` cancela. Sigue también las indicaciones del editor.

Para repetir con las mismas zonas y el mismo video:

```powershell
.\.venv\Scripts\python.exe .\seguir_compensado.py --video .\video_prueba01.mp4 --zonas .\zonas_compensadas.json --guardar-video
```

Si necesitas definir el fondo manualmente:

```powershell
.\.venv\Scripts\python.exe .\seguir_compensado.py --video .\video_prueba01.mp4 --zonas .\zonas_compensadas.json --fondo .\fondo_camara.json --editar-fondo --guardar-video
```

Reutilizamos el mismo editor para el fondo. Sus nombres y accesos son etiquetas administrativas: todos esos polígonos se unen para seleccionar estructuras estáticas. Si no hay suficientes puntos en la referencia, el programa se detendrá antes de procesar y pedirá seleccionar un fondo mejor.

`--imgsz 960`, la NMS agnóstica y los umbrales mencionados ya son los valores predeterminados. `--sin-preview` permite procesar sin ventana; `--max-frames 150` limita una ejecución corta. `--sin-compensacion` sirve como comparación con cámara supuesta fija: en ese modo no se mide ni se verifica su estabilidad.

## 7. Qué revisar en esta ejecución

El objetivo de aceptación es que las zonas sigan sus carriles durante los pequeños movimientos de cámara y que las trayectorias compensadas sean coherentes con el movimiento real observado. Revisa especialmente los instantes donde se movía el encuadre.

El programa crea una carpeta fechada dentro de `resultados_compensados` y conserva:

| Archivo | Uso |
|---|---|
| `seguimiento_compensado.mp4` | Revisión visual de zonas, IDs y observaciones sin ID |
| `referencia.png`, `referencia_fondo.jpg` | Comprobar referencia y selección de fondo |
| `camara.csv` | Transformación, correspondencias, error y motivo de rechazo por fotograma |
| `conteos.csv` | Observaciones con/sin ID, conteos por zona y fracción visible |
| `observaciones.csv` | Posiciones actuales y compensadas, identidad y asignación |
| `observaciones.jsonl` | Detalles completos por fotograma, incluidas cajas y calidad |
| `ids.csv` | Historial resumido de identidades publicadas |
| `resumen.json` | Configuración, versiones, rendimiento, fallos y límites |
| `ultimo_frame.jpg`, `ultimo_frame.json` | Estado final visual y estructurado |

Los tiempos usados corresponden al video de origen. El MP4 de salida se guarda con sus FPS nominales, sin audio. La velocidad de procesamiento puede ser menor que tiempo real; todavía estamos evaluando el método fuera de línea.

Para revisar juntos esta etapa bastan inicialmente `resumen.json`, `camara.csv` y una imagen donde se aprecie la alineación. Si hay un fallo específico, indica su segundo en el video.

## 8. Qué quedó comprobado y qué sigue

En el entorno de desarrollo se comprobaron traslaciones, rotación pequeña y cambio de escala sintéticos; rechazo de imágenes sin información o sin correspondencias; recuperación; asignación geométrica; unión de detecciones con/sin ID; reinicio de identidades y salidas CSV, JSONL y MP4. La prueba de integración utilizó salidas simuladas de YOLO y ByteTrack, con OpenCV real.

También se registraron dos fotogramas reales que compartiste: se obtuvieron 539 inliers y un error mediano de aproximadamente 1,41 píxeles en los puntos aceptados. Es una comprobación parcial del registro; no es una validación del video completo ni de los conteos. La ejecución real con tu instalación de Ultralytics y GPU todavía debe verificarse en tu equipo.

Después de verificar esta compensación pasaremos a **cola visible**: suavizar el movimiento residual, definir una condición de detención persistente con tiempos del video y agrupar vehículos detenidos o de avance lento por carril y contexto de aproximación. Contar todos los vehículos de un carril no equivale a contar su cola, y un vehículo estacionado tampoco debe convertirse automáticamente en parte de ella.

Las observaciones sin ID pueden contribuir al total visible, pero no reciben de inmediato un tiempo individual de espera. Cualquier estimación de permanencia durante oclusiones tendrá que distinguirse de las observaciones reales y limitarse temporalmente.

## Referencias técnicas

- Ultralytics, ByteTrack y su API: https://docs.ultralytics.com/reference/trackers/byte_tracker/
- OpenCV, extracción de características ORB: https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
- OpenCV, `estimateAffinePartial2D`: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
