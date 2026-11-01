# Actualización del historial de movimiento, versión 2

## Instalación

Reemplaza solamente `estado_cola.py` en tu carpeta `tesis-vision` por la versión entregada con esta guía. Tanto `analizar_cola.py` como `seguir_compensado.py` importan ese módulo y utilizarán la corrección en su siguiente ejecución. No se requieren dependencias nuevas. Si un proceso ya está en marcha, deja que termine: su código cargado no se actualizará hasta reiniciarlo.

Esta guía sustituye la descripción anterior del criterio de `salto_posicion`. Los demás criterios de cola se mantienen.

## Problema corregido

La versión anterior calculaba una velocidad instantánea entre dos observaciones consecutivas y reiniciaba el historial si excedía tres alturas de caja por segundo. A 30 FPS, una caja de 60 píxeles podía perder su historial con un desplazamiento de apenas seis píxeles. Esa variación puede incluir ruido de localización, cambios de caja o error residual de compensación, además de movimiento real.

Un historial que se reinicia frecuentemente no llega a completar la ventana temporal de un segundo. Por eso aparecían tantos signos `?`, aunque el tracker sí siguiera publicando un ID.

## Nueva regla

Separamos la tolerancia espacial de la parte que depende del movimiento:

```text
H = altura media de las cajas de las dos observaciones
Δt = tiempo transcurrido en el video
D = desplazamiento del punto compensado entre observaciones

Tolerancia = máximo(8 píxeles, 0,15 × H)
Límite = Tolerancia + 3 × H × Δt

Reiniciar el historial solamente si D > Límite
```

Para H=60 píxeles y Δt=1/30 segundo, la tolerancia es 9 píxeles y el límite completo es 15 píxeles. El cálculo de velocidad sobre la ventana de un segundo sigue usando las posiciones observadas: no restamos esa tolerancia a la velocidad ni convertimos automáticamente esos desplazamientos en cero.

Los valores 8 y 0,15 son heurísticas iniciales, no una calibración del error del detector. Los ocho píxeles se refieren a la resolución original del video. Deben revisarse para otros encuadres y resoluciones. Una tolerancia mayor conserva más historial, pero también podría permitir que algún cambio de identidad pase inadvertido.

Para modificarlos explícitamente están disponibles:

- `--tolerancia-salto-px 8`
- `--tolerancia-salto-alturas 0.15`

Con ambos en cero se recupera matemáticamente el criterio anterior. No recomiendo aumentarlos nuevamente sin una comprobación visual.

## Qué se mantiene

- Un hueco sin observación, cambio de zona o ID nuevo empieza una nueva evidencia temporal.
- No contamos vehículos ausentes desde la memoria.
- Una detección sin ID sigue siendo visible, pero no adquiere permanencia individual.
- Un fallo de alineación conserva los valores de cola como no disponibles.
- Los límites de entrada y salida de cola, el umbral de detención y los tiempos de persistencia no cambiaron.
- No se corrige aquí la fragmentación de IDs causada por el poste. Esta actualización actúa sobre el historial que usamos después de ByteTrack.

## Comparación con tus resultados guardados

La comparación usa exactamente las mismas detecciones, IDs, zonas, tiempos y matrices. Solo cambia la regla del historial. Se evaluó `zona_01`.

| Registro | Reinicios por salto antes / ahora | Observaciones indeterminadas antes / ahora | Máximo del proxy antes / ahora |
|---|---:|---:|---:|
| Video completo: 938 fotogramas, ORB 5.000 puntos | 974 / 42 | 48,6 % / 27,2 % | 2 / 2 |
| Prueba: 300 fotogramas, ORB 1.000 puntos | 258 / 15 | 52,1 % / 33,2 % | 2 / 2 |

Los porcentajes cuentan observaciones repetidas a lo largo del video, no vehículos únicos. Esta comparación demuestra que conservamos más continuidad. No demuestra que los estados de movimiento o las colas sean correctos: falta contrastarlos con una referencia manual.

En el último fotograma de la ejecución completa sigue habiendo un candidato a cola y quedan diez indeterminados, frente a once antes. No confundir ese estado final con el cambio acumulado durante todo el archivo.

## Cómo probarlo sin volver a ejecutar YOLO

Desde la carpeta del proyecto, indicando el JSONL completo de la ejecución de 938 fotogramas:

```powershell
.\.venv\Scripts\python.exe .\analizar_cola.py --observaciones .\resultados_compensados\20261006T230829_784694Z\observaciones.jsonl --zonas-cola zona_01
```

Si guardaste ese JSONL en otra ubicación, cambia solamente la ruta. Este comando genera resultados nuevos y no modifica el archivo original. Recalcula los estados a partir de posiciones e IDs, aunque el JSONL ya contenga estados de movimiento de una versión anterior.

Para ver el resultado visual y completar la comprobación de 1.000 puntos en todo el video:

```powershell
.\.venv\Scripts\python.exe .\seguir_compensado.py --video .\video_prueba01.mp4 --zonas .\zonas_compensadas.json --zonas-cola zona_01 --puntos-fondo 1000 --guardar-video
```

Este segundo comando sí vuelve a ejecutar detección y compensación. El video antiguo no cambia por reemplazar el módulo o ejecutar el reanálisis de JSONL. La mejora de rendimiento de ORB con 1.000 puntos solo había sido probada en los primeros 300 fotogramas de tu equipo.

## Comprobaciones realizadas

- Vehículo inmóvil con oscilación artificial de ±4 píxeles, a 15, 30 y 60 FPS: conserva historial y alcanza la condición temporal de detención.
- Movimiento sostenido con ruido: no se convierte en cola por el solo hecho de tolerar ese ruido.
- Salto de 200 píxeles: reinicia el historial y deja de heredar permanencia.
- Ausencia de observación, recuperación y fallo de cámara: conservan el comportamiento de incertidumbre.
- Valores negativos o no finitos en tolerancias: se rechazan.
- Reanálisis completo de ambos registros reales y ejecución del comando de exportación.

No se ejecutó nuevamente YOLO ni se midió hardware Jetson en estas pruebas.

## Migración a Jetson Orin Nano

Los módulos Python, el modelo `.pt` y las reglas de procesamiento se pueden reutilizar. Hay que crear un entorno nuevo para Linux/ARM64, instalar versiones compatibles de Python, PyTorch, CUDA y librerías con JetPack, y revisar lectura de cámara y ventanas de OpenCV. La `.venv` de Windows no es portable. Para TensorRT, la exportación y validación se realizan en el entorno de destino; no hay que asumir que un motor generado en la RTX 2070 sea portable a Jetson.

Puedes desarrollar localmente con teclado, mouse y monitor y VS Code para ARM64. También puedes usar VS Code Remote SSH desde tu laptop: editas desde ella y ejecutas el programa en la Jetson. Con acceso remoto sin escritorio, el modo `--sin-preview` evita abrir ventanas; el editor de zonas sí requiere una sesión gráfica o preparar las zonas sobre el mismo encuadre por otro medio.

El kit requiere almacenamiento para el sistema y los archivos, como microSD o NVMe. La documentación oficial indica que el kit estándar no incluye almacenamiento extraíble. Su salida de monitor es DisplayPort; para un monitor HDMI, revisa un adaptador DisplayPort a HDMI compatible. USB-C no ofrece salida de pantalla.

Estas indicaciones se refieren a **Jetson Orin Nano Developer Kit 8 GB**, no a Jetson Nano original.

Fuentes oficiales:

- Inicio y contenido del kit: https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit
- Conectores: https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/hardware_spec.html
- Ultralytics en Jetson: https://docs.ultralytics.com/guides/nvidia-jetson/
- VS Code Linux: https://code.visualstudio.com/docs/setup/linux
- VS Code Remote SSH: https://code.visualstudio.com/docs/remote/ssh
