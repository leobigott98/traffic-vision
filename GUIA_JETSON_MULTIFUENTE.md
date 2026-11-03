# Cámara y archivo simultáneos en el Jetson

## 1. Obtener el código

En la carpeta donde clonaste este repositorio:

```bash
git status --short
git fetch origin
git switch --track origin/codex/jetson-multifuente
```

Si tienes cambios locales, consérvalos antes de cambiar de rama. Si ya creaste
la rama local, usa `git switch codex/jetson-multifuente`.
No uses `reset --hard` para resolver un conflicto.

Activa el entorno que ya ejecuta YOLO en el Jetson:

```bash
source ~/tesis/vision/.venv/bin/activate
python -c "import sys; print(sys.executable)"
python -m pip install --no-deps -r requirements-jetson-extra.txt
python -c "import cv2, numpy, shapely, yaml, lap, torch, ultralytics; print('Ultralytics:', ultralytics.__version__); print('CUDA:', torch.cuda.is_available())"
```

`--no-deps` conserva las versiones de torch, torchvision, numpy y OpenCV.
Los complementos necesitan numpy, ya presente en el entorno de YOLO. No instales
OpenCV headless en el Jetson si quieres ventanas. No repitas la instalación de
JetPack. La advertencia existente de PyTorch CC 8.7 sigue pendiente; este código
no la corrige ni la oculta.

Mantén la swap que configuraste y cierra aplicaciones pesadas durante la prueba.
Un detector compartido reduce la duplicación del modelo; la decodificación,
las imágenes y la GPU siguen consumiendo memoria. No hay garantía de FPS hasta
medir ambas fuentes en tu equipo.

## 2. Configuración inicial sin zonas

```bash
cp config/jetson_demo.json config/jetson.local.json
```

Edita `config/jetson.local.json`: en la fuente `grabacion`, cambia `ruta` por
la ruta absoluta a tu video, por ejemplo:

```json
"ruta": "/home/leonardoandresbigott/tesis/vision/video_prueba.mp4"
```

La configuración de ejemplo asume cámaras fijas. Si el video se mueve, la prueba
sin zonas solo sirve para detección y visualización; activa compensación antes
de interpretar velocidades o colas. Todas las rutas relativas del JSON se
resuelven respecto a la carpeta de ese JSON.

En la misma terminal, introduce la URL RTSP completa y expórtala:

```bash
read -r -s -p "URL RTSP de la Tapo: " CAMARA_RTSP
printf '\n'
export CAMARA_RTSP
```

Formato: `rtsp://USUARIO:CONTRASENA@IP:554/stream1`. Si las credenciales contienen
caracteres reservados, codifícalos para una URL. El valor queda en el entorno de
esta sesión; el JSON solo contiene el nombre de la variable. El programa pasa
imágenes a YOLO, por lo que Ultralytics no recibe ni imprime la URL de cámara.

Valida archivos y configuración:

```bash
python ejecutar_vision.py --config config/jetson.local.json --validar
```

La validación no abre la red ni comprueba la GPU. No modifica la configuración.

## 3. Primera prueba simultánea, 60 segundos

Ejecuta desde el escritorio gráfico del Jetson (monitor conectado o sesión
remota con escritorio, no una conexión SSH sin pantalla):

```bash
python ejecutar_vision.py \
  --config config/jetson.local.json \
  --modelo ~/tesis/vision/yolo11n.pt \
  --device 0 --imgsz 640 --duracion-s 60
```

Aparecen dos ventanas: `tapo | rtsp` y `grabacion | archivo`. Q/Esc en una ventana
o Ctrl+C termina la ejecución. Puedes omitir `--duracion-s` para continuar hasta
que la detengas. Si el archivo termina, la cámara continúa. El archivo no se
repite automáticamente; repetirlo exigiría reiniciar también su evidencia temporal.

Sin monitor usa `--sin-preview`; la exportación continúa. No se guarda video
anotado en este ejecutor: evita escritura y codificación adicional durante la
prueba simultánea. Los scripts offline anteriores conservan su opción de grabar.

En otra terminal:

```bash
tegrastats --interval 1000
```

Comprueba RAM, swap y que ambas fuentes avanzan. El tiempo `inferencia_ms` del
JSON mide solo detector y transferencia interna, no latencia total ni FPS de
la cámara. La edad de recepción tampoco incluye el búfer interno de la cámara
ni de FFmpeg: para medir latencia extremo a extremo usa un reloj visible en escena.

**Sin zonas, la detección funciona pero los accesos se exportan como desconocidos.**
No hay polígonos universales: los archivos antiguos son de otros encuadres y no
sirven automáticamente para la Tapo.

## 4. Dibujar las zonas sobre cada fuente

Para `tapo`, cambia en el JSON local:

```json
"zonas": "../calibracion/tapo.json"
```

Luego:

```bash
python ejecutar_vision.py --config config/jetson.local.json --editar-zonas tapo
```

El editor toma una imagen, cierra la captura y permite dibujar:

1. Clic para agregar vértices; Enter para cerrar un polígono.
2. En la terminal, escribe nombre y acceso. Son etiquetas; la vinculación formal
   con RL se hace en `mapeo_accesos`.
3. Dibuja un polígono por aproximación observada, sobre la zona de acumulación
   antes de la línea de parada. Excluye estacionamiento y tránsito de salida.
4. S guarda; Esc cancela. El editor muestra los IDs `zona_01`, etc.

Se guardan `calibracion/tapo.json` y `calibracion/tapo.referencia.png`.
Para la grabación, configura `zonas` como `../calibracion/grabacion.json` y ejecuta:

```bash
python ejecutar_vision.py --config config/jetson.local.json --editar-zonas grabacion
```

En cada fuente, configura el mapeo según lo que realmente ve, por ejemplo:

```json
"mapeo_accesos": {
  "zona_01": "boyera_trinidad",
  "zona_02": "trinidad_boyera",
  "zona_03": "cigarral"
}
```

Es un ejemplo, no una asignación automática. Omite los accesos que esa cámara
no ve. Si solo ves Trinidad→Boyera, mapea únicamente esa zona. Un video de otra
ciudad no constituye una medición de Cigarral aunque pruebes el mismo nombre
lógico; sus resultados permanecen en el dominio `reproduccion`.

Para un archivo con cámara móvil, configura:

```json
"camara_fija": false,
"referencia": "../calibracion/grabacion.referencia.png"
```

La compensación utiliza la referencia guardada con las zonas, nunca una imagen
nueva que desplazaría su significado. Por defecto toma la franja superior del
40% como fondo; debe contener estructuras estáticas con textura. Puedes añadir
`fondo` con un JSON de polígonos de fondo en la misma referencia. Una alineación
rechazada produce datos nulos y reinicia seguimiento/evidencia. La compensación
es una transformación de similitud: no garantiza corrección de perspectiva o
paralaje. Si `camara_fija=true`, el montaje estable es una precondición que debes
verificar: mover físicamente la Tapo obliga a recalibrar.

Valida de nuevo y repite la prueba de 60 segundos. Para cada zona comprueba
visualmente los vehículos observados, indeterminados y cola*. La ventana temporal
inicial tarda aproximadamente 1 s más 2 s de permanencia en clasificar una cola;
las pérdidas de ID reinician esa evidencia.

## 5. Leer resultados

La terminal imprime una carpeta única bajo `resultados_multifuente/`:

- `estado_actual.json`: reemplazo atómico; lectura segura mientras se actualiza.
- `estado.jsonl`: estados periódicos para auditoría, con rotación de 10 MiB y
  dos respaldos (`.1`, `.2`). Copia los archivos antes de superar esa retención
  si necesitas una campaña larga para el tomo.
- `config_usada.json`: configuración, umbrales y versión de Ultralytics.

`accesos` contiene exclusivamente mediciones en vivo. Los resultados del archivo
están bajo `fuentes.grabacion.accesos` y `fuentes.grabacion.zonas`. Si quieres
probar un controlador sobre el video, ese controlador debe seleccionar
explícitamente el dominio de reproducción y su reloj, sin mandar acciones al campo.

Cada fuente tiene su propio tracker, reloj e historial. RTSP conserva solo el
último frame recibido; el archivo conserva el orden y se frena si el detector
no alcanza su ritmo. `ritmo_real=true` limita la reproducción a su FPS original,
pero no promete alcanzarlo con carga. La alternancia no sincroniza escenas ni
convierte la grabación en una segunda cámara del mismo instante.

Al desconectar RTSP, el lector intenta reconectar; mientras tanto el estado
caduca y sus métricas se vuelven nulas. Al recuperar, se reinician los IDs y la
historia temporal de esa fuente. El video puede seguir funcionando durante la
caída de la cámara. Los timeouts RTSP son de 5 s; la caducidad predeterminada es
3 s. El consumidor debe controlar también `generado_utc` por si el proceso entero
muere: el archivo no puede actualizarse después de un OOM.

## 6. Criterio para pasar a integración con RL

Antes de usar esta salida como observación del agente:

- Validar manualmente conteos y colas sobre escenas representativas; incluir
  oclusiones, noche y pérdidas de seguimiento.
- Definir y reproducir en SUMO las mismas regiones observadas y señales proxy,
  o implementar calibración métrica que haga comparables las mediciones.
- Acordar orden, escalas, máscaras, intervalo de decisión y contrato de la política.
- Incorporar fase actual, tiempo en fase, transiciones y estado de los esclavos
  desde el controlador; no desde el detector.
- Repetir la prueba prolongada en Jetson y comprobar consumo estable, latencia
  y desconexión. Esta entrega no valida estabilidad térmica ni rendimiento en Orin.

`listo_para_control` permanece `false` hasta esa integración; es intencional.
