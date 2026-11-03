# traffic-vision

Visión para el prototipo de control semafórico del corredor El Hatillo.

Ejecutor multifuente: `ejecutar_vision.py`. Una cámara RTSP y un video pueden
funcionar simultáneamente con un detector YOLO compartido y un ByteTrack por
fuente. Los resultados de archivos se mantienen separados de las mediciones
actuales del corredor.

- [Instalación y prueba en Jetson](GUIA_JETSON_MULTIFUENTE.md)
- [Revisión del código y contrato para RL](REVISION_Y_CONTRATO_RL.md)
- Configuración inicial: `config/jetson_demo.json`
- Pruebas: `python -m pytest -q` (requiere pytest y dependencias de visión).

Los scripts anteriores se conservan para reproducir los experimentos offline.
Esta entrega proporciona detecciones y un **proxy de cola visible**, no métricas
físicas calibradas ni una política RL terminada. No acciona semáforos.
