"""Lectores independientes: RTSP último frame, archivo con cola acotada y backpressure."""
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from queue import Queue, Empty, Full
from threading import Event, Thread
from time import monotonic


@dataclass
class Fotograma:
    imagen: object
    indice: int
    tiempo_s: float
    recibido_monotonic: float
    recibido_utc: str
    epoca: int


class Lector(Thread):
    def __init__(self, config, origen, capture_factory=None):
        super().__init__(name=config['id'], daemon=True)
        self.config, self.origen = config, origen
        self.cola = Queue(maxsize=1)
        self.detener = Event()
        self.estado = 'iniciando'
        self.error = None
        self.descartados = 0
        self.capture_factory = capture_factory

    def abrir(self):
        if self.capture_factory:
            return self.capture_factory(self.origen)
        import cv2
        if self.config['tipo'] == 'rtsp':
            # Timeouts se fijan al abrir: acotan bloqueos de red/lectura.
            return cv2.VideoCapture(self.origen, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        return cv2.VideoCapture(self.origen)

    def tomar(self):
        try:
            return self.cola.get_nowait()
        except Empty:
            return None

    def publicar(self, paquete):
        if self.config['tipo'] == 'rtsp':
            try:
                self.cola.put_nowait(paquete)
            except Full:
                try:
                    self.cola.get_nowait()
                    self.descartados += 1
                except Empty:
                    pass
                self.cola.put_nowait(paquete)
        else:
            while not self.detener.is_set():
                try:
                    self.cola.put(paquete, timeout=.1)
                    return
                except Full:
                    pass

    def run(self):
        import cv2
        epoca = 0
        inicio_live = monotonic()
        try:
            while not self.detener.is_set():
                cap = None
                try:
                    cap = self.abrir()
                    if not cap.isOpened():
                        raise RuntimeError('apertura_fallida')
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    if self.config['tipo'] == 'archivo' and not (math.isfinite(fps) and fps > 0):
                        raise ValueError('archivo_sin_fps_validos')
                    indice, previo = 0, -1.
                    inicio_archivo = monotonic()
                    while not self.detener.is_set():
                        ok, frame = cap.read()
                        if not ok:
                            if self.config['tipo'] == 'archivo':
                                nominal = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                if indice == 0 or (math.isfinite(nominal) and nominal > 0 and indice < int(nominal)):
                                    raise ValueError('archivo_truncado_o_ilegible')
                                # No declarar fin hasta que el consumidor tome el último frame.
                                while not self.cola.empty() and not self.detener.wait(.05):
                                    pass
                                self.estado = 'finalizado'
                                return
                            raise RuntimeError('lectura_rtsp_fallida')
                        if self.config['tipo'] == 'archivo':
                            from contar_video import tiempo_video
                            t, _ = tiempo_video(indice, cap.get(cv2.CAP_PROP_POS_MSEC), fps, previo)
                            if self.config.get('ritmo_real', True):
                                if self.detener.wait(max(0., inicio_archivo+t-monotonic())):
                                    break
                        else:
                            t = monotonic() - inicio_live
                        previo = t
                        paquete = Fotograma(frame, indice, t, monotonic(),
                            datetime.now(timezone.utc).isoformat(), epoca)
                        self.estado = 'activo'
                        self.publicar(paquete)
                        indice += 1
                except Exception as exc:
                    # No exportar texto del backend: podría contener usuario/contraseña RTSP.
                    self.error = type(exc).__name__
                    self.estado = 'desconectado' if self.config['tipo']=='rtsp' else 'error'
                    while self.tomar() is not None:
                        pass
                    if self.config['tipo'] == 'archivo':
                        return
                finally:
                    if cap is not None:
                        cap.release()
                epoca += 1
                if self.detener.wait(2.):
                    break
        finally:
            if self.detener.is_set():
                self.estado = 'detenido'
