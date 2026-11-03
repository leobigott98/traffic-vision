"""Cámara RTSP y videos simultáneos, detector compartido, estado parcial para RL."""
import argparse
from collections import deque
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from uuid import uuid4

from configuracion_vision import cargar, origen
from contrato_vision import construir_estado, guardar_atomico
from fuentes_video import Lector


class HistorialAcotado:
    """Solo trayectorias presentes: no acumula fichas de todos los vehículos del día."""
    def __init__(self):
        self.puntos = {}

    def actualizar(self, obs, frame, t):
        presentes = {o['track_id'] for o in obs if o['track_id'] is not None}
        self.puntos = {k:v for k,v in self.puntos.items() if k in presentes}
        for o in obs:
            if o['track_id'] is not None:
                self.puntos.setdefault(o['track_id'], deque(maxlen=30)).append((frame,t,*o['punto_xy']))


def conservar_contador_global(func):
    """Compatibilidad con ByteTrack antiguo: reset de A no reutiliza IDs activos en B."""
    from ultralytics.trackers.basetrack import BaseTrack
    contador = getattr(BaseTrack, '_count', None)
    try:
        return func()
    finally:
        if contador is not None:
            BaseTrack._count = contador


class Analizador:
    def __init__(self, src, tracker_cfg, parametros_cola):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from seguir_compensado import Identidades
        from estado_cola import EstimadorCola
        self.src = src
        self.parametros_cola = parametros_cola
        self.zonas = []
        self.config_zonas = None
        self.shape = None
        self.compensador = None
        # APIs recientes eliminaron frame_rate; conservar track_buffer en actualizaciones.
        kwargs = {'frame_rate':30} if 'frame_rate' in inspect.signature(BYTETracker).parameters else {}
        self.tracker = conservar_contador_global(lambda: BYTETracker(SimpleNamespace(**tracker_cfg), **kwargs))
        self.identidades = Identidades()
        self.historial = HistorialAcotado()
        self.estimador = None
        self.n = 0
        self.previo = None
        self.epoca = None
        self.valida_previa = True
        self.reglas = dict(minimo=.5, margen=.1, fraccion_inferior=.25, tolerancia=5.)

    def reiniciar(self):
        from estado_cola import EstimadorCola
        conservar_contador_global(self.tracker.reset)
        self.identidades.reiniciar()
        self.historial.puntos.clear()
        if self.zonas:
            self.estimador = EstimadorCola([z['id'] for z,_,_ in self.zonas], self.parametros_cola)

    def preparar(self, frame, model, params):
        import cv2
        from contar_multizonas import leer_config, preparar_zonas
        from seguir_compensado import crear_mascara
        from compensacion_camara import CompensadorCamara
        from estado_cola import EstimadorCola
        self.shape = frame.shape[:2]
        h,w = self.shape
        if self.src.get('zonas'):
            self.config_zonas = leer_config(Path(self.src['zonas']), w,h)
            self.zonas = preparar_zonas(self.config_zonas,w,h,5.)
            if set(self.src['mapeo_accesos']) - {z['id'] for z,_,_ in self.zonas}:
                raise ValueError('El mapeo contiene zonas inexistentes.')
            self.estimador = EstimadorCola([z['id'] for z,_,_ in self.zonas], self.parametros_cola)
        if not self.src['camara_fija']:
            ref = cv2.imread(self.src['referencia'])
            if ref is None or ref.shape[:2] != self.shape:
                raise ValueError('Referencia ilegible o de resolución distinta.')
            fondo = leer_config(Path(self.src['fondo']),w,h) if self.src.get('fondo') else None
            r = model.predict(ref, **params)[0]
            self.compensador = CompensadorCamara(ref, crear_mascara(ref,fondo,.4),
                r.boxes.xyxy.cpu().numpy(), ancho_analisis=640, max_puntos=2000)

    def procesar(self, paquete, model, params, salto_max_s, preview):
        import numpy as np
        from ultralytics.engine.results import Boxes
        from seguir_compensado import combinar, contar, dibujar
        from compensacion_camara import caja_en_referencia
        frame, t = paquete.imagen, paquete.tiempo_s
        if self.shape is None:
            self.preparar(frame,model,params)
        if frame.shape[:2] != self.shape:
            raise ValueError('Cambio de resolución: recalibra zonas antes de continuar.')
        if (self.epoca is not None and paquete.epoca != self.epoca) or (
                self.previo is not None and (t <= self.previo or t-self.previo > salto_max_s)):
            self.reiniciar()
        self.epoca, self.previo = paquete.epoca,t
        tic=time.monotonic()
        r=model.predict(frame,**params)[0]
        ms=(time.monotonic()-tic)*1000
        b=r.boxes.cpu().numpy()
        candidatos=np.column_stack((b.xyxy,b.conf,b.cls)).astype(np.float32)
        if self.compensador:
            matriz,info=self.compensador.estimar(frame,candidatos[:,:4])
        else:
            matriz=np.array([[1.,0.,0.],[0.,1.,0.]])
            info=dict(valida=True,motivo='fija_asumida',parejas=0,inliers=0,error_mediano_px=None)
        valida=matriz is not None
        if not valida and self.valida_previa:
            self.reiniciar()
        self.valida_previa=valida
        if valida:
            referencia=candidatos.copy()
            for i,caja in enumerate(candidatos[:,:4]):
                referencia[i,:4]=caja_en_referencia(caja,matriz)
            tracks=self.tracker.update(Boxes(referencia,self.shape))
        else:
            tracks=[]
        obs=combinar(candidatos,tracks,matriz,self.identidades,.25,self.zonas,self.reglas,r.names)
        self.historial.actualizar(obs,self.n,t)
        cuentas=contar(obs,self.zonas,matriz,self.shape)
        cola=self.estimador.actualizar(self.n,t,obs,cuentas,valida) if self.estimador else {}
        # Limitar mapas de IDs; solo activos/perdidos pueden reaparecer en ByteTrack.
        activos={x.track_id for x in self.tracker.tracked_stracks+self.tracker.lost_stracks}
        self.identidades.mapa={k:v for k,v in self.identidades.mapa.items() if k in activos}
        self.tracker.removed_stracks=self.tracker.removed_stracks[-1000:]
        vista=dibujar(frame,obs,self.zonas,matriz,info,self.historial,cuentas,self.n,t,cola) if preview else None
        self.n+=1
        data=dict(frame=paquete.indice, tiempo_fuente_s=t,
            recibido_monotonic=paquete.recibido_monotonic,recibido_utc=paquete.recibido_utc,
            epoca=self.identidades.epoca,inferencia_ms=ms,cola=cola)
        return data,vista


def editar(src):
    import cv2
    import numpy as np
    from contar_multizonas import editar_zonas, leer_config
    lector=Lector(src,origen(src))
    cap=lector.abrir()
    try:
        ok,frame=cap.read()
        if not ok:
            raise ValueError('No se pudo capturar la referencia para editar.')
    finally:
        cap.release()
    ruta=Path(src['zonas'])
    h,w=frame.shape[:2]
    anteriores=leer_config(ruta,w,h)['zonas'] if ruta.exists() else []
    zonas=editar_zonas(frame,anteriores,12,cv2,np)
    if zonas is None:
        return
    ref=ruta.with_suffix('.referencia.png')
    ref.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(ref),frame):
        raise ValueError('No se pudo guardar la imagen de referencia.')
    guardar_atomico(ruta,dict(version=2,alto_ancho=[h,w],zonas=zonas,referencia_frame=0))
    print('Zonas guardadas:',ruta)
    print('Referencia:',ref)
    print('IDs para mapeo_accesos:', ', '.join(z['id']+'='+z['nombre'] for z in zonas))


def argumentos():
    from estado_cola import agregar_argumentos
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--modelo',default='yolo11n.pt')
    p.add_argument('--device',default='0')
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--tracker',type=Path,default=Path(__file__).with_name('bytetrack_tesis.yaml'))
    p.add_argument('--salida',type=Path,default=Path('resultados_multifuente'))
    p.add_argument('--sin-preview',action='store_true')
    p.add_argument('--duracion-s',type=float,default=0,help='0: hasta Q/Ctrl+C o fin de todas las fuentes')
    p.add_argument('--editar-zonas',metavar='FUENTE_ID')
    p.add_argument('--validar',action='store_true',help='Valida configuración sin cargar YOLO ni abrir cámara')
    agregar_argumentos(p)
    a=p.parse_args()
    if a.zonas_cola:
        p.error('En multifuente se evalúan todas las zonas; configura su mapeo en el JSON.')
    import math
    if a.imgsz < 32 or not math.isfinite(a.duracion_s) or a.duracion_s < 0:
        p.error('imgsz >=32 y duración finita >=0.')
    return a


def main():
    a=argumentos()
    cfg=cargar(a.config,comprobar_archivos=not bool(a.editar_zonas))
    # No imprime URL ni pasa RTSP al logger de Ultralytics: el detector recibe matrices.
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']='rtsp_transport;tcp'
    os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL','-8')
    if a.editar_zonas:
        src=next((s for s in cfg['fuentes'] if s['id']==a.editar_zonas),None)
        if src is None or not src.get('zonas'):
            raise ValueError('Fuente desconocida o sin ruta zonas en la configuración.')
        editar(src)
        return
    # Validar geometrías y mapeos antes de reservar GPU o empezar lectores.
    from contar_multizonas import leer_config
    for src in cfg['fuentes']:
        origen(src)
        if src.get('zonas'):
            raw=json.loads(Path(src['zonas']).read_text(encoding='utf-8'))
            h,w=raw['alto_ancho']
            zonas=leer_config(Path(src['zonas']),w,h)['zonas']
            if set(src['mapeo_accesos'])-{z['id'] for z in zonas}:
                raise ValueError(f"{src['id']}: mapeo con zonas inexistentes.")
    from seguir_video import leer_tracker
    from estado_cola import parametros_argumentos
    tracker_cfg=leer_tracker(a.tracker,.1)
    cola_params=parametros_argumentos(a)
    if a.validar:
        print('Configuración válida. Fuentes:', ', '.join(s['id'] for s in cfg['fuentes']))
        print('Sin prueba de red, GPU, precisión ni correspondencia física de zonas.')
        return
    import cv2
    import ultralytics
    from ultralytics import YOLO
    from contar_multizonas import VEHICULOS
    cv2.setNumThreads(1)
    model=YOLO(a.modelo)
    if set(VEHICULOS)-set(model.names.values()):
        raise ValueError('El modelo debe incluir car, motorcycle, bus y truck.')
    params=dict(device=a.device,imgsz=a.imgsz,conf=.1,iou=.7,agnostic_nms=True,
        classes=[i for i,n in model.names.items() if n in VEHICULOS],save=False,verbose=False)
    lectores={s['id']:Lector(s,origen(s)) for s in cfg['fuentes']}
    analizadores={s['id']:Analizador(s,tracker_cfg,cola_params) for s in cfg['fuentes']}
    sesion=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid4().hex[:8]
    carpeta=a.salida.resolve()/sesion
    carpeta.mkdir(parents=True)
    guardar_atomico(carpeta/'config_usada.json',dict(config=cfg,ultralytics=ultralytics.__version__,
        modelo=a.modelo,imgsz=a.imgsz,device=a.device,tracker=tracker_cfg,cola=vars(cola_params)))
    ultimos,fallos={},{}
    inicio=time.monotonic()
    siguiente=inicio
    # Rotación por tamaño: máximo aprox. 3 x 10 MiB para el historial agregado.
    from logging.handlers import RotatingFileHandler
    import logging
    log=logging.getLogger('observaciones.'+sesion)
    log.setLevel(logging.INFO);log.propagate=False
    handler=RotatingFileHandler(carpeta/'estado.jsonl',maxBytes=10*1024*1024,backupCount=2,encoding='utf-8')
    log.addHandler(handler)
    def publicar(final=False):
        estados={k: ('detenido' if final else fallos.get(k,l.estado)) for k,l in lectores.items()}
        data=construir_estado(cfg,ultimos,estados,time.monotonic(),sesion)
        guardar_atomico(carpeta/'estado_actual.json',data)
        log.info(json.dumps(data,ensure_ascii=False,allow_nan=False))
    print('Resultados:',carpeta)
    print('Un detector compartido; fuentes independientes. Q/Esc o Ctrl+C termina.')
    try:
        publicar()
        for l in lectores.values():
            l.start()
        terminar=False
        while not terminar:
            trabajo=False
            for sid,l in lectores.items():
                if sid in fallos:
                    continue
                paquete=l.tomar()
                if paquete is None:
                    continue
                if l.config['tipo']=='rtsp' and time.monotonic()-paquete.recibido_monotonic>cfg['caducidad_s']:
                    continue
                trabajo=True
                try:
                    data,vista=analizadores[sid].procesar(paquete,model,params,cfg['salto_max_s'],not a.sin_preview)
                    data['descartados']=l.descartados
                    ultimos[sid]=data
                    if vista is not None:
                        cv2.imshow(sid+' | '+l.config['tipo'],vista)
                except Exception as exc:
                    fallos[sid]='error_analisis'
                    l.detener.set()
                    while l.tomar() is not None:
                        pass
                    print(f'{sid}: error de análisis ({type(exc).__name__}). {exc}' if l.config['tipo']=='archivo'
                          else f'{sid}: error de análisis ({type(exc).__name__}); revisa modelo, zonas y resolución.')
            if not a.sin_preview and cv2.waitKey(1)&0xff in (27,ord('q'),ord('Q')):
                terminar=True
            ahora=time.monotonic()
            if ahora>=siguiente:
                publicar();siguiente=ahora+cfg['publicar_cada_s']
                print(' | '.join(f'{sid}: {fallos.get(sid,l.estado)} frames={analizadores[sid].n}' for sid,l in lectores.items()))
            if a.duracion_s and ahora-inicio>=a.duracion_s:
                terminar=True
            if all(sid in fallos or (l.estado in ('finalizado','error','detenido') and l.cola.empty()) for sid,l in lectores.items()):
                terminar=True
            if not trabajo:
                time.sleep(.01)
    except KeyboardInterrupt:
        pass
    finally:
        for l in lectores.values():
            l.detener.set()
        publicar(final=True)
        for l in lectores.values():
            if l.ident is not None:
                l.join(timeout=6.)
        if not a.sin_preview:
            cv2.destroyAllWindows()
        handler.close();log.removeHandler(handler)
    if fallos or any(l.error is not None and l.config['tipo']=='archivo' for l in lectores.values()):
        raise SystemExit(1)


if __name__=='__main__':
    main()
