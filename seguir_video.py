"""Seguimiento de vehículos con YOLO11 + ByteTrack: IDs, trayectorias y CSV.

Requiere contar_video.py y contar_multizonas.py en esta misma carpeta.
Ejemplo:
  python seguir_video.py --video video_prueba01.mp4 --guardar-video

--zonas zonas_existentes.json es opcional. Las zonas deben tener el mismo encuadre.
Q/Esc: terminar. Los IDs son hipótesis de identidad, NO flujo ni cola.
Esta etapa no compensa movimiento de cámara ni calcula velocidades físicas.
"""
import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from time import perf_counter

from contar_multizonas import VEHICULOS, leer_config, preparar_zonas, asignar, ascii_text
from contar_video import tiempo_video


class Historial:
    """Guarda puntos recientes para dibujar y estadísticas por ID para auditar.

    No asigna IDs: eso lo hace ByteTrack. Si un ID vuelve después de una ausencia,
    iniciamos un nuevo segmento visual para no dibujar posiciones inventadas.
    """
    def __init__(self, max_puntos=60):
        self.max_puntos = max_puntos
        self.puntos = {}
        self.fichas = {}

    def actualizar(self, objetos, frame, tiempo):
        ids = [o['track_id'] for o in objetos]
        if len(ids) != len(set(ids)):
            raise ValueError('El tracker devolvió el mismo ID dos veces en un fotograma.')
        for o in objetos:
            ident = o['track_id']
            ficha = self.fichas.get(ident)
            if ficha is None:
                ficha = {'track_id':ident, 'primer_frame':frame, 'ultimo_frame':frame,
                         'primer_tiempo_s':tiempo, 'ultimo_tiempo_s':tiempo,
                         'observaciones':0, 'reapariciones':0, 'cambios_clase':0,
                         'ultima_clase':o['clase'], 'clases':Counter()}
                self.fichas[ident] = ficha
                salto = 0
            else:
                salto = frame - ficha['ultimo_frame'] - 1
                if salto < 0:
                    raise ValueError('Los fotogramas del historial deben avanzar en orden.')
                if salto > 0:
                    ficha['reapariciones'] += 1
                    self.puntos.pop(ident, None)
                if ficha['ultima_clase'] != o['clase']:
                    ficha['cambios_clase'] += 1
            o['frames_ausentes_previos'] = salto
            puntos = self.puntos.setdefault(ident, deque(maxlen=self.max_puntos))
            puntos.append((frame, tiempo, *o['punto_xy']))
            ficha.update(ultimo_frame=frame, ultimo_tiempo_s=tiempo, ultima_clase=o['clase'])
            ficha['observaciones'] += 1
            ficha['clases'][o['clase']] += 1
        # La tabla pequeña por ID permanece; los puntos viejos dejan de ocupar memoria.
        for ident in list(self.puntos):
            if frame - self.fichas[ident]['ultimo_frame'] > self.max_puntos:
                del self.puntos[ident]

    def filas_resumen(self):
        for ident, f in sorted(self.fichas.items()):
            yield {'track_id':ident, 'primer_frame':f['primer_frame'],
                   'ultimo_frame':f['ultimo_frame'], 'primer_tiempo_s':f['primer_tiempo_s'],
                   'ultimo_tiempo_s':f['ultimo_tiempo_s'],
                   'intervalo_primera_ultima_s':f['ultimo_tiempo_s']-f['primer_tiempo_s'],
                   'observaciones':f['observaciones'], 'reapariciones':f['reapariciones'],
                   'cambios_clase':f['cambios_clase'],
                   'clase_mas_frecuente':f['clases'].most_common(1)[0][0]}


def extraer_objetos(resultado):
    """Solo publica cajas que tienen ID; nunca inventa uno con enumerate().

    Algunas versiones devuelven cajas sin ID cuando no hay tracks confirmados.
    Las registramos como diagnóstico, pero no como vehículos seguidos.
    """
    cajas = resultado.boxes
    if cajas is None or len(cajas) == 0:
        return [], 0
    cajas = cajas.cpu()
    if cajas.id is None:
        return [], len(cajas)
    objetos = []
    for xyxy, clase_id, confianza, ident in zip(
            cajas.xyxy.tolist(), cajas.cls.tolist(), cajas.conf.tolist(), cajas.id.tolist()):
        if not math.isfinite(ident) or ident != int(ident):
            raise ValueError('ID de seguimiento no válido.')
        x1,y1,x2,y2 = map(float, xyxy)
        if not all(math.isfinite(x) for x in (x1,y1,x2,y2,confianza)) or x2<=x1 or y2<=y1:
            raise ValueError('Caja o confianza de seguimiento no válida.')
        clase = resultado.names[int(clase_id)]
        if clase not in VEHICULOS:
            raise ValueError(f'Clase inesperada: {clase}')
        objetos.append({'track_id':int(ident), 'clase':clase, 'confianza':float(confianza),
                        'xyxy':[x1,y1,x2,y2], 'punto_xy':[(x1+x2)/2,y2]})
    return objetos, 0


def asignar_y_contar(objetos, zonas, reglas):
    """Cuenta solo IDs devueltos en este frame; no añade tracks perdidos del buffer."""
    conteos = {z['id']:0 for z,_,_ in zonas}
    estados = dict.fromkeys(('asignado','ambiguo','fuera','sin_zonas'), 0)
    for o in objetos:
        a = asignar(o['xyxy'], zonas, **reglas) if zonas else {'estado':'sin_zonas','zona_id':None}
        o['asignacion'] = a
        estados[a['estado']] += 1
        if a['zona_id'] is not None:
            conteos[a['zona_id']] += 1
    return conteos, estados


def color_id(ident):
    # Determinista: el mismo ID conserva color. Los colores no son identificadores únicos.
    return (60+(ident*67)%196, 60+(ident*101)%196, 60+(ident*139)%196)


def crear_vista(frame, objetos, zonas, conteos, estados, historial, indice, t, ms, sin_id, cv2, np):
    alto, ancho = frame.shape[:2]
    escala = min(1280/ancho, 720/alto, 1.)
    w, h = max(2,round(ancho*escala)), max(2,round(alto*escala))
    vista = cv2.resize(frame,(w,h))
    def punto(x,y):
        return (round(x*w/ancho),round(y*h/alto))
    for z,_,_ in zonas:
        vertices = np.array([punto(x,y) for x,y in z['puntos_xy']],np.int32)
        cv2.polylines(vista,[vertices],True,(220,220,220),1)
        cv2.putText(vista,ascii_text(z['id']),tuple(vertices[0]),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    for o in objetos:
        ident = o['track_id']; color = color_id(ident)
        puntos = [punto(x,y) for _,_,x,y in historial.puntos.get(ident,())]
        if len(puntos)>1:
            cv2.polylines(vista,[np.array(puntos,np.int32)],False,color,2)
        x1,y1,x2,y2 = o['xyxy']
        cv2.rectangle(vista,punto(x1,y1),punto(x2,y2),color,1)
        cv2.circle(vista,punto(*o['punto_xy']),3,color,-1)
        x,y = punto(x1,y1)
        cv2.putText(vista,f"ID {ident} {o['clase']}",(x,max(14,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1)
    alto_panel = max(h,260+38*len(zonas))
    # Dimensiones pares: compatibles con el codificador MP4 empleado.
    salida = np.full((alto_panel+(alto_panel%2),w+400+((w+400)%2),3),28,np.uint8)
    salida[:h,:w] = vista
    def texto(s,y,color=(235,235,235)):
        cv2.putText(salida,ascii_text(s)[:48],(w+12,y),cv2.FONT_HERSHEY_SIMPLEX,.46,color,1)
    texto(f'Frame {indice} | video {t:.2f} s',25)
    texto(f'Analisis: {ms:.1f} ms',49)
    texto(f'IDs presentes: {len(objetos)}',80)
    texto(f'IDs distintos publicados: {len(historial.fichas)}',104)
    texto('IDs distintos NO equivalen a flujo',128,(0,195,255))
    texto(f'Salidas sin ID: {sin_id}',152)
    y=183
    for z,_,_ in zonas:
        texto(f"{z['id']}: {conteos[z['id']]} vehiculos",y)
        texto(z['nombre'],y+16,(175,175,175));y+=38
    if zonas:
        texto(f"Fuera: {estados['fuera']} | Ambiguos: {estados['ambiguo']}",y)
    else:
        texto('Sin zonas: seguimiento en imagen completa',y)
    texto('Sin compensacion de camara / sin cola',y+24,(0,195,255))
    texto('Q / Esc: finalizar',y+48)
    return salida


def leer_tracker(ruta, conf):
    import yaml
    cfg = yaml.safe_load(ruta.read_text(encoding='utf-8'))
    if not isinstance(cfg,dict) or cfg.get('tracker_type')!='bytetrack':
        raise ValueError('Esta etapa requiere un YAML de ByteTrack.')
    for clave in ('track_high_thresh','track_low_thresh','new_track_thresh','match_thresh'):
        valor = cfg.get(clave)
        if isinstance(valor,bool) or not isinstance(valor,(int,float)) or not math.isfinite(valor) or not 0<=valor<=1:
            raise ValueError(f'{clave} debe estar entre 0 y 1.')
    if not 0<conf<=cfg['track_low_thresh']<cfg['track_high_thresh']<=cfg['new_track_thresh']<=1:
        raise ValueError('Requiere 0 < conf <= track_low_thresh < track_high_thresh <= new_track_thresh <= 1. Usa --conf 0.10 con el YAML inicial.')
    if type(cfg.get('track_buffer')) is not int or cfg['track_buffer']<1:
        raise ValueError('track_buffer debe ser un entero positivo.')
    if type(cfg.get('fuse_score')) is not bool:
        raise ValueError('fuse_score debe ser true o false.')
    return cfg


def argumentos():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--zonas',type=Path,help='JSON existente; omitir para seguir toda la imagen')
    p.add_argument('--modelo',default='yolo11n.pt')
    p.add_argument('--device',default='0')
    p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--conf',type=float,default=.10,help='Umbral de candidatos para ByteTrack, no para iniciar IDs')
    p.add_argument('--nms-agnostica',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--tracker',type=Path,default=Path(__file__).resolve().with_name('bytetrack_tesis.yaml'))
    p.add_argument('--trayectoria',type=int,default=60,help='Número máximo de observaciones del trazo')
    p.add_argument('--max-frames',type=int,default=0,help='0 procesa todo el video')
    p.add_argument('--sin-preview',action='store_true')
    p.add_argument('--guardar-video',action='store_true',help='MP4 anotado a FPS nominales de la fuente, sin audio')
    p.add_argument('--salida',type=Path,default=Path(__file__).resolve().parent/'resultados_seguimiento')
    p.add_argument('--tolerancia',type=float,default=5)
    p.add_argument('--cobertura-minima',type=float,default=.5)
    p.add_argument('--margen',type=float,default=.1)
    p.add_argument('--fraccion-inferior',type=float,default=1.)
    args = p.parse_args()
    if not args.video.is_file():p.error(f'No existe el video: {args.video}')
    if not args.tracker.is_file():p.error(f'No existe el YAML: {args.tracker}')
    if args.zonas and not args.zonas.is_file():p.error(f'No existe el JSON de zonas: {args.zonas}')
    if args.imgsz<1 or args.trayectoria<2 or args.max_frames<0:p.error('Revisa imgsz, trayectoria y max-frames.')
    numeros=(args.conf,args.tolerancia,args.cobertura_minima,args.margen,args.fraccion_inferior)
    if not all(math.isfinite(v) for v in numeros):p.error('Los números deben ser finitos.')
    if not (args.tolerancia>=0 and 0<args.cobertura_minima<=1 and 0<=args.margen<=1 and 0<args.fraccion_inferior<=1):
        p.error('Parámetros geométricos fuera de rango.')
    try:cfg=leer_tracker(args.tracker,args.conf)
    except (ValueError,TypeError) as exc:p.error(str(exc))
    return args,cfg


def main():
    args,cfg = argumentos()
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    captura = cv2.VideoCapture(str(args.video.resolve()))
    try:
        ok,primero = captura.read()
        if not captura.isOpened() or not ok:raise RuntimeError('No se pudo abrir o leer el video.')
        fps = float(captura.get(cv2.CAP_PROP_FPS))
        total_nominal = float(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        captura.release()
    fps = fps if math.isfinite(fps) and fps>0 else 0.
    total_nominal = total_nominal if math.isfinite(total_nominal) and total_nominal>0 else 0.
    if args.guardar_video and fps<=0:
        raise ValueError('No hay FPS nominales válidos para escribir el MP4. Ejecuta sin --guardar-video.')
    alto,ancho = primero.shape[:2]
    config = leer_config(args.zonas,ancho,alto) if args.zonas else None
    zonas = preparar_zonas(config,ancho,alto,args.tolerancia) if config else []
    reglas = dict(minimo=args.cobertura_minima,margen=args.margen,
                  fraccion_inferior=args.fraccion_inferior,tolerancia=args.tolerancia)
    model = YOLO(args.modelo)
    if set(VEHICULOS)-set(model.names.values()):raise ValueError('Al modelo le faltan clases de vehículos.')
    params = dict(device=args.device,imgsz=args.imgsz,conf=args.conf,iou=.7,
                  agnostic_nms=args.nms_agnostica,
                  classes=[i for i,nombre in model.names.items() if nombre in VEHICULOS],
                  save=False,verbose=False)
    # Calentamos solo el detector. Ningún frame se procesa dos veces por el tracker.
    model.predict(primero,**params)
    ahora = datetime.now(timezone.utc)
    carpeta = args.salida.resolve()/ahora.strftime('%Y%m%dT%H%M%S_%fZ')
    carpeta.mkdir(parents=True,exist_ok=False)
    tracker_local = carpeta/'tracker_usado.yaml'
    tracker_local.write_text(args.tracker.read_text(encoding='utf-8'),encoding='utf-8')
    historial = Historial(args.trayectoria)
    columnas_tracks = ['frame','tiempo_s','tipo_tiempo','track_id','clase','confianza',
                       'x1','y1','x2','y2','punto_x','punto_y','zona_id','estado_zona',
                       'frames_ausentes_previos']
    columnas_frames = ['frame','tiempo_s','tipo_tiempo','ids_presentes','salidas_sin_id',
                       'ids_distintos_publicados','analisis_ms','lectura_ms',
                       'asignado','ambiguo','fuera','sin_zonas']+[f"{z['id']}_vehiculos" for z,_,_ in zonas]
    captura = cv2.VideoCapture(str(args.video.resolve()))
    video_salida=None;ultima=None;datos_ultimo=None;tiempos=[];tipos=set()
    n=0;anterior=-1.;motivo='error';error=None;sin_id_total=0;frames_vacios=0
    inicio = perf_counter()
    print('Seguimiento: YOLO11 + ByteTrack. IDs publicados no equivalen a vehículos únicos reales.')
    print('Zonas fijas; sin compensación de cámara. Se procesan todos los fotogramas en orden.')
    try:
        if not captura.isOpened():raise RuntimeError('No se pudo reabrir el video.')
        with (carpeta/'tracks.csv').open('w',newline='',encoding='utf-8') as ft, \
             (carpeta/'conteos.csv').open('w',newline='',encoding='utf-8') as fc:
            wt=csv.DictWriter(ft,fieldnames=columnas_tracks);wt.writeheader()
            wc=csv.DictWriter(fc,fieldnames=columnas_frames);wc.writeheader()
            while True:
                inicio_lectura=perf_counter();ok,frame=captura.read()
                lectura_ms=(perf_counter()-inicio_lectura)*1000
                if not ok:
                    motivo='lectura_incompleta' if total_nominal and n<int(total_nominal) else 'fin_lectura_archivo'
                    break
                if frame.shape[:2]!=(alto,ancho):raise ValueError('Cambió la resolución del video.')
                t,tipo=tiempo_video(n,captura.get(cv2.CAP_PROP_POS_MSEC),fps,anterior)
                anterior=t;tipos.add(tipo)
                inicio_analisis=perf_counter()
                # persist=True conserva el tracker entre las llamadas de este mismo video.
                result=model.track(frame,persist=True,tracker=str(tracker_local),**params)[0]
                objetos,sin_id=extraer_objetos(result)
                conteos,estados=asignar_y_contar(objetos,zonas,reglas)
                historial.actualizar(objetos,n,t)
                ms=(perf_counter()-inicio_analisis)*1000;tiempos.append(ms)
                sin_id_total+=sin_id;frames_vacios+=int(not objetos)
                for o in objetos:
                    x1,y1,x2,y2=o['xyxy'];px,py=o['punto_xy'];a=o['asignacion']
                    wt.writerow(dict(frame=n,tiempo_s=t,tipo_tiempo=tipo,track_id=o['track_id'],
                                     clase=o['clase'],confianza=o['confianza'],x1=x1,y1=y1,x2=x2,y2=y2,
                                     punto_x=px,punto_y=py,zona_id=a['zona_id'] or '',estado_zona=a['estado'],
                                     frames_ausentes_previos=o['frames_ausentes_previos']))
                wc.writerow(dict(frame=n,tiempo_s=t,tipo_tiempo=tipo,ids_presentes=len(objetos),
                                 salidas_sin_id=sin_id,ids_distintos_publicados=len(historial.fichas),
                                 analisis_ms=ms,lectura_ms=lectura_ms,**estados,
                                 **{f'{z}_vehiculos':c for z,c in conteos.items()}))
                datos_ultimo={'frame':n,'tiempo_s':t,'tipo_tiempo':tipo,'objetos':objetos,
                              'conteos_zona':conteos,'estados':estados,'salidas_sin_id':sin_id}
                ultima=crear_vista(frame,objetos,zonas,conteos,estados,historial,n,t,ms,sin_id,cv2,np)
                if args.guardar_video:
                    if video_salida is None:
                        video_salida=cv2.VideoWriter(str(carpeta/'seguimiento.mp4'),
                                                     cv2.VideoWriter_fourcc(*'mp4v'),fps,
                                                     (ultima.shape[1],ultima.shape[0]))
                        if not video_salida.isOpened():raise RuntimeError('No se pudo crear el MP4 (codec mp4v).')
                    video_salida.write(ultima)
                n+=1
                if not args.sin_preview:
                    cv2.imshow('Seguimiento | Q o Esc',ultima)
                    tecla=cv2.waitKey(1)&0xFF
                    if tecla in (27,ord('q'),ord('Q')) or cv2.getWindowProperty('Seguimiento | Q o Esc',cv2.WND_PROP_VISIBLE)<1:
                        motivo='usuario';break
                if n%30==0:
                    ft.flush();fc.flush()
                    print(f'Frame {n-1} | t={t:.2f}s | IDs presentes={len(objetos)} | analisis={ms:.1f}ms')
                if args.max_frames and n>=args.max_frames:
                    motivo='limite_fotogramas';break
    except KeyboardInterrupt:
        motivo='interrupcion_teclado'
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';motivo='error'
    finally:
        duracion=perf_counter()-inicio
        captura.release()
        if video_salida is not None:video_salida.release()
        if not args.sin_preview:cv2.destroyAllWindows()
        filas=list(historial.filas_resumen())
        columnas_ids=['track_id','primer_frame','ultimo_frame','primer_tiempo_s','ultimo_tiempo_s',
                      'intervalo_primera_ultima_s','observaciones','reapariciones','cambios_clase','clase_mas_frecuente']
        with (carpeta/'ids.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=columnas_ids);writer.writeheader();writer.writerows(filas)
        avisos=[]
        if ultima is not None:
            ok,jpg=cv2.imencode('.jpg',ultima)
            if ok:jpg.tofile(str(carpeta/'ultimo_frame.jpg'))
            else:avisos.append('No se pudo codificar ultimo_frame.jpg.')
            (carpeta/'ultimo_frame.json').write_text(json.dumps(datos_ultimo,indent=2,ensure_ascii=False),encoding='utf-8')
        resumen={'fecha_utc':ahora.isoformat(),'fuente':str(args.video.resolve()),
                 'motivo_fin':motivo,'error':error,'avisos':avisos,'fotogramas_procesados':n,
                 'fps_fuente_reportados':fps,'frames_fuente_reportados':total_nominal,
                 'ultimo_tiempo_fuente_s':anterior if n else None,'tipos_tiempo':sorted(tipos),
                 'duracion_bucle_s':duracion,'fps_bucle_observados':n/duracion if duracion else 0,
                 'analisis_ms_media':float(np.mean(tiempos)) if tiempos else None,
                 'analisis_ms_p95':float(np.percentile(tiempos,95)) if tiempos else None,
                 'ids_distintos_publicados':len(filas),'fotogramas_sin_ids':frames_vacios,
                 'salidas_sin_id_acumuladas':sin_id_total,
                 'reapariciones_mismo_id':sum(f['reapariciones'] for f in filas),
                 'ids_con_hasta_2_observaciones':sum(f['observaciones']<=2 for f in filas),
                 'modelo':args.modelo,'imgsz':args.imgsz,'conf_candidatos':args.conf,'iou_nms':.7,
                 'agnostic_nms':args.nms_agnostica,'clases':list(VEHICULOS),'device':args.device,
                 'tracker':cfg,'trayectoria_max_puntos':args.trayectoria,
                 'configuracion_zonas':config,'reglas':reglas,
                 'video_anotado_solicitado':args.guardar_video,'codec_video':'mp4v' if args.guardar_video else None,
                 'versiones':{'python':platform.python_version(),'torch':torch.__version__,
                             'ultralytics':ultralytics.__version__,'opencv':cv2.__version__},
                 'metricas':'Analisis incluye detector, tracker, copia a CPU, zonas e historial. Bucle incluye lectura, CSV, dibujo y MP4; excluye carga y calentamiento del detector.',
                 'alcance':'IDs y trayectorias en coordenadas de imagen. Sin compensación, velocidad física, flujo ni cola. IDs distintos no equivalen a vehículos reales únicos.',
                 'conteos':'Solo cajas con ID publicadas en el frame actual. Tracks perdidos no se suman. No sumar filas para calcular flujo.',
                 'tiempo':'Tiempo del archivo, no reloj de procesamiento. El MP4 usa FPS nominales constantes; para fuente con cadencia variable consultar tiempos del CSV.',
                 'diagnosticos':'Reapariciones y tracks cortos no miden ID switches ni precisión; requieren revisión visual.'}
        (carpeta/'resumen.json').write_text(json.dumps(resumen,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Finalizado: {motivo}. Frames: {n}. Resultados: {carpeta}')
    if error:raise RuntimeError(error)


if __name__=='__main__':
    main()
