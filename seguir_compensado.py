"""Seguimiento compensado, perfil de rendimiento y proxy opcional de cola visible.

Requiere compensacion_camara.py, seguir_video.py, contar_video.py,
contar_multizonas.py y bytetrack_tesis.yaml en esta carpeta.
"""
import argparse
from dataclasses import asdict
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np
from shapely.geometry import Polygon
from compensacion_camara import CompensadorCamara, transformar, esquinas_caja, caja_en_referencia
from contar_multizonas import VEHICULOS, leer_config, editar_zonas, preparar_zonas, decidir, ascii_text
from contar_video import tiempo_video
from seguir_video import Historial, color_id, leer_tracker
from estado_cola import EstimadorCola, agregar_argumentos, parametros_argumentos


def json_escribir(ruta, contenido):
    ruta.write_text(json.dumps(contenido,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def imagen_escribir(ruta, imagen):
    ok,datos=cv2.imencode(ruta.suffix,imagen)
    if not ok:raise RuntimeError(f'No se pudo codificar {ruta.name}')
    datos.tofile(str(ruta))


def cargar_o_editar(ruta, editar, frame):
    h,w=frame.shape[:2]
    if ruta is None:return None
    if not ruta.exists() and not editar:
        raise ValueError(f'No existe {ruta}; usa la opción de edición para crearlo.')
    cfg=leer_config(ruta,w,h) if ruta.exists() else {'version':2,'alto_ancho':[h,w],'zonas':[]}
    if editar:
        zonas=editar_zonas(frame,cfg['zonas'],12,cv2,np)
        if zonas is None:raise KeyboardInterrupt('Edición cancelada.')
        cfg['zonas']=zonas;cfg['referencia_frame']=0
        ruta.parent.mkdir(parents=True,exist_ok=True);json_escribir(ruta,cfg)
    if cfg.get('referencia_frame',0)!=0:
        raise ValueError('Esta versión requiere zonas y fondo definidos sobre el primer fotograma.')
    return cfg


def crear_mascara(frame, fondo, franja):
    h,w=frame.shape[:2];mask=np.zeros((h,w),np.uint8)
    if fondo:
        for z in fondo['zonas']:
            # Cada polígono añade superficie: el solapamiento no crea agujeros.
            cv2.fillPoly(mask,[np.rint(z['puntos_xy']).astype(np.int32)],255)
    else:
        mask[:max(1,round(h*franja)),:]=255
    return mask


def asignacion_compensada(caja, matriz, zonas, reglas):
    """Transforma el polígono real de la caja, sin inflar su área por rotación."""
    if not zonas:return {'estado':'sin_zonas','zona_id':None}
    if matriz is None:return {'estado':'alineacion_invalida','zona_id':None}
    region=Polygon(transformar(esquinas_caja(caja,reglas['fraccion_inferior']),matriz))
    def puntajes(ampliado):
        return {z['id']:min(1.,max(0.,region.intersection(b if ampliado else p).area/region.area))
                for z,p,b in zonas}
    scores=puntajes(False);estado,ident=decidir(scores,reglas['minimo'],reglas['margen']);modo='original'
    if estado=='fuera' and reglas['tolerancia']>0:
        scores=puntajes(True);estado,ident=decidir(scores,reglas['minimo'],reglas['margen']);modo='tolerancia'
    return {'estado':estado,'zona_id':ident,'coberturas':scores,'evaluacion':modo}


class Identidades:
    """Evita reutilizar un ID público después de reiniciar ByteTrack por fallo de cámara."""
    def __init__(self):self.mapa={};self.siguiente=1;self.epoca=0
    def reiniciar(self):self.mapa.clear();self.epoca+=1
    def publicar(self, local):
        if local not in self.mapa:
            self.mapa[local]=self.siguiente;self.siguiente+=1
        return self.mapa[local]


def combinar(candidatos, tracks, matriz, identidades, umbral, zonas, reglas, nombres):
    """Una observación por candidato: track asociado O detección fuerte sin ID."""
    asociados={};vistos=set()
    for fila in tracks:
        if len(fila)!=8:raise ValueError('Salida ByteTrack incompatible: se esperan 8 columnas para detección.')
        if not np.isfinite(fila).all():raise ValueError('ByteTrack devolvió datos no finitos.')
        indice=int(fila[7]);local=int(fila[4])
        if fila[7]!=indice or fila[4]!=local or not 0<=indice<len(candidatos) or indice in asociados or local in vistos:
            raise ValueError('Índices o IDs inconsistentes en ByteTrack.')
        asociados[indice]=local;vistos.add(local)
    observaciones=[]
    for indice,fila in enumerate(candidatos):
        caja=fila[:4].tolist();conf=float(fila[4]);clase=nombres[int(fila[5])]
        local=asociados.get(indice)
        if local is None and conf<umbral:continue
        ident=identidades.publicar(local) if local is not None else None
        punto=[(caja[0]+caja[2])/2,caja[3]]
        punto_ref=transformar([punto],matriz)[0].tolist() if matriz is not None else None
        a=asignacion_compensada(caja,matriz,zonas,reglas)
        observaciones.append({'indice_candidato':indice,'track_id':ident,'track_id_local':local,
                              'epoca_tracker':identidades.epoca,'clase':clase,'confianza':conf,
                              'tipo':'con_id' if ident is not None else 'sin_id',
                              'xyxy_actual':caja,'punto_actual_xy':punto,
                              'xyxy':caja_en_referencia(caja,matriz) if matriz is not None else None,
                              'punto_xy':punto_ref,'asignacion':a})
    return observaciones


def contar(obs, zonas, matriz, forma):
    counts={z['id']:{'con_id':0,'sin_id':0,'total':0,'fraccion_visible':None} for z,_,_ in zonas}
    if matriz is None:
        for c in counts.values():c.update(con_id=None,sin_id=None,total=None)
    else:
        h,w=forma
        huella=Polygon(transformar([[0,0],[w,0],[w,h],[0,h]],matriz))
        for z,p,_ in zonas:counts[z['id']]['fraccion_visible']=min(1.,p.intersection(huella).area/p.area)
        for o in obs:
            ident=o['asignacion']['zona_id']
            if ident is not None:
                counts[ident][o['tipo']]+=1;counts[ident]['total']+=1
    return counts


def dibujar(frame, obs, zonas, matriz, info, historial, cuentas, frame_id, tiempo, cola=None):
    h,w=frame.shape[:2];escala=min(1280/w,720/h,1.)
    vw,vh=max(2,round(w*escala)),max(2,round(h*escala))
    vista=cv2.resize(frame,(vw,vh))
    def punto(p):return round(p[0]*vw/w),round(p[1]*vh/h)
    inversa=cv2.invertAffineTransform(matriz) if matriz is not None else None
    if inversa is not None:
        for z,_,_ in zonas:
            pts=transformar(z['puntos_xy'],inversa)
            vertices=np.array([punto(p) for p in pts],np.int32)
            cv2.polylines(vista,[vertices],True,(255,255,0),2)
            cv2.putText(vista,ascii_text(z['id']),tuple(vertices[0]),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,0),1)
    for o in obs:
        ident=o['track_id'];color=color_id(ident) if ident is not None else (0,185,255)
        x1,y1,x2,y2=o['xyxy_actual']
        cv2.rectangle(vista,punto([x1,y1]),punto([x2,y2]),color,1)
        texto=f"ID {ident}" if ident is not None else 'sin ID'
        movimiento=o.get('movimiento',{})
        if movimiento.get('en_cola_proxy'):texto+=' COLA*'
        elif movimiento.get('estado')=='sin_historial':texto+=' ?'
        x,y=punto([x1,y1]);cv2.putText(vista,f"{texto} {o['clase']}",(x,max(14,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
        if ident is not None and inversa is not None:
            pts=[(x,y) for _,_,x,y in historial.puntos.get(ident,())]
            if len(pts)>1:
                linea=np.array([punto(p) for p in transformar(pts,inversa)],np.int32)
                cv2.polylines(vista,[linea],False,color,2)
    ah=max(vh,350+36*len(zonas)+55*len(cola or {}));aw=vw+460
    salida=np.full((ah+ah%2,aw+aw%2,3),28,np.uint8);salida[:vh,:vw]=vista
    def linea(s,y,color=(235,235,235)):
        cv2.putText(salida,ascii_text(s)[:49],(vw+10,y),cv2.FONT_HERSHEY_SIMPLEX,.44,color,1)
    con=sum(o['track_id'] is not None for o in obs)
    linea(f'Frame {frame_id} | t={tiempo:.2f}s',25)
    linea(f"Camara: {info['motivo']}",52,(70,220,70) if info['valida'] else (0,80,255))
    linea(f"Parejas {info['parejas']} | inliers {info['inliers']}",76)
    err=info.get('error_mediano_px');linea(f'Error fondo: {err:.2f} px' if err is not None else 'Error fondo: N/D',100)
    linea(f'Con ID: {con} | sin ID: {len(obs)-con}',131)
    linea(f'Observaciones actuales: {len(obs)}',155)
    linea('Cola*: proxy visible, NO flujo' if cola else 'Conteo visible, NO cola ni flujo',179,(0,190,255))
    y=210
    for z,_,_ in zonas:
        c=cuentas[z['id']]
        if c['total'] is None:linea(f"{z['id']}: N/D (camara)",y,(0,80,255))
        else:
            sufijo=' PARCIAL' if c['fraccion_visible']<.95 else ''
            linea(f"{z['id']}: {c['total']} ({c['con_id']} ID + {c['sin_id']} s/ID){sufijo}",y)
        y+=36
    if not zonas:linea('Sin zonas; vista completa',y);y+=25
    if cola:
        for ident,c in cola.items():
            cantidad=c['cola_visible_proxy']
            linea(f"{ident}: cola*={cantidad if cantidad is not None else 'N/D'}",y,(0,190,255))
            y+=23
            linea(f"Sin clasificar: {c['indeterminados']} | {c['estado']}",y);y+=32
    linea('Trayectorias en referencia del frame 0',y)
    linea('Q / Esc: terminar',y+25)
    return salida


def argumentos():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--zonas',type=Path)
    p.add_argument('--editar-zonas',action='store_true')
    p.add_argument('--fondo',type=Path,help='Polígonos de estructuras estáticas; opcional')
    p.add_argument('--editar-fondo',action='store_true')
    p.add_argument('--franja-fondo',type=float,default=.40,help='Sin JSON: fracción superior del frame 0 para fondo')
    p.add_argument('--sin-compensacion',action='store_true',help='Referencia identidad: presupone cámara fija')
    p.add_argument('--modelo',default='yolo11n.pt');p.add_argument('--device',default='0')
    p.add_argument('--imgsz',type=int,default=960);p.add_argument('--conf',type=float,default=.10)
    p.add_argument('--conf-observacion',type=float,default=.25)
    p.add_argument('--nms-agnostica',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--tracker',type=Path,default=Path(__file__).with_name('bytetrack_tesis.yaml'))
    p.add_argument('--trayectoria',type=int,default=60);p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--guardar-video',action='store_true');p.add_argument('--sin-preview',action='store_true')
    p.add_argument('--salida',type=Path,default=Path(__file__).parent/'resultados_compensados')
    p.add_argument('--tolerancia',type=float,default=5);p.add_argument('--cobertura-minima',type=float,default=.5)
    p.add_argument('--margen',type=float,default=.1);p.add_argument('--fraccion-inferior',type=float,default=1.)
    p.add_argument('--puntos-fondo',type=int,default=5000,help='Máximo de características ORB por imagen')
    p.add_argument('--ancho-fondo',type=int,default=960)
    agregar_argumentos(p)
    a=p.parse_args()
    if not a.video.is_file() or not a.tracker.is_file():p.error('Video o YAML inexistente.')
    if a.editar_zonas and not a.zonas:p.error('--editar-zonas requiere --zonas archivo.json')
    if a.editar_fondo and not a.fondo:p.error('--editar-fondo requiere --fondo archivo.json')
    if a.puntos_fondo<100 or a.ancho_fondo<64:p.error('puntos-fondo >=100 y ancho-fondo >=64.')
    if a.zonas_cola and not a.zonas:p.error('--zonas-cola requiere --zonas.')
    try:parametros_argumentos(a)
    except ValueError as exc:p.error(str(exc))
    if a.imgsz<1 or a.trayectoria<2 or a.max_frames<0:p.error('Revisa imgsz, trayectoria y max-frames.')
    valores=(a.conf,a.conf_observacion,a.franja_fondo,a.tolerancia,a.cobertura_minima,a.margen,a.fraccion_inferior)
    if not all(math.isfinite(x) for x in valores):p.error('Los parámetros deben ser finitos.')
    if not (0<a.franja_fondo<=1 and a.conf<=a.conf_observacion<=1 and a.tolerancia>=0 and 0<a.cobertura_minima<=1 and 0<=a.margen<=1 and 0<a.fraccion_inferior<=1):p.error('Parámetros fuera de rango.')
    try:cfg=leer_tracker(a.tracker,a.conf)
    except (ValueError,TypeError) as exc:p.error(str(exc))
    return a,cfg


def main():
    a,cfg=argumentos()
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    cap=cv2.VideoCapture(str(a.video.resolve()))
    try:
        ok,ref=cap.read();fps=float(cap.get(cv2.CAP_PROP_FPS));nominal=float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not ok:raise RuntimeError('No se pudo leer el primer fotograma.')
    finally:cap.release()
    fps=fps if math.isfinite(fps) and fps>0 else 0.
    nominal=nominal if math.isfinite(nominal) and nominal>0 else 0.
    if a.guardar_video and not fps:raise ValueError('Se necesitan FPS válidos para guardar MP4.')
    h,w=ref.shape[:2]
    try:
        config=cargar_o_editar(a.zonas,a.editar_zonas,ref)
        fondo=cargar_o_editar(a.fondo,a.editar_fondo,ref)
    except KeyboardInterrupt:
        print('Edición cancelada.');return
    zonas=preparar_zonas(config,w,h,a.tolerancia) if config else []
    seleccion=set(a.zonas_cola or [])
    if seleccion-{z['id'] for z,_,_ in zonas}:raise ValueError('Hay IDs de --zonas-cola que no existen en el JSON.')
    estimador=EstimadorCola(seleccion,parametros_argumentos(a)) if seleccion else None
    reglas=dict(minimo=a.cobertura_minima,margen=a.margen,fraccion_inferior=a.fraccion_inferior,tolerancia=a.tolerancia)
    model=YOLO(a.modelo)
    if set(VEHICULOS)-set(model.names.values()):raise ValueError('Faltan clases de vehículos en el modelo.')
    params=dict(device=a.device,imgsz=a.imgsz,conf=a.conf,iou=.7,agnostic_nms=a.nms_agnostica,
                classes=[i for i,n in model.names.items() if n in VEHICULOS],save=False,verbose=False)
    primero=model.predict(ref,**params)[0]
    mask=crear_mascara(ref,fondo,a.franja_fondo)
    compensador=None if a.sin_compensacion else CompensadorCamara(ref,mask,primero.boxes.xyxy.cpu().numpy(),ancho_analisis=a.ancho_fondo,max_puntos=a.puntos_fondo)
    ahora=datetime.now(timezone.utc);carpeta=a.salida.resolve()/ahora.strftime('%Y%m%dT%H%M%S_%fZ');carpeta.mkdir(parents=True)
    imagen_escribir(carpeta/'referencia.png',ref)
    fondo_vista=ref.copy();fondo_vista[mask==0]=(fondo_vista[mask==0]*.25).astype(np.uint8)
    imagen_escribir(carpeta/'referencia_fondo.jpg',fondo_vista)
    if config:json_escribir(carpeta/'zonas_referencia.json',config)
    if fondo:json_escribir(carpeta/'fondo_referencia.json',fondo)
    (carpeta/'tracker_usado.yaml').write_text(a.tracker.read_text(encoding='utf-8'),encoding='utf-8')
    tracker=BYTETracker(SimpleNamespace(**cfg));identidades=Identidades();historial=Historial(a.trayectoria)
    anterior_valida=True;n=0;prev=-1.;tiempos=[];calidades=Counter();tipos=set();motivo='error';error=None;ultima=None;ultimo=None;writer_video=None
    rendimientos=[];cola=None
    columnas_perfil=['frame','tiempo_s','lectura_ms','deteccion_ms','compensacion_ms','seguimiento_ms','zonas_ms','cola_ms','exportacion_ms','dibujo_ms','video_ms','preview_ms','total_frame_ms']
    cols=['frame','tiempo_s','tipo_tiempo','camara_valida','con_id','sin_id','observaciones','candidatos','analisis_ms']
    for z,_,_ in zonas:cols += [f"{z['id']}_{k}" for k in ('con_id','sin_id','total','fraccion_visible')]
    cols_cam=['frame','tiempo_s','valida','motivo','parejas','inliers','proporcion_inliers','error_mediano_px','escala','rotacion_grados','vista_comun']+[f'm{i}{j}' for i in range(2) for j in range(3)]
    cols_obs=['frame','tiempo_s','camara_valida','track_id','track_id_local','epoca_tracker','tipo','clase','confianza','x_actual','y_actual','x_ref','y_ref','zona_id','estado_zona']
    cap=cv2.VideoCapture(str(a.video.resolve()));inicio=perf_counter()
    print('Compensación a frame 0. El fondo debe ser estático. Si falla la alineación, zonas=N/D.')
    try:
        if not cap.isOpened():raise RuntimeError('No se pudo reabrir el video.')
        with (carpeta/'conteos.csv').open('w',newline='',encoding='utf-8') as fc, (carpeta/'camara.csv').open('w',newline='',encoding='utf-8') as fm, (carpeta/'observaciones.csv').open('w',newline='',encoding='utf-8') as fo, (carpeta/'observaciones.jsonl').open('w',encoding='utf-8') as fj, (carpeta/'rendimiento.csv').open('w',newline='',encoding='utf-8') as fp:
            wp=csv.DictWriter(fp,fieldnames=columnas_perfil);wp.writeheader()
            wc=csv.DictWriter(fc,fieldnames=cols);wc.writeheader();wm=csv.DictWriter(fm,fieldnames=cols_cam);wm.writeheader();wo=csv.DictWriter(fo,fieldnames=cols_obs);wo.writeheader()
            while True:
                inicio_frame=perf_counter();ok,frame=cap.read()
                perfil={'lectura_ms':(perf_counter()-inicio_frame)*1000}
                if not ok:
                    motivo='lectura_incompleta' if nominal and n<int(nominal) else 'fin_lectura_archivo';break
                if frame.shape[:2]!=(h,w):raise ValueError('Cambió la resolución.')
                t,tipo=tiempo_video(n,cap.get(cv2.CAP_PROP_POS_MSEC),fps,prev);prev=t;tipos.add(tipo)
                inicio_analisis=perf_counter();r=model.predict(frame,**params)[0]
                b=r.boxes.cpu().numpy()
                candidatos=np.column_stack((b.xyxy,b.conf,b.cls)).astype(np.float32)
                perfil['deteccion_ms']=(perf_counter()-inicio_analisis)*1000;tic=perf_counter()
                if a.sin_compensacion or n==0:
                    matriz=np.array([[1.,0.,0.],[0.,1.,0.]])
                    info=dict(valida=True,motivo='desactivada' if a.sin_compensacion else 'referencia',parejas=0,inliers=0,proporcion_inliers=0.,error_mediano_px=None,escala=1.,rotacion_grados=0.,vista_comun=1.,matriz_actual_a_ref=matriz.tolist())
                else:matriz,info=compensador.estimar(frame,candidatos[:,:4])
                perfil['compensacion_ms']=(perf_counter()-tic)*1000;tic=perf_counter()
                valida=matriz is not None;calidades[info['motivo']]+=1
                if not valida and anterior_valida:
                    tracker.reset();identidades.reiniciar();historial.puntos.clear()
                anterior_valida=valida
                if valida:
                    referencia=candidatos.copy()
                    for i,caja in enumerate(candidatos[:,:4]):referencia[i,:4]=caja_en_referencia(caja,matriz)
                    tracks=tracker.update(Boxes(referencia,(h,w)))
                else:tracks=[]
                perfil['seguimiento_ms']=(perf_counter()-tic)*1000;tic=perf_counter()
                obs=combinar(candidatos,tracks,matriz,identidades,a.conf_observacion,zonas,reglas,r.names)
                con_id=[o for o in obs if o['track_id'] is not None]
                historial.actualizar(con_id,n,t);cuentas=contar(obs,zonas,matriz,(h,w))
                perfil['zonas_ms']=(perf_counter()-tic)*1000;tic=perf_counter()
                cola=estimador.actualizar(n,t,obs,cuentas,valida) if estimador else None
                perfil['cola_ms']=(perf_counter()-tic)*1000
                ms=(perf_counter()-inicio_analisis)*1000;tiempos.append(ms);tic=perf_counter()
                fila=dict(frame=n,tiempo_s=t,tipo_tiempo=tipo,camara_valida=valida,con_id=len(con_id),sin_id=len(obs)-len(con_id),observaciones=len(obs),candidatos=len(candidatos),analisis_ms=ms)
                fila.update({f'{z}_{k}':v for z,c in cuentas.items() for k,v in c.items()});wc.writerow(fila)
                fm_fila={k:info.get(k) for k in cols_cam};fm_fila.update(frame=n,tiempo_s=t)
                if valida:fm_fila.update({f'm{i}{j}':float(matriz[i,j]) for i in range(2) for j in range(3)})
                wm.writerow(fm_fila)
                for o in obs:
                    px,py=o['punto_actual_xy'];pr=o['punto_xy'] or [None,None];asig=o['asignacion']
                    wo.writerow(dict(frame=n,tiempo_s=t,camara_valida=valida,track_id=o['track_id'],track_id_local=o['track_id_local'],epoca_tracker=o['epoca_tracker'],tipo=o['tipo'],clase=o['clase'],confianza=o['confianza'],x_actual=px,y_actual=py,x_ref=pr[0],y_ref=pr[1],zona_id=asig['zona_id'],estado_zona=asig['estado']))
                ultimo=dict(frame=n,tiempo_s=t,tipo_tiempo=tipo,camara=info,observaciones=obs,conteos_zona=cuentas,cola=cola)
                fj.write(json.dumps(ultimo,ensure_ascii=False,allow_nan=False)+'\n')
                perfil['exportacion_ms']=(perf_counter()-tic)*1000;tic=perf_counter()
                ultima=dibujar(frame,obs,zonas,matriz,info,historial,cuentas,n,t,cola)
                perfil['dibujo_ms']=(perf_counter()-tic)*1000;tic=perf_counter()
                n+=1
                if a.guardar_video:
                    if writer_video is None:
                        writer_video=cv2.VideoWriter(str(carpeta/'seguimiento_compensado.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(ultima.shape[1],ultima.shape[0]))
                        if not writer_video.isOpened():raise RuntimeError('No se pudo crear MP4.')
                    writer_video.write(ultima)
                perfil['video_ms']=(perf_counter()-tic)*1000;tic=perf_counter();terminar=False
                if not a.sin_preview:
                    cv2.imshow('Compensacion | Q/Esc',ultima);tecla=cv2.waitKey(1)&0xFF
                    if tecla in (27,ord('q'),ord('Q')) or cv2.getWindowProperty('Compensacion | Q/Esc',cv2.WND_PROP_VISIBLE)<1:
                        motivo='usuario';terminar=True
                perfil['preview_ms']=(perf_counter()-tic)*1000
                perfil.update(frame=n-1,tiempo_s=t,total_frame_ms=(perf_counter()-inicio_frame)*1000)
                wp.writerow(perfil);rendimientos.append(perfil)
                if terminar:break
                if n%30==0:
                    fc.flush();fm.flush();fo.flush();fj.flush();fp.flush();print(f"Frame {n-1} t={t:.2f}s camara={info['motivo']} con_ID={len(con_id)} sin_ID={len(obs)-len(con_id)}")
                if a.max_frames and n>=a.max_frames:motivo='limite_fotogramas';break
    except KeyboardInterrupt:motivo='interrupcion_teclado'
    except Exception as exc:motivo='error';error=f'{type(exc).__name__}: {exc}'
    finally:
        duracion=perf_counter()-inicio;cap.release()
        if writer_video is not None:writer_video.release()
        if not a.sin_preview:cv2.destroyAllWindows()
        rows=list(historial.filas_resumen())
        cols_ids=['track_id','primer_frame','ultimo_frame','primer_tiempo_s','ultimo_tiempo_s','intervalo_primera_ultima_s','observaciones','reapariciones','cambios_clase','clase_mas_frecuente']
        with (carpeta/'ids.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=cols_ids);writer.writeheader();writer.writerows(rows)
        validos=sum(calidades[k] for k in ('ok','referencia','desactivada'))
        resumen=dict(fecha_utc=ahora.isoformat(),fuente=str(a.video.resolve()),motivo_fin=motivo,error=error,
                     fotogramas_procesados=n,frames_fuente_reportados=nominal,fps_fuente_reportados=fps,
                     ultimo_tiempo_fuente_s=prev if n else None,tipos_tiempo=sorted(tipos),
                     duracion_bucle_s=duracion,fps_bucle_observados=n/duracion if duracion else 0,
                     analisis_ms_media=float(np.mean(tiempos)) if tiempos else None,
                     analisis_ms_p95=float(np.percentile(tiempos,95)) if tiempos else None,
                     compensacion_activada=not a.sin_compensacion,camara_frames_validos=validos,
                     camara_frames_invalidos=sum(calidades.values())-validos,calidad_camara=dict(calidades),
                     parametros_compensacion=compensador.parametros if compensador else None,
                     reinicios_por_alineacion=identidades.epoca,ids_distintos_publicados=len(rows),
                     configuracion_zonas=config,configuracion_fondo=fondo,franja_fondo=a.franja_fondo if not fondo else None,
                     perfil_ms={k:dict(media=float(np.mean([r[k] for r in rendimientos])),p95=float(np.percentile([r[k] for r in rendimientos],95))) for k in columnas_perfil[2:]} if rendimientos else {},
                     cola_parametros=asdict(estimador.p) if estimador else None,zonas_cola=sorted(seleccion),
                     device=a.device,video_anotado_solicitado=a.guardar_video,
                     reglas=reglas,modelo=a.modelo,imgsz=a.imgsz,conf_candidatos=a.conf,conf_observacion=a.conf_observacion,
                     iou_nms=.7,agnostic_nms=a.nms_agnostica,tracker=cfg,
                     versiones=dict(python=platform.python_version(),torch=torch.__version__,ultralytics=ultralytics.__version__,opencv=cv2.__version__),
                     alcance='Observaciones presentes y proxy opcional de cola visible (movimiento reducido persistente). Sin causalidad semafórica, flujo, metros ni vehículos ocultos.',
                     referencia='Primer frame del mismo archivo. M: actual a referencia. Puntos en píxeles originales; no metros.',
                     fallos='Alineación inválida: zonas null, sin puntos de referencia ni IDs; detecciones fuertes de imagen siguen disponibles. Reset del tracker y nuevos IDs públicos al recuperar.',
                     conteos='Una observación por candidato: ID asociado (incluso débil) o detección sin ID >= conf_observacion. No se suman tracks perdidos. No equivale al conteo anterior de solo IDs.',
                     metricas='Análisis incluye detección con transferencia a CPU, registro, tracker, geometría y cola. rendimiento.csv separa etapas. Total por frame excluye escribir su propia fila y flush periódico; bucle incluye todo salvo edición, inicialización, carga y exportación final.',
                     advertencia='RANSAC y los límites son controles heurísticos, no garantía de alineación de toda la calle. Revisar perspectiva/paralaje. MP4 a FPS nominales constantes, sin audio.')
        json_escribir(carpeta/'resumen.json',resumen)
        if ultima is not None:imagen_escribir(carpeta/'ultimo_frame.jpg',ultima);json_escribir(carpeta/'ultimo_frame.json',ultimo)
    print(f'Finalizado: {motivo}. Resultados: {carpeta}')
    if error:raise RuntimeError(error)


if __name__=='__main__':main()
