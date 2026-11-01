"""Paso 3: conteo instantáneo por zona en video o webcam, sin seguimiento.

Requiere contar_multizonas.py en la MISMA carpeta y su entorno de dependencias.
  python contar_video.py --camara 0 --zonas zonas_webcam.json --max-frames 300
  python contar_video.py --video calle.mp4 --zonas zonas_calle.json

Q/Esc o cerrar ventana finaliza. Se guardan conteos.csv, resumen.json y
ultimo_frame.jpg. No se guarda el video completo en esta etapa.
--nms-agnostica activa NMS entre clases (sin la opción se conserva el filtro anterior).
--capturas-s 5 10 15 20 25 guarda originales, anotaciones y detecciones en esos instantes.
No sumar conteos entre fotogramas para calcular vehículos únicos o flujo.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
from time import perf_counter

from contar_multizonas import (
    CLASES, VEHICULOS, leer_config, editar_zonas, preparar_zonas,
    advertir_solapes, asignar, ascii_text,
)


def tiempo_video(indice, pts_ms, fps, anterior):
    """Prefiere tiempo del archivo; si falla, estima usando FPS nominales.

    La estimación por FPS supone cadencia constante. Se marca en cada fila.
    No es una medición exacta del tiempo de exposición de la cámara.
    """
    pts = pts_ms / 1000
    if math.isfinite(pts) and pts >= 0 and (indice == 0 or pts > anterior):
        return pts, "archivo_opencv"
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("El archivo no ofrece timestamps ni FPS utilizables.")
    estimado = indice / fps
    if indice > 0 and estimado <= anterior:
        estimado = anterior + 1 / fps
    return estimado, "estimado_fps"


def contar_detecciones(detecciones, zonas, reglas):
    conteos = {z['id']: dict.fromkeys(CLASES, 0) for z, _, _ in zonas}
    estados = {s: dict.fromkeys(CLASES, 0) for s in ('asignado', 'ambiguo', 'fuera')}
    for d in detecciones:
        a = asignar(d['xyxy'], zonas, **reglas)
        d['asignacion'] = a
        estados[a['estado']][d['clase']] += 1
        if a['zona_id'] is not None:
            conteos[a['zona_id']][d['clase']] += 1
    return conteos, estados


def crear_vista(frame, detecciones, zonas, conteos, estados, indice, t, analisis_ms, cv2, np):
    """Reduce solo la visualización; la geometría sigue en píxeles originales."""
    alto, ancho = frame.shape[:2]
    escala = min(900 / ancho, 600 / alto, 1)
    w, h = max(1, round(ancho * escala)), max(1, round(alto * escala))
    vista = cv2.resize(frame, (w, h))
    paleta = [(0,200,0),(255,170,0),(200,80,220),(0,180,240),(200,200,0),(150,100,255)]
    colores = {z['id']: paleta[i % len(paleta)] for i,(z,_,_) in enumerate(zonas)}
    def punto(x,y):
        return (round(x*w/ancho),round(y*h/alto))
    for d in detecciones:
        a = d['asignacion']
        color = colores[a['zona_id']] if a['zona_id'] else ((0,140,255) if a['estado']=='ambiguo' else (140,140,140))
        x1,y1,x2,y2 = d['xyxy']
        cv2.rectangle(vista,punto(x1,y1),punto(x2,y2),color,2)
        label = f"{d['clase']} {a['zona_id'] or a['estado']}"
        x,y = punto(x1,y1)
        cv2.putText(vista,label,(x,max(15,y-5)),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
    for z,_,_ in zonas:
        vertices=np.array([punto(x,y) for x,y in z['puntos_xy']],np.int32)
        cv2.polylines(vista,[vertices],True,colores[z['id']],2)
    accesos={z['acceso'] for z,_,_ in zonas}
    panel_h=max(h,240+42*len(zonas)+25*len(accesos))
    salida=np.full((panel_h,w+390,3),28,np.uint8)
    salida[:h,:w]=vista
    def texto(s,y,color=(230,230,230)):
        cv2.putText(salida,ascii_text(s)[:47],(w+12,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
    texto(f"Fotograma: {indice} | t={t:.2f}s",25)
    texto(f"Analisis: {analisis_ms:.1f} ms",48)
    texto("Conteo actual (no acumulado)",75)
    y=105
    for z,_,_ in zonas:
        c=conteos[z['id']]
        texto(f"{z['id']}: {z['nombre']}",y,colores[z['id']])
        texto(f"  Vehiculos={sum(c[n] for n in VEHICULOS)} Personas={c['person']}",y+19)
        y+=42
    texto('Por acceso:',y);y+=23
    for acceso in sorted(accesos):
        cs=[conteos[z['id']] for z,_,_ in zonas if z['acceso']==acceso]
        v=sum(c[n] for c in cs for n in VEHICULOS)
        personas=sum(c['person'] for c in cs)
        texto(f"{acceso}: V={v} P={personas}",y);y+=25
    texto(f"Ambiguos: {sum(estados['ambiguo'].values())} | Fuera: {sum(estados['fuera'].values())}",y+8)
    texto('Q / Esc: finalizar',y+35)
    return salida


def guardar_captura(carpeta, frame, detecciones, conteos, estados, indice, t,
                    tipo_tiempo, objetivos, contexto, cv2, np):
    """Guarda el primer frame cuyo tiempo alcanza cada objetivo, sin otra inferencia.

    Varios objetivos alcanzados por un mismo frame comparten archivos. La imagen
    original mantiene la resolución y los píxeles decodificados (PNG sin pérdida).
    Los números de la anotación son índices locales, NO IDs de seguimiento.
    """
    destino = carpeta / 'capturas' / f'frame_{indice:06d}'
    destino.mkdir(parents=True, exist_ok=False)
    anotada = frame.copy()
    paleta = [(0,200,0),(255,170,0),(200,80,220),(0,180,240),(200,200,0),(150,100,255)]
    zonas = contexto['configuracion_zonas']['zonas']
    colores = {z['id']: paleta[i % len(paleta)] for i,z in enumerate(zonas)}
    for z in zonas:
        cv2.polylines(anotada, [np.rint(z['puntos_xy']).astype(np.int32)],
                      True, colores[z['id']], 1)
    numeradas = []
    for numero, d in enumerate(detecciones, 1):
        numeradas.append(dict(d, indice_deteccion=numero))
        a = d['asignacion']
        color = colores.get(a['zona_id'], (0,140,255) if a['estado']=='ambiguo' else (140,140,140))
        x1,y1,x2,y2 = map(round, d['xyxy'])
        cv2.rectangle(anotada, (x1,y1), (x2,y2), color, 1)
        etiqueta = f"{numero} {d['clase']} {d['confianza']:.2f}"
        cv2.putText(anotada, etiqueta, (x1,max(12,y1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    for nombre, imagen in [('original.png',frame),('anotada.jpg',anotada)]:
        ok, datos = cv2.imencode(Path(nombre).suffix, imagen)
        if not ok:
            raise RuntimeError(f'No se pudo codificar {nombre} de frame {indice}.')
        datos.tofile(str(destino / nombre))
    informe = dict(contexto, frame=indice, tiempo_s=t, tipo_tiempo=tipo_tiempo,
                   objetivos_s=objetivos, alto_ancho_original=list(frame.shape[:2]),
                   conteos_zona=conteos, estados_por_clase=estados,
                   detecciones=numeradas,
                   criterio_captura='Primer fotograma con tiempo_s >= objetivo_s.',
                   nota='indice_deteccion solo identifica una caja en este fotograma; no es un ID de seguimiento.')
    (destino / 'detecciones.json').write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'frame':indice, 'tiempo_s':t, 'objetivos_s':objetivos,
            'carpeta':destino.relative_to(carpeta).as_posix()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    fuente=p.add_mutually_exclusive_group(required=True)
    fuente.add_argument('--video',type=Path)
    fuente.add_argument('--camara',type=int)
    p.add_argument('--zonas',type=Path,required=True)
    p.add_argument('--editar',action='store_true')
    p.add_argument('--max-frames',type=int,default=0,help='0: hasta fin del archivo o interrupción')
    p.add_argument('--sin-preview',action='store_true')
    p.add_argument('--modelo',default='yolo11n.pt')
    p.add_argument('--device',default='0')
    p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--nms-agnostica',action='store_true',
                   help='Suprime cajas muy superpuestas también entre clases distintas')
    p.add_argument('--capturas-s',type=float,nargs='+',default=[],metavar='SEGUNDOS',
                   help='Solo video: guarda primer frame en o después de cada tiempo solicitado')
    p.add_argument('--tolerancia',type=float,default=5)
    p.add_argument('--snap',type=float,default=12)
    p.add_argument('--cobertura-minima',type=float,default=.5)
    p.add_argument('--margen',type=float,default=.1)
    p.add_argument('--fraccion-inferior',type=float,default=1)
    args=p.parse_args()
    if args.video and not args.video.is_file():p.error('No existe el video.')
    if args.zonas.suffix.lower()!='.json':p.error('Las zonas deben guardarse en un .json.')
    if args.max_frames<0 or args.imgsz<=0 or (args.camara is not None and args.camara<0):p.error('Revisa max-frames, imgsz y camara.')
    numeros=(args.conf,args.tolerancia,args.snap,args.cobertura_minima,args.margen,args.fraccion_inferior)
    if not all(math.isfinite(x) for x in numeros):p.error('Parámetros numéricos no finitos.')
    if not (0<=args.conf<=1 and args.tolerancia>=0 and args.snap>=3 and 0<args.cobertura_minima<=1 and 0<=args.margen<=1 and 0<args.fraccion_inferior<=1):p.error('Parámetros fuera de rango.')
    if any(not math.isfinite(t) or t<0 for t in args.capturas_s):p.error('capturas-s requiere segundos finitos y no negativos.')
    if args.capturas_s and args.camara is not None:p.error('capturas-s solo está disponible con --video.')
    pendientes=sorted(set(args.capturas_s))
    capturas_guardadas=[]
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    origen=str(args.video.resolve()) if args.video else args.camara
    # Solo se usa este primer fotograma como referencia del editor. Se libera
    # la cámara mientras editamos para no acumular capturas antiguas.
    captura=cv2.VideoCapture(origen)
    try:
        ok,primero=captura.read()
        if not captura.isOpened() or not ok:raise RuntimeError('No se pudo abrir o leer la fuente de video.')
    finally:
        captura.release()
    alto,ancho=primero.shape[:2]
    config=leer_config(args.zonas,ancho,alto) if args.zonas.exists() else {'version':2,'alto_ancho':[alto,ancho],'zonas':[]}
    if args.editar or not args.zonas.exists():
        nuevas=editar_zonas(primero,config['zonas'],args.snap,cv2,np)
        if nuevas is None:
            print('Edición cancelada; no se procesó video.');return
        config['zonas']=nuevas
        config['fuente_referencia']=origen
        args.zonas.parent.mkdir(parents=True,exist_ok=True)
        args.zonas.write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    zonas=preparar_zonas(config,ancho,alto,args.tolerancia)
    advertir_solapes(zonas)
    print('Mantén el mismo encuadre. El CSV registra conteos por fotograma, no vehículos únicos.')
    model=YOLO(args.modelo)
    if set(CLASES)-set(model.names.values()):raise ValueError('El modelo no contiene todas las clases requeridas.')
    ids=[i for i,nombre in model.names.items() if nombre in CLASES]
    pred_args=dict(device=args.device,conf=args.conf,imgsz=args.imgsz,iou=.7,
                   agnostic_nms=args.nms_agnostica,classes=ids,save=False,verbose=False)
    # Calentamiento real con una imagen; no se registra como fotograma de prueba.
    model.predict(primero,**pred_args)
    reglas=dict(minimo=args.cobertura_minima,margen=args.margen,fraccion_inferior=args.fraccion_inferior,tolerancia=args.tolerancia)
    contexto_capturas={'fuente':origen,'configuracion_zonas':config,'reglas':reglas,
                       'modelo':args.modelo,'conf':args.conf,'imgsz':args.imgsz,
                       'device':args.device,'iou':.7,'agnostic_nms':args.nms_agnostica,
                       'versiones':{'python':platform.python_version(),'ultralytics':ultralytics.__version__,
                                    'torch':torch.__version__,'opencv':cv2.__version__}}
    ahora=datetime.now(timezone.utc)
    carpeta=Path(__file__).resolve().parent/'resultados_video'/ahora.strftime('%Y%m%dT%H%M%S_%fZ')
    carpeta.mkdir(parents=True,exist_ok=False)
    columnas=['frame','tiempo_s','tipo_tiempo','lectura_ms','analisis_ms']
    columnas += [f"{z['id']}_{c}" for z,_,_ in zonas for c in CLASES]
    columnas += [f'{s}_{c}' for s in ('ambiguo','fuera') for c in CLASES]
    captura=cv2.VideoCapture(origen)
    fps_fuente=float(captura.get(cv2.CAP_PROP_FPS))
    total_nominal=float(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps_fuente):fps_fuente=0.0
    if not math.isfinite(total_nominal):total_nominal=0.0
    inicio=perf_counter();n=0;anterior=-1.0;origen_recepcion=None
    tiempos=[];tipos=set();ultima=None;motivo='error';error=None
    ventana='Conteo por zonas | Q o Esc para terminar'
    try:
        if not captura.isOpened():raise RuntimeError('No se pudo reabrir la fuente.')
        with (carpeta/'conteos.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=columnas);writer.writeheader()
            while True:
                leer_inicio=perf_counter()
                ok,frame=captura.read()
                recepcion=perf_counter()
                if not ok:
                    motivo='fallo_camara' if args.camara is not None else 'fin_lectura_archivo'
                    if args.video and total_nominal>0 and n<int(total_nominal)-1:motivo='lectura_incompleta'
                    break
                if frame.shape[:2]!=(alto,ancho):raise ValueError('Cambió la resolución respecto a las zonas; procesamiento detenido.')
                if args.video:
                    t,tipo=tiempo_video(n,captura.get(cv2.CAP_PROP_POS_MSEC),fps_fuente,anterior)
                else:
                    if origen_recepcion is None:origen_recepcion=recepcion
                    t,tipo=recepcion-origen_recepcion,'recepcion_webcam'
                anterior=t;tipos.add(tipo)
                comenzar=perf_counter()
                result=model.predict(frame,**pred_args)[0]
                ds=[{'clase':result.names[int(b.cls.item())],'confianza':float(b.conf.item()),'xyxy':b.xyxy[0].tolist()} for b in result.boxes.cpu()]
                conteos,estados=contar_detecciones(ds,zonas,reglas)
                ms=(perf_counter()-comenzar)*1000;tiempos.append(ms)
                fila={'frame':n,'tiempo_s':t,'tipo_tiempo':tipo,'lectura_ms':(recepcion-leer_inicio)*1000,'analisis_ms':ms}
                fila.update({f"{z['id']}_{c}":conteos[z['id']][c] for z,_,_ in zonas for c in CLASES})
                fila.update({f'{s}_{c}':estados[s][c] for s in ('ambiguo','fuera') for c in CLASES})
                writer.writerow(fila)
                ultima=(frame,ds,conteos,estados,n,t,ms)
                n+=1
                objetivos=[objetivo for objetivo in pendientes if objetivo<=t]
                if objetivos:
                    guardada=guardar_captura(carpeta,frame,ds,conteos,estados,n-1,t,tipo,
                                            objetivos,contexto_capturas,cv2,np)
                    capturas_guardadas.append(guardada)
                    pendientes=pendientes[len(objetivos):]
                    print(f'Captura: frame {n-1}, t={t:.3f}s, objetivos={objetivos}')
                if not args.sin_preview:
                    vista=crear_vista(frame,ds,zonas,conteos,estados,n-1,t,ms,cv2,np)
                    cv2.imshow(ventana,vista)
                    tecla=cv2.waitKey(1)&0xFF
                    if tecla in (27,ord('q'),ord('Q')) or cv2.getWindowProperty(ventana,cv2.WND_PROP_VISIBLE)<1:
                        motivo='usuario';break
                if n%30==0:
                    f.flush()
                    print(f'Fotogramas: {n}; tiempo fuente: {t:.2f}s; análisis: {ms:.1f} ms')
                if args.max_frames and n>=args.max_frames:
                    motivo='limite_fotogramas';break
    except KeyboardInterrupt:
        motivo='interrupcion_teclado'
    except Exception as exc:
        motivo='error';error=f'{type(exc).__name__}: {exc}'
    finally:
        duracion=perf_counter()-inicio
        captura.release()
        if not args.sin_preview:cv2.destroyAllWindows()
        resumen={'fecha_utc':ahora.isoformat(),'fuente':origen,'motivo_fin':motivo,'error':error,
                 'fotogramas_procesados':n,'fps_fuente_reportados':fps_fuente,'frames_fuente_reportados':total_nominal,
                 'duracion_bucle_s':duracion,'fps_bucle_observados':n/duracion if duracion else 0,
                 'analisis_ms_media':float(np.mean(tiempos)) if tiempos else None,
                 'analisis_ms_p95':float(np.percentile(tiempos,95)) if tiempos else None,
                 'ultimo_tiempo_fuente_s':anterior if n else None,'tipos_tiempo':sorted(tipos),
                 'configuracion_zonas':config,'reglas':reglas,
                 'modelo':args.modelo,'conf':args.conf,'imgsz':args.imgsz,'device':args.device,
                 'iou':.7,'agnostic_nms':args.nms_agnostica,
                 'capturas_s_solicitadas':sorted(set(args.capturas_s)),
                 'capturas_guardadas':capturas_guardadas,'capturas_s_pendientes':pendientes,
                 'versiones':{'python':platform.python_version(),'ultralytics':ultralytics.__version__,'torch':torch.__version__,'opencv':cv2.__version__},
                 'metricas':'Analisis: predict + traslado de cajas a CPU + asignación. Bucle: incluye lectura, CSV, preview y escritura de capturas solicitadas; excluye editor, carga y calentamiento.',
                 'alcance':'Conteos por imagen; sin IDs, flujo ni cola. Tiempo webcam: recepción, no exposición; puede haber buffering o frames perdidos en el backend.',
                 'advertencia_tiempo':'estimado_fps supone cadencia constante; revisar archivos con FPS variable.'}
        (carpeta/'resumen.json').write_text(json.dumps(resumen,indent=2,ensure_ascii=False),encoding='utf-8')
        if ultima is not None:
            frame,ds,c,e,i,t,ms=ultima
            vista=crear_vista(frame,ds,zonas,c,e,i,t,ms,cv2,np)
            ok,jpg=cv2.imencode('.jpg',vista)
            if ok:jpg.tofile(str(carpeta/'ultimo_frame.jpg'))
    print(f'\nFinalizado: {motivo}. Fotogramas: {n}. Resultados: {carpeta}')
    if pendientes:print(f'Tiempos de captura no alcanzados: {pendientes}')
    if error:raise RuntimeError(error)


if __name__=='__main__':
    main()
