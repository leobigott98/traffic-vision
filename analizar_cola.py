"""Reprocesa observaciones compensadas sin ejecutar YOLO ni volver a leer video."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
from time import perf_counter
from estado_cola import EstimadorCola, agregar_argumentos, parametros_argumentos


def escribir(ruta, datos):
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observaciones', type=Path, required=True)
    p.add_argument('--salida', type=Path, default=Path(__file__).parent/'resultados_cola')
    agregar_argumentos(p)
    a = p.parse_args()
    if not a.observaciones.is_file():p.error('No existe el JSONL de observaciones.')
    if not a.zonas_cola:p.error('Indica --zonas-cola con los IDs de las zonas de aproximación.')
    try:estimador = EstimadorCola(a.zonas_cola, parametros_argumentos(a))
    except ValueError as exc:p.error(str(exc))
    carpeta = a.salida.resolve()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    carpeta.mkdir(parents=True)
    campos = ['frame','tiempo_s','zona_id','estado','observados','cola_visible_proxy','detenidos',
              'lentos','indeterminados','cobertura_clasificacion','fraccion_visible','max_episodio_reducido_s']
    campos_o = ['frame','tiempo_s','track_id','zona_id','estado','velocidad_relativa','velocidad_px_s',
                'en_cola_proxy','episodio_reducido_s','detenido_observado_s']
    inicio = perf_counter();n = 0;ultimo = None;error = None
    maximos = {z: 0 for z in estimador.zonas};sin_dato = {z: 0 for z in estimador.zonas}
    try:
        with a.observaciones.open(encoding='utf-8-sig') as src, (carpeta/'cola.csv').open('w', newline='', encoding='utf-8') as fc, (carpeta/'movimiento.csv').open('w', newline='', encoding='utf-8') as fo, (carpeta/'cola.jsonl').open('w', encoding='utf-8') as fj:
            wc = csv.DictWriter(fc, fieldnames=campos);wc.writeheader()
            wo = csv.DictWriter(fo, fieldnames=campos_o);wo.writeheader()
            for linea in src:
                if not linea.strip():continue
                datos = json.loads(linea)
                frame, t = datos['frame'], datos['tiempo_s']
                obs = datos['observaciones']
                cola = estimador.actualizar(frame, t, obs, datos['conteos_zona'], datos['camara']['valida'])
                for zona, cuenta in cola.items():
                    wc.writerow(dict(frame=frame, tiempo_s=t, zona_id=zona, **cuenta))
                    if cuenta['cola_visible_proxy'] is None:sin_dato[zona] += 1
                    else:maximos[zona] = max(maximos[zona], cuenta['cola_visible_proxy'])
                for o in obs:
                    wo.writerow(dict(frame=frame, tiempo_s=t, track_id=o['track_id'],
                                     zona_id=o['asignacion'].get('zona_id'), **o['movimiento']))
                ultimo = dict(frame=frame, tiempo_s=t, cola=cola,
                    vehiculos=[dict(track_id=o['track_id'], xyxy_actual=o['xyxy_actual'],
                                    zona_id=o['asignacion'].get('zona_id'), **o['movimiento']) for o in obs])
                fj.write(json.dumps(ultimo, ensure_ascii=False, allow_nan=False)+'\n')
                n += 1
        if not n:raise ValueError('El archivo no contiene fotogramas.')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        resumen = dict(fuente=str(a.observaciones.resolve()), fotogramas=n, error=error,
            duracion_reanalisis_s=perf_counter()-inicio, zonas=sorted(estimador.zonas),
            parametros=asdict(estimador.p), max_cola_visible_proxy=maximos,
            fotogramas_sin_dato=sin_dato, ultimo=ultimo['cola'] if ultimo else None,
            alcance='Proxy: vehículos observados con ID y movimiento reducido persistente en zonas seleccionadas. No causalidad semafórica, cola oculta, metros ni espera total.',
            tiempo='Segundos del archivo fuente; huecos, cambios de zona y pérdidas de ID reinician evidencia.',
            advertencia='Umbrales iniciales sin calibración. Indeterminados no equivalen a vehículos en movimiento. No sumar colas entre fotogramas.')
        escribir(carpeta/'resumen_cola.json', resumen)
        if ultimo:escribir(carpeta/'ultimo_cola.json', ultimo)
    print(f'Fotogramas: {n}; resultados: {carpeta}')
    print('Máximos de proxy de cola visible:', maximos)
    if error:raise RuntimeError(error)


if __name__ == '__main__':main()
