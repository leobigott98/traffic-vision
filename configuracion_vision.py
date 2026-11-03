"""Configuración estricta y rutas relativas al JSON; nunca guarda credenciales."""
import json
import math
import os
from pathlib import Path
from contrato_vision import ACCESOS


def cargar(ruta, comprobar_archivos=True):
    ruta = Path(ruta).resolve()
    cfg = json.loads(ruta.read_text(encoding='utf-8'))
    if cfg.get('version') != 1:
        raise ValueError('La configuración debe tener version=1.')
    for key, default in [('caducidad_s', 3.), ('publicar_cada_s', 1.), ('salto_max_s', 1.)]:
        value = cfg.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} debe ser positivo y finito.')
    if cfg['publicar_cada_s'] > cfg['caducidad_s']:
        raise ValueError('publicar_cada_s debe ser <= caducidad_s.')
    if not isinstance(cfg.get('fuentes'), list) or not cfg['fuentes']:
        raise ValueError('Configura al menos una fuente.')
    ids, vivos = set(), set()
    for src in cfg['fuentes']:
        sid = src.get('id')
        if not isinstance(sid, str) or not sid.replace('_', '').replace('-', '').isalnum() or sid in ids:
            raise ValueError('Cada fuente necesita un id alfanumérico único.')
        ids.add(sid)
        if src.get('tipo') not in ('rtsp', 'archivo'):
            raise ValueError(f'{sid}: tipo debe ser rtsp o archivo.')
        if src['tipo'] == 'rtsp':
            if not isinstance(src.get('url_env'), str) or not src['url_env'] or 'url' in src:
                raise ValueError(f'{sid}: usa url_env para las credenciales, no url.')
        elif not isinstance(src.get('ruta'), str) or not src['ruta']:
            raise ValueError(f'{sid}: falta ruta del archivo.')
        for key in ('ruta', 'zonas', 'referencia', 'fondo'):
            if src.get(key):
                src[key] = str((ruta.parent / src[key]).resolve())
                if comprobar_archivos and not Path(src[key]).is_file():
                    raise ValueError(f'{sid}: no existe el archivo de {key}.')
        if not isinstance(src.get('camara_fija'), bool):
            raise ValueError(f'{sid}: declara camara_fija=true o false.')
        if not src['camara_fija'] and not src.get('referencia'):
            raise ValueError(f'{sid}: compensación requiere referencia guardada con las zonas.')
        mapeo = src.setdefault('mapeo_accesos', {})
        if not isinstance(mapeo, dict) or any(not isinstance(k, str) or v not in ACCESOS for k,v in mapeo.items()):
            raise ValueError(f'{sid}: mapeo_accesos no válido.')
        if len(set(mapeo.values())) != len(mapeo):
            raise ValueError(f'{sid}: usa un polígono por acceso, no sumes zonas superpuestas.')
        if mapeo and not src.get('zonas'):
            raise ValueError(f'{sid}: el mapeo requiere zonas.')
        if src['tipo'] == 'rtsp':
            if vivos.intersection(mapeo.values()):
                raise ValueError('Un acceso en vivo no puede tener dos fuentes (doble conteo).')
            if set(mapeo.values()).intersection(('av1','av2')):
                raise ValueError('Av. 1 y Av. 2 no tienen cámara en este despliegue.')
            vivos.update(mapeo.values())
    return cfg


def origen(src):
    if src['tipo'] == 'archivo':
        return src['ruta']
    valor = os.environ.get(src['url_env'], '')
    if not valor.startswith(('rtsp://', 'rtsps://')):
        raise ValueError(f"{src['id']}: define {src['url_env']} con la URL RTSP.")
    return valor
