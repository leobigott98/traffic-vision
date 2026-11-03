"""Contrato de observación parcial. Sin dependencias de visión ni valores inventados."""
from datetime import datetime, timezone
import json
from pathlib import Path

ACCESOS = ('boyera_trinidad', 'trinidad_boyera', 'cigarral', 'av1', 'av2')
METRICAS = ('observados', 'cola_visible_proxy', 'detenidos', 'lentos',
            'indeterminados', 'cobertura_clasificacion', 'fraccion_visible',
            'max_episodio_reducido_s')


def utc():
    return datetime.now(timezone.utc).isoformat()


def guardar_atomico(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    temp.replace(path)


def desconocido(motivo, fuente=None):
    return dict(estado=motivo, fuente_id=fuente, observado=False,
                **{k: None for k in METRICAS}, velocidad_media_m_s=None,
                longitud_cola_m=None, espera_total_s=None, flujo_veh_h=None)


def construir_estado(config, ultimos, estados, ahora, sesion):
    """El reloj monotónico solo se usa dentro del proceso; se exporta edad + UTC.

    Una reproducción nunca entra en el campo 'accesos' de mediciones en vivo.
    El consumidor debe comprobar además generado_utc por si este proceso murió.
    """
    accesos = {k: desconocido('sin_camara') for k in ACCESOS}
    fuentes = {}
    for src in config['fuentes']:
        sid = src['id']
        ultimo = ultimos.get(sid)
        estado = estados.get(sid, 'iniciando')
        edad = max(0., ahora - ultimo['recibido_monotonic']) if ultimo else None
        vigente = ultimo is not None and estado == 'activo' and edad <= config['caducidad_s']
        motivo = estado if estado != 'activo' else ('vigente' if vigente else 'caducado')
        mapped = {}
        for zona, acceso in src['mapeo_accesos'].items():
            item = desconocido(motivo if not vigente else 'sin_zona', sid)
            cuenta = ultimo.get('cola', {}).get(zona) if ultimo else None
            if vigente and cuenta is not None and cuenta['observados'] is not None:
                item.update({k: cuenta[k] for k in METRICAS})
                item.update(estado=cuenta['estado'], observado=True)
            elif vigente and cuenta is not None:
                item['estado'] = cuenta['estado']
            mapped[acceso] = item
            if src['tipo'] == 'rtsp':
                accesos[acceso] = item
        fuentes[sid] = dict(tipo=src['tipo'], dominio='en_vivo' if src['tipo']=='rtsp' else 'reproduccion',
            estado=motivo, vigente=vigente, edad_recepcion_s=edad,
            frame=ultimo.get('frame') if ultimo else None,
            tiempo_fuente_s=ultimo.get('tiempo_fuente_s') if ultimo else None,
            recibido_utc=ultimo.get('recibido_utc') if ultimo else None,
            tiempo_base='recepcion_monotonic' if src['tipo']=='rtsp' else 'pts_archivo',
            epoca=ultimo.get('epoca') if ultimo else None,
            inferencia_ms=ultimo.get('inferencia_ms') if ultimo else None,
            descartados=ultimo.get('descartados') if ultimo else None,
            zonas=ultimo.get('cola', {}) if vigente else {}, accesos=mapped)
    return dict(schema_version='traffic-vision/1', sesion_id=sesion, generado_utc=utc(),
        caducidad_s=config['caducidad_s'], fuentes=fuentes, accesos=accesos,
        mascara_observacion={k: v['observado'] for k,v in accesos.items()},
        listo_para_control=False,
        limitaciones=['cola_visible_proxy_no_equivale_a_halting_SUMO',
                      'sin_calibracion_metrica', 'sin_estado_semaforico',
                      'politica_RL_y_normalizacion_pendientes_de_integracion'])
