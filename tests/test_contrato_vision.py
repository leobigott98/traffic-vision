import json
from copy import deepcopy
import pytest
from contrato_vision import construir_estado, guardar_atomico
from configuracion_vision import cargar


def config():
    return dict(version=1,caducidad_s=3,publicar_cada_s=1,salto_max_s=1,fuentes=[
        dict(id='cam',tipo='rtsp',url_env='CAMARA_RTSP',camara_fija=True,zonas='zones.json',
             mapeo_accesos={'zona_01':'boyera_trinidad'}),
        dict(id='file',tipo='archivo',ruta='video.mp4',camara_fija=True,zonas='zones.json',
             mapeo_accesos={'zona_01':'boyera_trinidad'})])


def muestra(n=0):
    return dict(recibido_monotonic=10.,cola={'zona_01':dict(estado='visible',observados=n,
        cola_visible_proxy=n,detenidos=n,lentos=0,indeterminados=0,cobertura_clasificacion=1.,
        fraccion_visible=1.,max_episodio_reducido_s=2.)})


def test_replay_never_overwrites_live():
    r=construir_estado(config(),{'cam':muestra(2),'file':muestra(9)},dict(cam='activo',file='activo'),11,'s')
    assert r['accesos']['boyera_trinidad']['observados']==2
    assert r['fuentes']['file']['accesos']['boyera_trinidad']['observados']==9
    assert r['accesos']['av1']['observados'] is None
    assert r['mascara_observacion']['av2'] is False
    assert r['listo_para_control'] is False


def test_zero_is_observed_but_missing_is_null():
    r=construir_estado(config(),{'cam':muestra()},dict(cam='activo'),11,'s')
    assert r['accesos']['boyera_trinidad']['observados']==0
    assert r['accesos']['boyera_trinidad']['observado'] is True
    assert r['accesos']['cigarral']['observados'] is None


@pytest.mark.parametrize('status,now',[('activo',14),('desconectado',11),('detenido',11),('finalizado',11)])
def test_stale_or_disconnected_is_not_zero(status,now):
    r=construir_estado(config(),{'cam':muestra(5)},dict(cam=status),now,'s')
    assert r['accesos']['boyera_trinidad']['cola_visible_proxy'] is None
    assert not r['mascara_observacion']['boyera_trinidad']
    assert r['fuentes']['cam']['zonas']=={}


def test_invalid_alignment_propagates_null():
    m=muestra()
    m['cola']['zona_01'].update(estado='alineacion_invalida',observados=None)
    r=construir_estado(config(),{'cam':m},dict(cam='activo'),11,'s')
    assert r['accesos']['boyera_trinidad']['estado']=='alineacion_invalida'
    assert not r['mascara_observacion']['boyera_trinidad']


def test_no_camera_does_not_inherit_replay():
    r=construir_estado(config(),{'file':muestra(9)},dict(file='activo'),11,'s')
    assert r['accesos']['boyera_trinidad']['observados'] is None


def test_atomic_json_replaced_and_finite(tmp_path):
    p=tmp_path/'state.json'
    guardar_atomico(p,{'value':1});guardar_atomico(p,{'value':2})
    assert json.loads(p.read_text())=={'value':2}
    with pytest.raises(ValueError):
        guardar_atomico(p,{'value':float('nan')})
    assert json.loads(p.read_text())=={'value':2}


@pytest.mark.parametrize('change', ['duplicate','av1','same_zone','nan','bad_type'])
def test_config_rejects_invalid_mapping(tmp_path,change):
    cfg=config()
    if change=='duplicate':
        other=deepcopy(cfg['fuentes'][0]);other['id']='cam2';cfg['fuentes'].append(other)
    elif change=='av1':cfg['fuentes'][0]['mapeo_accesos']['zona_01']='av1'
    elif change=='same_zone':cfg['fuentes'][0]['mapeo_accesos']['zona_02']='boyera_trinidad'
    elif change=='nan':cfg['caducidad_s']=float('nan')
    else:cfg['fuentes'][0]['camara_fija']='true'
    p=tmp_path/'cfg.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):cargar(p,comprobar_archivos=False)
