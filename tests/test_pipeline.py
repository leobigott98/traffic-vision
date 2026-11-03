"""Integración con ByteTrack real y detecciones controladas: requiere ultralytics."""
import json
import time
from types import SimpleNamespace
import numpy as np
import pytest
pytest.importorskip('ultralytics')
from ultralytics.engine.results import Boxes
from ejecutar_vision import Analizador
from estado_cola import ParametrosCola
from fuentes_video import Fotograma
from seguir_video import leer_tracker
from pathlib import Path


class Detector:
    names={2:'car'}
    def predict(self, frame, **kwargs):
        boxes=Boxes(np.array([[20,20,40,60,.9,2]],dtype=np.float32),frame.shape[:2])
        return [SimpleNamespace(boxes=boxes,names=self.names)]


def source(tmp_path, sid):
    zone=tmp_path/'zones.json'
    zone.write_text(json.dumps(dict(version=2,alto_ancho=[100,100],zonas=[dict(
        id='z',nombre='test',acceso='test',puntos_xy=[[0,0],[99,0],[99,99],[0,99]])])))
    return dict(id=sid,tipo='rtsp',camara_fija=True,zonas=str(zone),mapeo_accesos={'z':'boyera_trinidad'})


def packet(i, epoch=0):
    return Fotograma(np.zeros((100,100,3),np.uint8),i,i/10,time.monotonic(),'utc',epoch)


def test_real_tracker_independent_and_reset_clears_queue(tmp_path):
    cfg=leer_tracker(Path('bytetrack_tesis.yaml'),.1)
    a=Analizador(source(tmp_path,'a'),cfg,ParametrosCola())
    b=Analizador(source(tmp_path,'b'),cfg,ParametrosCola())
    model=Detector()
    for i in range(45):
        da,_=a.procesar(packet(i),model,{},1.,False)
        db,_=b.procesar(packet(i),model,{},1.,False)
    assert da['cola']['z']['cola_visible_proxy']==1
    assert db['cola']['z']['cola_visible_proxy']==1
    # Reset de A no cambia historial ni IDs públicos de B.
    id_b=dict(b.identidades.mapa)
    da,_=a.procesar(packet(45,1),model,{},1.,False)
    db,_=b.procesar(packet(45),model,{},1.,False)
    assert da['cola']['z']['cola_visible_proxy']==0
    assert db['cola']['z']['cola_visible_proxy']==1
    assert b.identidades.mapa==id_b
    assert da['epoca']==1
    assert len(a.historial.puntos)==1
    assert len(b.historial.puntos)==1


def test_large_gap_resets_and_preview_is_drawn(tmp_path):
    cfg=leer_tracker(Path('bytetrack_tesis.yaml'),.1)
    a=Analizador(source(tmp_path,'a'),cfg,ParametrosCola())
    for i in range(45):a.procesar(packet(i),Detector(),{},1.,False)
    data,view=a.procesar(packet(70),Detector(),{},1.,True)
    assert data['cola']['z']['cola_visible_proxy']==0
    assert view is not None and view.shape[2]==3
    assert data['epoca']==1
