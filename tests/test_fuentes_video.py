import time
import cv2
import numpy as np
from fuentes_video import Fotograma,Lector


def test_live_keeps_latest_only():
    l=Lector(dict(id='a',tipo='rtsp'),'unused')
    for i in range(200):
        l.publicar(Fotograma(None,i,i,i,'utc',0))
    assert l.cola.qsize()==1
    assert l.tomar().indice==199
    assert l.descartados==199


def test_real_file_all_frames_in_order(tmp_path):
    p=tmp_path/'test.avi'
    writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
    assert writer.isOpened()
    for n in range(12):writer.write(np.full((48,64,3),n*10,np.uint8))
    writer.release()
    l=Lector(dict(id='file',tipo='archivo',ritmo_real=False),str(p));l.start()
    frames=[];deadline=time.monotonic()+5
    while (l.is_alive() or not l.cola.empty()) and time.monotonic()<deadline:
        frame=l.tomar()
        if frame:frames.append(frame)
        else:time.sleep(.005)
    l.detener.set();l.join(1)
    assert [x.indice for x in frames]==list(range(12))
    assert all(b.tiempo_s>a.tiempo_s for a,b in zip(frames,frames[1:]))
    assert l.estado=='finalizado'
    assert l.descartados==0


def test_reconnect_increments_epoch_and_releases():
    import threading
    opened=[]
    class Capture:
        def __init__(self):self.calls=0;self.released=False;opened.append(self)
        def isOpened(self):return True
        def get(self,key):return 15.
        def read(self):
            self.calls+=1
            if len(opened)==1:return False,None
            time.sleep(.01)
            return True,np.zeros((4,4,3),np.uint8)
        def release(self):self.released=True
    l=Lector(dict(id='cam',tipo='rtsp'),'secret',lambda _:Capture());l.start()
    deadline=time.monotonic()+4;frame=None
    while time.monotonic()<deadline and frame is None:
        frame=l.tomar();time.sleep(.01)
    l.detener.set();l.join(1)
    assert frame is not None and frame.epoca==1
    assert all(c.released for c in opened)
    assert l.estado=='detenido'
