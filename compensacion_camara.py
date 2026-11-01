"""Registro a referencia fija con ORB + RANSAC (similitud, cuatro parámetros).

M transforma puntos del frame ACTUAL a píxeles del frame de REFERENCIA.
No deforma imágenes ni detecta vehículos. Los controles de calidad son heurísticos.
"""
import math
import cv2
import numpy as np


def transformar(puntos, matriz):
    puntos = np.asarray(puntos, np.float64).reshape(-1, 2)
    return puntos @ np.asarray(matriz, np.float64)[:, :2].T + matriz[:, 2]


def esquinas_caja(caja, fraccion_inferior=1.):
    x1,y1,x2,y2 = caja
    y1 = y2 - (y2-y1)*fraccion_inferior
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float64)


def caja_en_referencia(caja, matriz):
    puntos = transformar(esquinas_caja(caja), matriz)
    return [*puntos.min(axis=0), *puntos.max(axis=0)]


class CompensadorCamara:
    """Compara cada frame directamente con la referencia: no acumula matrices.

    La máscara selecciona SOLO fondo estático en la referencia. En ambos frames
    se excluyen las cajas conocidas; esto no descubre vehículos que YOLO omitió.
    """
    def __init__(self, referencia, mascara_fondo, cajas_referencia=(), ancho_analisis=960, max_puntos=5000):
        self.alto, self.ancho = referencia.shape[:2]
        escala = min(ancho_analisis/self.ancho, 1.)
        self.w = max(32, round(self.ancho*escala))
        self.h = max(32, round(self.alto*escala))
        self.factores = np.array([self.ancho/self.w, self.alto/self.h])
        if max_puntos < 100 or ancho_analisis < 64:
            raise ValueError('Se requieren al menos 100 puntos y ancho de análisis >=64.')
        self.orb = cv2.ORB_create(nfeatures=max_puntos, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.mascara = np.asarray(mascara_fondo, np.uint8)
        if self.mascara.shape != (self.alto,self.ancho):
            raise ValueError('La máscara del fondo no coincide con la referencia.')
        self.k_ref, self.d_ref = self._extraer(referencia, cajas_referencia, self.mascara)
        if self.d_ref is None or len(self.k_ref)<60:
            raise ValueError('Hay menos de 60 puntos de fondo. Selecciona edificios o marcas estáticas con --editar-fondo.')
        self.parametros = dict(ancho_analisis=self.w, max_puntos=max_puntos, puntos_referencia=len(self.k_ref),
                               ratio_descriptor=.70, min_parejas=30, min_inliers=25,
                               proporcion_inliers_min=.45, ransac_error_px=3.,
                               error_mediano_max_px=2., extension_x_min=.15,
                               extension_y_min=.08, area_soporte_min=.01,
                               escala_min=.8, escala_max=1.25, rotacion_max_grados=12.,
                               fraccion_vista_comun_min=.60)

    def _extraer(self, frame, cajas, mascara=None):
        mask = np.full((self.alto,self.ancho),255,np.uint8) if mascara is None else mascara.copy()
        for caja in cajas:
            x1,y1,x2,y2 = caja
            a,b = max(0,int(x1)-8),max(0,int(y1)-8)
            c,d = min(self.ancho-1,int(x2)+8),min(self.alto-1,int(y2)+8)
            if a<=c and b<=d:cv2.rectangle(mask,(a,b),(c,d),0,-1)
        gray = cv2.cvtColor(cv2.resize(frame,(self.w,self.h)),cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask,(self.w,self.h),interpolation=cv2.INTER_NEAREST)
        return self.orb.detectAndCompute(gray,mask)

    def estimar(self, frame, cajas=()):
        info = dict(valida=False,motivo='',parejas=0,inliers=0,proporcion_inliers=0.,
                    error_mediano_px=None,escala=None,rotacion_grados=None,
                    vista_comun=None,matriz_actual_a_ref=None)
        if frame.shape[:2]!=(self.alto,self.ancho):
            raise ValueError('La resolución cambió respecto al fondo de referencia.')
        kp,des = self._extraer(frame,cajas)
        if des is None or len(des)<2:
            info['motivo']='sin_descriptores';return None,info
        parejas = [v[0] for v in self.matcher.knnMatch(des,self.d_ref,k=2)
                   if len(v)==2 and v[0].distance < .70*v[1].distance]
        # Una característica de referencia no puede justificar varios emparejamientos.
        usados=set();buenas=[]
        for m in sorted(parejas,key=lambda v:v.distance):
            if m.trainIdx not in usados:
                buenas.append(m);usados.add(m.trainIdx)
        info['parejas']=len(buenas)
        if len(buenas)<30:
            info['motivo']='pocas_parejas';return None,info
        actual=np.float64([kp[m.queryIdx].pt for m in buenas])*self.factores
        ref=np.float64([self.k_ref[m.trainIdx].pt for m in buenas])*self.factores
        matriz,inliers=cv2.estimateAffinePartial2D(actual,ref,method=cv2.RANSAC,
                                                   ransacReprojThreshold=3.,maxIters=3000,
                                                   confidence=.995,refineIters=10)
        if matriz is None or inliers is None or not np.isfinite(matriz).all():
            info['motivo']='sin_transformacion';return None,info
        buenos=inliers.ravel().astype(bool);n=int(buenos.sum())
        info.update(inliers=n,proporcion_inliers=n/len(buenas))
        if n<25 or n/len(buenas)<.45:
            info['motivo']='consenso_insuficiente';return None,info
        errores=np.linalg.norm(transformar(actual[buenos],matriz)-ref[buenos],axis=1)
        mediana=float(np.median(errores));info['error_mediano_px']=mediana
        escala=math.hypot(matriz[0,0],matriz[1,0])
        angulo=math.degrees(math.atan2(matriz[1,0],matriz[0,0]))
        info.update(escala=escala,rotacion_grados=angulo)
        span=np.ptp(ref[buenos],axis=0)/[self.ancho,self.alto]
        area=cv2.contourArea(cv2.convexHull(ref[buenos].astype(np.float32)))/(self.ancho*self.alto)
        info['extension_relativa_xy']=span.tolist();info['area_soporte_relativa']=area
        if span[0]<.15 or span[1]<.08 or area<.01:
            info['motivo']='puntos_demasiado_concentrados';return None,info
        if mediana>2. or not .8<=escala<=1.25 or abs(angulo)>12.:
            info['motivo']='error_o_movimiento_excesivo';return None,info
        marco=np.float32([[0,0],[self.ancho,0],[self.ancho,self.alto],[0,self.alto]])
        movido=transformar(marco,matriz).astype(np.float32)
        inter,_=cv2.intersectConvexConvex(marco,movido)
        fraccion=float(inter)/(self.ancho*self.alto);info['vista_comun']=fraccion
        if fraccion<.60:
            info['motivo']='poca_vista_comun';return None,info
        info.update(valida=True,motivo='ok',matriz_actual_a_ref=matriz.tolist())
        return matriz,info
