"""Estado temporal de movimiento y proxy de cola visible en zonas seleccionadas.

Solo utiliza observaciones presentes con ID y coordenadas compensadas.
No infiere vehículos ocultos, causas de detención ni longitud física de cola.
"""
from collections import deque
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class ParametrosCola:
    ventana_s: float = 1.0
    permanencia_s: float = 2.0
    salida_s: float = 0.5
    umbral_detenido: float = 0.06
    umbral_entrada: float = 0.20
    umbral_salida: float = 0.30
    salto_maximo: float = 3.0
    muestras_minimas: int = 4
    tolerancia_salto_px: float = 8.0
    tolerancia_salto_alturas: float = 0.15

    def __post_init__(self):
        numeros = [v for k, v in asdict(self).items() if k not in ('muestras_minimas', 'tolerancia_salto_px', 'tolerancia_salto_alturas')]
        if not all(math.isfinite(v) and v > 0 for v in numeros):
            raise ValueError('Los tiempos y umbrales deben ser positivos y finitos.')
        if not self.umbral_detenido < self.umbral_entrada < self.umbral_salida < self.salto_maximo:
            raise ValueError('Se requiere detenido < entrada < salida < salto máximo.')
        for v in (self.tolerancia_salto_px, self.tolerancia_salto_alturas):
            if not math.isfinite(v) or v < 0:
                raise ValueError('Las tolerancias de salto deben ser finitas y no negativas.')
        if self.muestras_minimas < 3:
            raise ValueError('Se necesitan al menos tres muestras de posición.')


class EstimadorCola:
    def __init__(self, zonas_seleccionadas, parametros=None):
        self.zonas = set(zonas_seleccionadas)
        if not self.zonas:
            raise ValueError('Selecciona al menos una zona de aproximación para estimar cola.')
        self.p = parametros or ParametrosCola()
        self.tracks = {}
        self.ultimo_frame = None
        self.ultimo_tiempo = None

    @staticmethod
    def _nuevo(zona):
        return dict(zona=zona, puntos=deque(), baja_desde=None, detenido_desde=None,
                    salida_desde=None, en_cola=False)

    def actualizar(self, frame, tiempo, observaciones, cuentas, camara_valida):
        """Modifica obs añadiendo 'movimiento'; devuelve métricas por zona.

        Un hueco de detección/ID/zona rompe la evidencia continua. Se borran
        los ausentes: jamás se añaden a la cola visible desde la memoria.
        """
        if not math.isfinite(tiempo):
            raise ValueError('Tiempo de origen no finito.')
        if self.ultimo_frame is not None:
            if frame <= self.ultimo_frame or tiempo <= self.ultimo_tiempo:
                raise ValueError('Frame y tiempo deben avanzar estrictamente.')
            if frame != self.ultimo_frame + 1:
                self.tracks.clear()
        self.ultimo_frame, self.ultimo_tiempo = frame, tiempo
        if not camara_valida:
            self.tracks.clear()
        presentes = set()
        for o in observaciones:
            o['movimiento'] = self._observacion(o, tiempo, camara_valida, presentes)
        self.tracks = {k: v for k, v in self.tracks.items() if k in presentes}
        resultado = {}
        for zona in sorted(self.zonas):
            if zona not in cuentas:
                raise ValueError(f'No hay conteos para la zona seleccionada {zona}.')
            fraccion = cuentas[zona].get('fraccion_visible')
            if not camara_valida or cuentas[zona]['total'] is None:
                resultado[zona] = dict(estado='alineacion_invalida', observados=None,
                    cola_visible_proxy=None, detenidos=None, lentos=None, indeterminados=None,
                    cobertura_clasificacion=None, fraccion_visible=fraccion,
                    max_episodio_reducido_s=None)
                continue
            grupo = [o for o in observaciones if o['asignacion'].get('zona_id') == zona]
            indeterminados = sum(o['movimiento']['velocidad_relativa'] is None for o in grupo)
            episodios = [o['movimiento']['episodio_reducido_s'] for o in grupo
                         if o['movimiento']['en_cola_proxy']]
            resultado[zona] = dict(
                estado='parcial' if fraccion is not None and fraccion < .95 else 'visible',
                observados=len(grupo), cola_visible_proxy=sum(o['movimiento']['en_cola_proxy'] for o in grupo),
                detenidos=sum(o['movimiento']['estado'] == 'detenido' for o in grupo),
                lentos=sum(o['movimiento']['estado'] == 'lento' for o in grupo),
                indeterminados=indeterminados,
                cobertura_clasificacion=(len(grupo)-indeterminados)/len(grupo) if grupo else None,
                fraccion_visible=fraccion, max_episodio_reducido_s=max(episodios, default=0.))
        return resultado

    def _observacion(self, o, tiempo, camara_valida, presentes):
        salida = dict(estado='sin_historial', velocidad_relativa=None,
                      velocidad_px_s=None, en_cola_proxy=False,
                      episodio_reducido_s=0., detenido_observado_s=0.)
        if not camara_valida:
            salida['estado'] = 'alineacion_invalida'
            return salida
        zona = o['asignacion'].get('zona_id')
        if zona not in self.zonas:
            salida['estado'] = 'zona_no_evaluada'
            return salida
        ident = o.get('track_id')
        if ident is None:
            salida['estado'] = 'sin_id'
            return salida
        if ident in presentes:
            raise ValueError('ID repetido en un fotograma.')
        presentes.add(ident)
        x, y = o['punto_xy']
        caja = o['xyxy']
        alto = caja[3] - caja[1]
        if not all(math.isfinite(v) for v in (x, y, alto)) or alto <= 0:
            raise ValueError('Punto o altura de caja no válidos.')
        tr = self.tracks.get(ident)
        if tr is None or tr['zona'] != zona:
            tr = self._nuevo(zona)
            self.tracks[ident] = tr
        puntos = tr['puntos']
        if puntos:
            t0, x0, y0, h0 = puntos[-1]
            # La incertidumbre de localización no desaparece al aumentar los FPS.
            # Separamos una tolerancia espacial del desplazamiento por movimiento.
            # Esto NO modifica la velocidad calculada ni sus umbrales de cola.
            altura_media = max((alto+h0)/2, 1.)
            tolerancia = max(self.p.tolerancia_salto_px,
                             self.p.tolerancia_salto_alturas * altura_media)
            limite = tolerancia + self.p.salto_maximo * altura_media * (tiempo-t0)
            if math.hypot(x-x0, y-y0) > limite:
                tr = self._nuevo(zona)
                self.tracks[ident] = tr
                puntos = tr['puntos']
                salida['estado'] = 'salto_posicion'
        puntos.append((tiempo, x, y, alto))
        # Conserva una muestra inmediatamente anterior al borde de la ventana.
        while len(puntos) > 2 and puntos[1][0] <= tiempo-self.p.ventana_s:
            puntos.popleft()
        if len(puntos) < self.p.muestras_minimas or tiempo-puntos[0][0] < self.p.ventana_s-1e-6:
            return salida
        datos = np.asarray(puntos, dtype=float)
        dt = datos[:, 0] - datos[:, 0].mean()
        pos = datos[:, 1:3]
        pendiente = (dt[:, None]*(pos-pos.mean(axis=0))).sum(axis=0) / (dt@dt)
        velocidad = float(np.linalg.norm(pendiente))
        relativa = velocidad / max(float(np.median(datos[:, 3])), 1.)
        salida.update(velocidad_relativa=relativa, velocidad_px_s=velocidad)
        if relativa <= self.p.umbral_detenido:
            if tr['detenido_desde'] is None:
                tr['detenido_desde'] = tiempo
        else:
            tr['detenido_desde'] = None
        detenido_s = 0. if tr['detenido_desde'] is None else tiempo-tr['detenido_desde']
        if relativa <= self.p.umbral_entrada:
            if tr['baja_desde'] is None:
                tr['baja_desde'] = tiempo
            tr['salida_desde'] = None
            if tiempo-tr['baja_desde'] >= self.p.permanencia_s-1e-6:
                tr['en_cola'] = True
        elif not tr['en_cola']:
            tr['baja_desde'] = None
        elif relativa >= self.p.umbral_salida:
            if tr['salida_desde'] is None:
                tr['salida_desde'] = tiempo
            if tiempo-tr['salida_desde'] >= self.p.salida_s-1e-6:
                tr['en_cola'] = False
                tr['baja_desde'] = None
                tr['salida_desde'] = None
        else:
            tr['salida_desde'] = None
        if detenido_s >= self.p.permanencia_s-1e-6:
            estado = 'detenido'
        elif relativa <= self.p.umbral_entrada:
            estado = 'lento'
        elif relativa >= self.p.umbral_salida:
            estado = 'movimiento'
        else:
            estado = 'intermedio'
        salida.update(estado=estado, en_cola_proxy=tr['en_cola'],
            episodio_reducido_s=0. if tr['baja_desde'] is None else tiempo-tr['baja_desde'],
            detenido_observado_s=detenido_s)
        return salida


def agregar_argumentos(parser):
    parser.add_argument('--zonas-cola', nargs='+', help='IDs de zonas de aproximación, por ejemplo zona_01 zona_02')
    parser.add_argument('--ventana-s', type=float, default=1.)
    parser.add_argument('--permanencia-s', type=float, default=2.)
    parser.add_argument('--salida-s', type=float, default=.5)
    parser.add_argument('--umbral-detenido', type=float, default=.06)
    parser.add_argument('--umbral-entrada', type=float, default=.20)
    parser.add_argument('--umbral-salida', type=float, default=.30)
    parser.add_argument('--tolerancia-salto-px', type=float, default=8.)
    parser.add_argument('--tolerancia-salto-alturas', type=float, default=.15)


def parametros_argumentos(args):
    return ParametrosCola(**{k: getattr(args, k) for k in (
        'ventana_s', 'permanencia_s', 'salida_s', 'umbral_detenido', 'umbral_entrada', 'umbral_salida',
        'tolerancia_salto_px', 'tolerancia_salto_alturas')})
