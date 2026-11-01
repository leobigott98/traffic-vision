"""Conteo en imagen con múltiples zonas, polígonos cóncavos y cierre con snapping.

Ejemplo:
  python contar_multizonas.py --imagen bus.jpg --zonas zonas.json --tolerancia 5

Editor: clic agrega; clic cerca del primer punto o Enter cierra el polígono.
Luego escribe nombre y acceso en la TERMINAL. Repite para otra zona.
S guarda todas; R limpia el polígono actual; botón derecho deshace un punto;
U elimina la última zona guardada en el editor; Esc cancela toda la edición.

Asignación exclusiva por área de caja (o su banda inferior) dentro de cada zona.
La tolerancia solo se intenta si ninguna zona alcanza la cobertura mínima.
Una detección ambigua no se cuenta en ninguna zona, pero queda en el informe.
Los conteos son de objetos detectados en UNA imagen, no de vehículos en cola.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import unicodedata

from shapely.geometry import Polygon, box
from shapely.validation import explain_validity
import shapely

VEHICULOS = ("car", "motorcycle", "bus", "truck")
CLASES = ("person", *VEHICULOS)


def poligono_valido(puntos, ancho, alto):
    if len(puntos) < 3:
        raise ValueError("Se necesitan al menos tres vértices.")
    if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in puntos):
        raise ValueError("Coordenadas no válidas.")
    if any(not (0 <= x < ancho and 0 <= y < alto) for x, y in puntos):
        raise ValueError("Hay vértices fuera de la imagen.")
    if len(set(map(tuple, puntos))) != len(puntos):
        raise ValueError("No repitas vértices; el cierre se almacena implícitamente.")
    p = Polygon(puntos)
    if not p.is_valid or p.area <= 1e-6:
        raise ValueError(f"Polígono degenerado o con cruces: {explain_validity(p)}")
    return p


def preparar_zonas(config, ancho, alto, tolerancia):
    if config["alto_ancho"] != [alto, ancho]:
        raise ValueError("La resolución cambió. Dibuja un nuevo archivo de zonas.")
    zonas = config["zonas"]
    if not zonas:
        raise ValueError("El archivo no contiene zonas.")
    if len({z["id"] for z in zonas}) != len(zonas):
        raise ValueError("Cada zona debe tener un identificador diferente.")
    marco = box(0, 0, ancho, alto)
    preparados = []
    for z in zonas:
        if not all(isinstance(z.get(k), str) and z[k].strip() for k in ("id", "nombre", "acceso")):
            raise ValueError("Cada zona necesita id, nombre y acceso no vacíos.")
        p = poligono_valido(z["puntos_xy"], ancho, alto)
        preparados.append((z, p, p.buffer(tolerancia).intersection(marco)))
    return preparados


def decidir(puntajes, minimo, margen):
    orden = sorted(puntajes.items(), key=lambda par: (-par[1], par[0]))
    primero, valor = orden[0]
    segundo = orden[1][1] if len(orden) > 1 else 0.0
    if valor + 1e-9 < minimo:
        return "fuera", None
    # Incluso con margen=0, un empate exacto nunca se resuelve por orden del archivo.
    if segundo > 0 and (abs(valor - segundo) <= 1e-9 or valor - segundo + 1e-9 < margen):
        return "ambiguo", None
    return "asignado", primero


def asignar(xyxy, zonas, minimo=0.5, margen=0.1, fraccion_inferior=1.0, tolerancia=5.0):
    x1, y1, x2, y2 = xyxy
    if not all(math.isfinite(v) for v in xyxy) or x2 <= x1 or y2 <= y1:
        raise ValueError("Caja de detección inválida.")
    # 1.0 usa la caja completa; 0.25 usa el cuarto inferior de su altura.
    region = box(x1, y2 - (y2 - y1) * fraccion_inferior, x2, y2)
    crudos = {z["id"]: min(1.0, max(0.0, region.intersection(p).area / region.area))
              for z, p, ampliado in zonas}
    estado, elegida = decidir(crudos, minimo, margen)
    usados = crudos
    modo = "original"
    if estado == "fuera" and tolerancia > 0:
        usados = {z["id"]: min(1.0, max(0.0, region.intersection(ampliado).area / region.area))
                  for z, p, ampliado in zonas}
        estado, elegida = decidir(usados, minimo, margen)
        modo = "tolerancia"
    return {"estado": estado, "zona_id": elegida, "evaluacion": modo,
            "coberturas_originales": crudos, "coberturas_usadas": usados,
            "region_evaluada_xyxy": list(region.bounds)}


def leer_config(ruta, ancho, alto):
    config = json.loads(ruta.read_text(encoding="utf-8"))
    # Compatibilidad con zona_prueba.json del ejercicio anterior.
    if "zonas" not in config and "puntos_xy" in config:
        config = {"version": 2, "alto_ancho": config["alto_ancho"], "zonas": [
            {"id": "zona_01", "nombre": "zona_anterior", "acceso": "acceso_01",
             "puntos_xy": config["puntos_xy"]}]}
    preparar_zonas(config, ancho, alto, 0)
    return config


def ascii_text(texto):
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def editar_zonas(imagen, existentes, snap, cv2, np):
    """Clics y snapping en pantalla; persistencia en coordenadas originales."""
    alto, ancho = imagen.shape[:2]
    escala = min(1200 / ancho, 650 / alto, 1)
    w, h = max(1, round(ancho * escala)), max(1, round(alto * escala))
    base = cv2.resize(imagen, (w, h))
    zonas = json.loads(json.dumps(existentes))
    actual, cursor = [], [0, 0]
    cerrar = [False]
    titulo = "Zonas: clic dibuja | Enter cierra | S guarda | R limpia | U elimina zona | Esc cancela"

    def pantalla(p):
        return (round(p[0] * w / ancho), round(p[1] * h / alto))

    def cerca_inicio(x, y):
        return len(actual) >= 3 and math.dist((x, y), pantalla(actual[0])) <= snap

    def raton(evento, x, y, flags, param):
        cursor[:] = [x, y]
        if evento == cv2.EVENT_RBUTTONDOWN and actual:
            actual.pop()
        elif evento == cv2.EVENT_LBUTTONDOWN and 0 <= x < w and 0 <= y < h:
            if cerca_inicio(x, y):
                cerrar[0] = True
            elif not any(math.dist((x, y), pantalla(p)) < 3 for p in actual):
                actual.append((x * ancho / w, y * alto / h))
            else:
                print("Punto demasiado cercano a otro vértice.")

    def abrir():
        cv2.namedWindow(titulo, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(titulo, raton)

    abrir()
    try:
        while True:
            vista = base.copy()
            for i, z in enumerate(zonas):
                pts = np.array([pantalla(p) for p in z["puntos_xy"]], np.int32)
                cv2.polylines(vista, [pts], True, (220, 150, 50), 2)
                cv2.putText(vista, ascii_text(z["nombre"]), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 150, 50), 1)
            if actual:
                pts = np.array([pantalla(p) for p in actual], np.int32)
                cv2.polylines(vista, [pts], False, (0, 255, 255), 2)
                fin = pantalla(actual[0]) if cerca_inicio(*cursor) else tuple(cursor)
                cv2.line(vista, pantalla(actual[-1]), fin, (0, 255, 255), 1)
                for p in pts:
                    cv2.circle(vista, tuple(p), 3, (0, 255, 255), -1)
                cv2.circle(vista, pantalla(actual[0]), round(snap),
                           (0, 255, 0) if cerca_inicio(*cursor) else (0, 255, 255), 2)
            cv2.imshow(titulo, vista)
            tecla = cv2.waitKey(30) & 0xFF
            if tecla == 27 or cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if tecla in (ord("r"), ord("R")):
                actual.clear()
            if tecla in (ord("u"), ord("U")) and not actual and zonas:
                zonas.pop()
            if tecla in (10, 13):
                cerrar[0] = True
            if cerrar[0]:
                cerrar[0] = False
                try:
                    poligono_valido(actual, ancho, alto)
                except ValueError as exc:
                    print(exc)
                    continue
                cv2.destroyAllWindows()
                indice = 1
                while f"zona_{indice:02}" in {z["id"] for z in zonas}:
                    indice += 1
                ident = f"zona_{indice:02}"
                print("Polígono cerrado. Completa estos datos en la terminal:")
                nombre = input(f"Nombre de zona/carril [{ident}]: ").strip() or ident
                acceso = input("Acceso/calle [acceso_01]: ").strip() or "acceso_01"
                zonas.append({"id": ident, "nombre": nombre, "acceso": acceso, "puntos_xy": list(actual)})
                actual.clear()
                abrir()
            if tecla in (ord("s"), ord("S")):
                if actual:
                    print("Cierra el polígono pendiente con Enter o límpialo con R antes de guardar.")
                elif zonas:
                    return zonas
                else:
                    print("Dibuja al menos una zona.")
    finally:
        cv2.destroyAllWindows()


def advertir_solapes(zonas):
    for i, (a, pa, _) in enumerate(zonas):
        for b, pb, _ in zonas[i + 1:]:
            if pa.intersection(pb).area > 1e-6:
                print(f"Solapamiento: {a['nombre']} / {b['nombre']}. Se aplicará asignación exclusiva.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--imagen", type=Path, required=True)
    p.add_argument("--zonas", type=Path, required=True)
    p.add_argument("--editar", action="store_true", help="Abre editor con las zonas existentes")
    p.add_argument("--tolerancia", type=float, default=5, help="Expansión geométrica en píxeles ORIGINALES")
    p.add_argument("--snap", type=float, default=12, help="Radio de cierre en píxeles de PANTALLA")
    p.add_argument("--cobertura-minima", type=float, default=0.5)
    p.add_argument("--margen", type=float, default=0.1, help="Diferencia mínima entre las dos mejores coberturas")
    p.add_argument("--fraccion-inferior", type=float, default=1.0, help="1=caja completa; 0.25=cuarto inferior")
    p.add_argument("--modelo", default="yolo11n.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--nms-agnostica", action="store_true",
                   help="Suprime cajas muy superpuestas también entre clases distintas")
    p.add_argument("--detecciones", type=Path, help="Opcional: reutiliza un conteo.json previo de esta misma imagen")
    args = p.parse_args()
    valores = (args.tolerancia, args.snap, args.cobertura_minima, args.margen, args.fraccion_inferior, args.conf)
    if not all(math.isfinite(v) for v in valores):
        p.error("Los parámetros numéricos deben ser finitos.")
    if args.tolerancia < 0 or args.snap < 3 or not 0 < args.cobertura_minima <= 1 or not 0 <= args.margen <= 1:
        p.error("tolerancia>=0; snap>=3; cobertura mínima en (0,1]; margen en [0,1].")
    if not 0 < args.fraccion_inferior <= 1 or not 0 <= args.conf <= 1 or args.imgsz <= 0:
        p.error("fracción inferior en (0,1]; conf en [0,1]; imgsz positivo.")
    if not args.imagen.is_file() or args.zonas.suffix.lower() != ".json":
        p.error("Indica una imagen existente y un archivo de zonas con extensión .json.")
    import cv2
    import numpy as np
    imagen = cv2.imdecode(np.fromfile(str(args.imagen), np.uint8), cv2.IMREAD_COLOR)
    if imagen is None:
        p.error("No se pudo abrir la imagen.")
    alto, ancho = imagen.shape[:2]
    try:
        config = leer_config(args.zonas, ancho, alto) if args.zonas.exists() else {
            "version": 2, "alto_ancho": [alto, ancho], "zonas": []}
        if args.editar or not args.zonas.exists():
            nuevas = editar_zonas(imagen, config["zonas"], args.snap, cv2, np)
            if nuevas is None:
                print("Edición cancelada. No se guardaron cambios.")
                return
            config["zonas"] = nuevas
            config["imagen_referencia"] = str(args.imagen.resolve())
            args.zonas.parent.mkdir(parents=True, exist_ok=True)
            args.zonas.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        zonas = preparar_zonas(config, ancho, alto, args.tolerancia)
    except (ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print("Las zonas requieren el mismo encuadre, incluso si la resolución coincide.")
    advertir_solapes(zonas)

    if args.detecciones:
        previo = json.loads(args.detecciones.read_text(encoding="utf-8"))
        if previo.get("alto_ancho_original") != [alto, ancho]:
            p.error("El JSON de detecciones debe tener la misma resolución y proceder de esta imagen.")
        detecciones = [d.copy() for d in previo["detecciones"] if d["clase"] in CLASES]
        motor = {"origen": str(args.detecciones.resolve()), "modelo": previo.get("modelo"),
                 "configuracion": previo.get("configuracion"), "versiones": previo.get("versiones"),
                 "nota": "Reutilización: --modelo, --device, --conf, --imgsz y --nms-agnostica no se aplican; se conservan las detecciones del JSON."}
        print(motor["nota"])
    else:
        import torch
        import ultralytics
        from ultralytics import YOLO
        model = YOLO(args.modelo)
        if set(CLASES) - set(model.names.values()):
            p.error("El modelo no contiene todas las clases requeridas.")
        ids = [i for i, nombre in model.names.items() if nombre in CLASES]
        result = model.predict(imagen, device=args.device, conf=args.conf, imgsz=args.imgsz,
                               iou=0.7, agnostic_nms=args.nms_agnostica, classes=ids, save=False)[0]
        detecciones = [{"clase_id": int(b.cls.item()), "clase": result.names[int(b.cls.item())],
                        "confianza": float(b.conf.item()), "xyxy": b.xyxy[0].tolist()}
                       for b in result.boxes.cpu()]
        motor = {"modelo": args.modelo, "conf": args.conf, "imgsz": args.imgsz,
                 "device": args.device, "iou": 0.7, "tiempos_ms": result.speed,
                 "agnostic_nms": args.nms_agnostica,
                 "versiones": {"torch": torch.__version__, "ultralytics": ultralytics.__version__}}

    conteos = {z["id"]: dict.fromkeys(CLASES, 0) for z, _, _ in zonas}
    por_acceso = {z["acceso"]: dict.fromkeys(CLASES, 0) for z, _, _ in zonas}
    estado_clases = {s: dict.fromkeys(CLASES, 0) for s in ("asignado", "ambiguo", "fuera")}
    por_id = {z["id"]: z for z, _, _ in zonas}
    paleta = [(0,200,0), (255,170,0), (200,80,220), (0,180,240), (200,200,0), (150,100,255)]
    colores = {z["id"]: paleta[i % len(paleta)] for i, (z, _, _) in enumerate(zonas)}
    anotada = imagen.copy()
    for d in detecciones:
        a = asignar(d["xyxy"], zonas, args.cobertura_minima, args.margen, args.fraccion_inferior, args.tolerancia)
        d["asignacion"] = a
        estado_clases[a["estado"]][d["clase"]] += 1
        ident = a["zona_id"]
        if ident is not None:
            conteos[ident][d["clase"]] += 1
            por_acceso[por_id[ident]["acceso"]][d["clase"]] += 1
        color = colores[ident] if ident else ((0,140,255) if a["estado"] == "ambiguo" else (140,140,140))
        x1,y1,x2,y2 = map(round,d["xyxy"])
        cv2.rectangle(anotada,(x1,y1),(x2,y2),color,2)
        if args.fraccion_inferior < 1:
            rx1,ry1,rx2,ry2 = map(round,a["region_evaluada_xyxy"])
            cv2.rectangle(anotada,(rx1,ry1),(rx2,ry2),color,1)
        etiqueta = ascii_text(f"{d['clase']} {por_id[ident]['nombre'] if ident else a['estado']}")
        cv2.putText(anotada,etiqueta,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)
    for z, poli, ampliado in zonas:
        cv2.polylines(anotada,[np.rint(z["puntos_xy"]).astype(np.int32)],True,colores[z["id"]],3)
        v = sum(conteos[z["id"]][c] for c in VEHICULOS)
        punto = tuple(np.rint(z["puntos_xy"][0]).astype(int))
        cv2.putText(anotada,ascii_text(f"{z['nombre']}: V={v} P={conteos[z['id']]['person']}"),
                    punto,cv2.FONT_HERSHEY_SIMPLEX,0.55,colores[z["id"]],2)

    ahora = datetime.now(timezone.utc)
    salida = Path(__file__).resolve().parent / "resultados_multizonas" / ahora.strftime("%Y%m%dT%H%M%S_%fZ")
    salida.mkdir(parents=True, exist_ok=False)
    ok, jpg = cv2.imencode(".jpg",anotada)
    if not ok:
        raise RuntimeError("No se pudo generar la imagen.")
    jpg.tofile(str(salida / "detecciones_multizonas.jpg"))
    informe = {"fecha_utc": ahora.isoformat(), "imagen": str(args.imagen.resolve()),
               "alto_ancho_original": [alto,ancho], "configuracion_zonas": config, "motor": motor,
               "reglas": {"metrica": "area_interseccion/area_region_de_caja", "tolerancia_px_originales": args.tolerancia,
                          "cobertura_minima": args.cobertura_minima, "margen": args.margen,
                          "fraccion_inferior": args.fraccion_inferior, "tolerancia_solo_si_fuera": True},
               "versiones_geometria": {"python": platform.python_version(), "shapely": shapely.__version__, "opencv": cv2.__version__},
               "conteos_zona": conteos, "conteos_acceso": por_acceso, "estados_por_clase": estado_clases,
               "detecciones": detecciones, "alcance": "Asignación de detecciones en una imagen, no flujo ni cola."}
    (salida / "conteo_multizonas.json").write_text(json.dumps(informe,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\nZona | Acceso | Vehículos | Personas")
    for z,_,_ in zonas:
        c=conteos[z["id"]]
        print(f"{z['nombre']} | {z['acceso']} | {sum(c[n] for n in VEHICULOS)} | {c['person']}")
    print("\nTotales asignados por acceso:")
    for acceso,c in por_acceso.items():
        print(f"{acceso}: vehículos={sum(c[n] for n in VEHICULOS)}, personas={c['person']}")
    print("\nEstados (incluyen personas y vehículos):")
    for estado,c in estado_clases.items():
        print(f"{estado}: {sum(c.values())}")
    print(f"Resultados: {salida}")


if __name__ == "__main__":
    main()
