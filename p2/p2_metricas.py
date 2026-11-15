"""
Métricas de evaluación del seguimiento (sin dependencias de ROS: sólo numpy).

En cada paso del episodio se registra:
    t                instante (s desde el inicio del episodio)
    seguidor         posición (x, z) y orientación del seguidor
    guía             posición (x, z) del robot guía (la pelota va pegada a él)
    separación       distancia seguidor-guía (m)
    desviación       ángulo (grados) entre hacia dónde mira el seguidor y la dirección del guía
    acción           acción aplicada (a0 = velocidad, a1 = giro)

y de ahí salen, en conjunto y desglosadas por configuración del guía:
    Completados  número y % de episodios completados
    Desenlaces   número de episodios acabados por cada causa
    Separación   % de pasos dentro de la banda objetivo y separación media
    Desviación   media de |desviación| en grados y % de pasos por debajo de 15°
    Paradas      (sólo C4) paradas del guía y cuántas se superan

SUPUESTOS (comprobar con los datos reales; están todos en las constantes de aquí abajo):
  * Las poses de RoboboSim vienen en milímetros: (x, z, yaw).  [calibrar() imprime esa distancia como "distmm"]
  * El yaw (rotation.y de Unity) viene en grados y crece hacia la derecha: el frente es (sin yaw, cos yaw) en (x, z).
  * La separación se mide entre las posiciones de los dos robots (la pelota va pegada al guía).
  * "Completado" = el episodio llega a pasos_max sin choque, sin perder al guía y sin error del entorno.
"""
import csv
import math
import os
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------ constantes
BANDA_OBJETIVO_M = (0.20, 0.45)      # banda objetivo de separación (la misma que el entorno)
UMBRAL_DESV_GRADOS = 15.0            # "desviación pequeña": por debajo de estos grados
POSE_A_M = 1e-3                      # las poses del simulador están en mm
YAW_EN_GRADOS = True                 # False si rotation.y viniese en radianes

CAUSAS = ("completado", "choque", "perdido", "error")   # causas de fin de episodio (B.3)
CONFIGS_CON_PARADAS = ("C4",)        # configuraciones en las que el guía se para

# Paradas del guía (se detectan a partir de su posición, paso a paso)
V_PARADA_M_S = 0.01                  # por debajo de esta velocidad un robot está parado
MIN_PARADA_PASOS = 3                 # pasos seguidos parado para contar como una parada (filtra ruido de pose)
VENTANA_TRAS_REANUDAR_S = 10.0       # el seguidor debe seguir sin fallo este tiempo tras la reanudación


# ------------------------------------------------------------------ geometría
def _pose_a_si(pose):
    """(x_mm, z_mm, yaw) -> (x_m, z_m, yaw_grados), o None si falta."""
    if pose is None or any(v is None for v in pose[:3]):
        return None
    yaw = float(pose[2]) if YAW_EN_GRADOS else math.degrees(float(pose[2]))
    return float(pose[0]) * POSE_A_M, float(pose[1]) * POSE_A_M, yaw


def desviacion_deg(seguidor, guia):
    """Ángulo (grados, en (-180, 180]) entre el frente del seguidor y la dirección hacia el guía.
    Positivo = el guía queda a la derecha del seguidor."""
    dx, dz = guia[0] - seguidor[0], guia[1] - seguidor[1]
    if dx == 0.0 and dz == 0.0:
        return float("nan")
    ang = math.degrees(math.atan2(dx, dz)) - seguidor[2]
    return (ang + 180.0) % 360.0 - 180.0


def _separacion_estimada(info):
    """Plan B si no hay poses: la distancia que estima el entorno a partir del blob
    (info['distancia_real_mm'] viene en METROS en p1_environment.py; si fuese >5 se trata como mm)."""
    v = info.get("distancia_real_mm")
    if v is None:
        return float("nan")
    v = float(v)
    return v / 1000.0 if v > 5.0 else v


# ------------------------------------------------------------------ registro
@dataclass
class Paso:
    t: float
    seg_x: float
    seg_z: float
    seg_yaw: float
    guia_x: float
    guia_z: float
    separacion_m: float
    desviacion_deg: float          # con signo; las métricas usan el valor absoluto
    a0: float
    a1: float
    visible: bool
    choque: bool
    perdido: bool

    @property
    def en_banda(self):
        return BANDA_OBJETIVO_M[0] <= self.separacion_m <= BANDA_OBJETIVO_M[1]


@dataclass
class Episodio:
    config: str
    indice: int
    pasos: list = field(default_factory=list)
    desenlace: str = "completado"
    duracion_s: float = 0.0
    modelo: str = ""              # modelo (.pkl) con el que se jugó el episodio: cada configuración tiene el suyo


def crear_paso(t, info, accion, pose_guia):
    """Construye el registro de un paso a partir de info (del entorno), la acción y la pose del guía."""
    nan = float("nan")
    seg = _pose_a_si(info.get("pose_sim"))
    gui = _pose_a_si(pose_guia)
    if seg is not None and gui is not None:
        sep = math.hypot(gui[0] - seg[0], gui[1] - seg[1])
        dev = desviacion_deg(seg, gui)
    else:
        sep, dev = _separacion_estimada(info), nan
    return Paso(
        t=float(t),
        seg_x=seg[0] if seg else nan, seg_z=seg[1] if seg else nan, seg_yaw=seg[2] if seg else nan,
        guia_x=gui[0] if gui else nan, guia_z=gui[1] if gui else nan,
        separacion_m=sep, desviacion_deg=dev,
        a0=float(accion[0]), a1=float(accion[1]),
        visible=bool(info.get("visible", False)),
        choque=bool(info.get("choque", False)), perdido=bool(info.get("perdido", False)))


def clasificar_desenlace(info_final, error=False):
    if error:
        return "error"
    if info_final.get("choque"):
        return "choque"
    if info_final.get("perdido"):
        return "perdido"
    return "completado"


# ------------------------------------------------------------------ paradas
def _velocidades(ep, x, z):
    """Velocidad (m/s) en cada paso a partir de la posición (x, z) y de los instantes reales."""
    t = np.array([p.t for p in ep.pasos])
    xs = np.array([getattr(p, x) for p in ep.pasos])
    zs = np.array([getattr(p, z) for p in ep.pasos])
    v = np.full(len(t), np.nan)
    if len(t) > 1:
        dt = np.maximum(np.diff(t), 1e-6)
        v[1:] = np.hypot(np.diff(xs), np.diff(zs)) / dt
    return t, v


def analizar_paradas(ep):
    """Lista de paradas del guía en un episodio, con el veredicto de cada una.

    Una parada es un tramo de >= MIN_PARADA_PASOS pasos con el guía por debajo de V_PARADA_M_S.
    Se SUPERA si:
      1) el seguidor llega a pararse (velocidad < V_PARADA_M_S) mientras el guía está parado,
      2) no hay choque desde que el guía se para, y
      3) no hay fallo (choque / guía perdido) hasta VENTANA_TRAS_REANUDAR_S s después de que el guía reanude.
    Si el episodio acaba (completado) antes de esos 10 s se da por superada y se marca ventana_truncada.
    Si el guía no llega a reanudar antes de que acabe el episodio no se puede comprobar: no se supera.
    """
    n = len(ep.pasos)
    if n < 2:
        return []
    t, vg = _velocidades(ep, "guia_x", "guia_z")
    _, vs = _velocidades(ep, "seg_x", "seg_z")
    parado = np.where(np.isnan(vg), False, vg < V_PARADA_M_S)

    paradas, i = [], 0
    while i < n:
        if not parado[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and parado[j + 1]:
            j += 1
        if j - i + 1 >= MIN_PARADA_PASOS:
            paradas.append((i, j))
        i = j + 1

    t_fallo = ep.pasos[-1].t if ep.desenlace in ("choque", "perdido", "error") else None
    resultado = []
    for a, b in paradas:
        reanuda = b + 1 < n                       # existe un paso posterior con el guía en marcha
        t_reanuda = t[b + 1] if reanuda else None
        # el seguidor se detiene mientras el guía está parado (o justo al reanudar)
        hasta = min(b + 1, n - 1)
        se_detiene = bool(np.any(~np.isnan(vs[a:hasta + 1]) & (vs[a:hasta + 1] < V_PARADA_M_S)))
        choca = t_fallo is not None and ep.desenlace == "choque" and t_fallo >= t[a] and (
            t_reanuda is None or t_fallo <= t_reanuda + VENTANA_TRAS_REANUDAR_S)
        falla_tras = False
        truncada = False
        if reanuda:
            limite = t_reanuda + VENTANA_TRAS_REANUDAR_S
            falla_tras = t_fallo is not None and t_fallo >= t[a] and t_fallo <= limite
            truncada = (t[-1] < limite) and t_fallo is None
        superada = bool(reanuda and se_detiene and not choca and not falla_tras)
        resultado.append(dict(inicio_s=float(t[a]), fin_s=float(t[b]),
                              reanuda_s=None if t_reanuda is None else float(t_reanuda),
                              seguidor_se_detiene=se_detiene, choque=bool(choca),
                              fallo_tras_reanudar=bool(falla_tras), ventana_truncada=bool(truncada),
                              # sólo si el episodio acabó bien y aun así el guía no llegó a reanudar
                              sin_reanudar=(not reanuda) and t_fallo is None, superada=superada))
    return resultado


# ------------------------------------------------------------------ métricas
def metricas(episodios, con_paradas=False):
    """Métricas de un conjunto de episodios (todos de la misma configuración, o todos juntos)."""
    n = len(episodios)
    desenlaces = {c: 0 for c in CAUSAS}
    for ep in episodios:
        desenlaces[ep.desenlace] = desenlaces.get(ep.desenlace, 0) + 1

    sep = np.array([p.separacion_m for ep in episodios for p in ep.pasos], dtype=float)
    dev = np.abs(np.array([p.desviacion_deg for ep in episodios for p in ep.pasos], dtype=float))
    sep_ok, dev_ok = sep[~np.isnan(sep)], dev[~np.isnan(dev)]

    m = dict(
        episodios=n,
        modelo=(sorted({e.modelo for e in episodios}) or [""])[0] if len({e.modelo for e in episodios}) == 1 else "",
        completados=desenlaces["completado"],
        pct_completados=100.0 * desenlaces["completado"] / n if n else float("nan"),
        desenlaces=desenlaces,
        pasos=int(sum(len(ep.pasos) for ep in episodios)),
        pct_banda=100.0 * float(np.mean((sep_ok >= BANDA_OBJETIVO_M[0]) & (sep_ok <= BANDA_OBJETIVO_M[1])))
        if len(sep_ok) else float("nan"),
        sep_media_m=float(sep_ok.mean()) if len(sep_ok) else float("nan"),
        desv_media_abs_deg=float(dev_ok.mean()) if len(dev_ok) else float("nan"),
        pct_desv_menor_umbral=100.0 * float(np.mean(dev_ok < UMBRAL_DESV_GRADOS)) if len(dev_ok) else float("nan"),
        paradas=None, paradas_superadas=None, paradas_detalle=None)
    if con_paradas:
        det = [dict(p, episodio=ep.indice) for ep in episodios for p in analizar_paradas(ep)]
        m["paradas"] = len(det)
        m["paradas_superadas"] = sum(p["superada"] for p in det)
        m["paradas_detalle"] = det
    return m


def metricas_por_configuracion(episodios):
    """{config: métricas, ..., 'TOTAL': métricas de todo junto}. Las paradas sólo se evalúan en CONFIGS_CON_PARADAS."""
    configs = sorted({ep.config for ep in episodios})
    out = {c: metricas([e for e in episodios if e.config == c], con_paradas=c in CONFIGS_CON_PARADAS)
           for c in configs}
    paradas_eps = [e for e in episodios if e.config in CONFIGS_CON_PARADAS]
    total = metricas(episodios, con_paradas=False)
    if paradas_eps:
        det = [p for c in CONFIGS_CON_PARADAS if c in out and out[c]["paradas_detalle"] for p in out[c]["paradas_detalle"]]
        total["paradas"], total["paradas_superadas"] = len(det), sum(p["superada"] for p in det)
        total["paradas_detalle"] = det
    out["TOTAL"] = total
    return out


# ------------------------------------------------------------------ salida
def _f(v, fmt):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def formatear_tabla(res):
    """Tabla de texto: una columna por configuración y otra con el total."""
    cols = [c for c in res if c != "TOTAL"] + ["TOTAL"]
    ancho = 17
    filas = [("Modelo", lambda m: os.path.basename(m["modelo"])[-(ancho - 2):] if m.get("modelo") else "—"),
             ("Episodios", lambda m: str(m["episodios"])),
             ("Completados", lambda m: f"{m['completados']} ({_f(m['pct_completados'], '{:.0f}')}%)")]
    filas += [(f"  fin: {c}", (lambda c: lambda m: str(m["desenlaces"].get(c, 0)))(c)) for c in CAUSAS]
    filas += [("Separación en banda", lambda m: _f(m["pct_banda"], "{:.1f}%")),
              ("Separación media", lambda m: _f(m["sep_media_m"], "{:.3f} m")),
              ("|Desviación| media", lambda m: _f(m["desv_media_abs_deg"], "{:.1f}°")),
              (f"Pasos con desv.<{UMBRAL_DESV_GRADOS:.0f}°", lambda m: _f(m["pct_desv_menor_umbral"], "{:.1f}%")),
              ("Paradas superadas", lambda m: "—" if m["paradas"] is None else f"{m['paradas_superadas']}/{m['paradas']}")]
    lin = ["{:<26s}".format("Métrica") + "".join("{:>{w}s}".format(c, w=ancho) for c in cols)]
    lin.append("-" * len(lin[0]))
    for nombre, f in filas:
        lin.append("{:<26s}".format(nombre) + "".join("{:>{w}s}".format(f(res[c]), w=ancho) for c in cols))
    return "\n".join(lin)


CABECERA_CSV = ["config", "episodio", "paso", "t_s", "seguidor_x_m", "seguidor_z_m", "seguidor_yaw_deg",
                "guia_x_m", "guia_z_m", "separacion_m", "desviacion_deg", "a0", "a1",
                "visible", "choque", "perdido", "desenlace_episodio", "modelo"]


def escribir_csv(episodios, ruta):
    """Un registro por paso (instante, posiciones, separación, desviación, acción) de todos los episodios."""
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CABECERA_CSV)
        for ep in episodios:
            for k, p in enumerate(ep.pasos, 1):
                w.writerow([ep.config, ep.indice, k, f"{p.t:.3f}", f"{p.seg_x:.4f}", f"{p.seg_z:.4f}",
                            f"{p.seg_yaw:.2f}", f"{p.guia_x:.4f}", f"{p.guia_z:.4f}", f"{p.separacion_m:.4f}",
                            f"{p.desviacion_deg:.2f}", f"{p.a0:.4f}", f"{p.a1:.4f}",
                            int(p.visible), int(p.choque), int(p.perdido), ep.desenlace, ep.modelo])
