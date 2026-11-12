"""
follow_sim.py - lightweight 2D simulator of a differential-drive robot that has to
follow a leader robot around a closed green lane, seen through a front camera.

* The physics/sensor core only needs the standard library + numpy, so it is safe and
  fast to use inside multiprocessing workers (pygame is imported lazily by the viewer).
* The control period `dt` is arbitrary; physics are integrated with sub-steps of <= 10 ms.
* `FollowEnv.step()` returns the same `info` dict your real simulator produces (plus a few extras).
* There is a single lane (the green one of track.png). The episode ends as soon as the follower's
  centre leaves it (info["fuera_camino"]).
* The leader is configured with `LeaderConfig`: its speed on straights and on curves, and the
  probability / duration of its stops.

Run `python follow_sim.py` to watch a hand-written P controller (sanity check).
"""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# World. Geometry measured on track.png (449 x 503 px): the green lane is 25 px wide and its centre
# line is the rounded rectangle below (fitted to the image, within ~2 px of the true centre line).
# PX_PER_M is the ONE assumption about real size: 200 px/m makes the lane 12.5 cm wide.
# --------------------------------------------------------------------------- #
PX_PER_M = 200.0
WORLD_W_PX, WORLD_H_PX = 449, 503
LANE_WIDTH_PX = 25
LANE_HALF_M = LANE_WIDTH_PX / 2.0 / PX_PER_M            # 0.0625 m: the follower must stay this close to the centre line
GREEN_RECT = dict(left=23.3, right=416.7, top=24.5, bottom=476.3, radius=61.9)

# --------------------------------------------------------------------------- #
# Robot / sensor parameters
# --------------------------------------------------------------------------- #
ROBOT_LEN_M = 0.16
ROBOT_WID_M = 0.10

# Actions exactly as in RoboboSeguimientoEnv.step():
#   v = V_MIN + (a0 + 1) / 2 * (V_MAX - V_MIN)      (forward speed, never negative)
#   w = a1 * W_MAX                                   (angular speed, + = turn left)
V_MIN = 0.0             # m/s
V_MAX = 0.221           # m/s, MEASURED at full command (the real env commands 0.15 but the robot does ~0.221)
W_MAX = 0.54            # rad/s, MEASURED at full command (31 deg/s; the real env commands 0.35 rad/s)
W_SIGN = 1.0            # +1: positive angular speed turns left (ROS convention); -1 if the real robot turns right
MOTOR_TAU = 0.08        # first-order lag of the executed v / w (s)
SUBSTEP = 0.01          # max integration sub-step (s)

HALF_FOV = math.radians(35.0)
CAM_RANGE_M = 1.2
BLOB_NOISE = 0.02       # std of blob_x, normalised units (centroid jitter)
# --------------------------------------------------------------------------- #
# Camera + ball model, CALIBRATED ON THE OFFICIAL SIMULATOR (datos.txt: blob size at known positions).
#
# What the official simulator reports as the blob "size" is an AREA in pixels of a very small image of the ball
# (about 13 px across at 0.30 m), and it only takes discrete values (7, 14, 22, 43, 57, 72, 90, 109, 131, ...):
#       radius_px = BLOB_C_MM_PX / distance_mm                     (area ~ 1/d^2: pinhole)
#       area      = pi * r_q^2   with   r_q = BLOB_STEP_PX * n + BLOB_OFFS_PX,  n = 3, 4, 5, ... (integer)
# so one quantum is ~8 % of the distance at 0.30 m (25 mm), plus a flicker of about one quantum that lasts
# 1-3 frames. The conversion the controller uses to get a distance from the area is size_to_dist_m() below:
# USE THE SAME FUNCTION WHEN YOU DEPLOY (in the official-simulator client).
# --------------------------------------------------------------------------- #
IMG_W_PX = 160                  # camera image width (ASSUMED; it only matters for the ball cut by the image border)
BLOB_C_MM_PX = 1975.0           # fitted on datos.txt (bootstrap 5-95 %: 1820-2130). Distance = centre-to-centre.
BLOB_STEP_PX = 0.555            # radius quantum of the reported area (fitted: levels 7,14,22,43,57,72,90,109,131)
BLOB_OFFS_PX = -0.20
BLOB_N_MIN = 3                  # smallest detectable blob: n = 3  (area 7 px)
RADIUS_NOISE_PX = 0.25          # frame-to-frame jitter of the blob radius (px) -> the one-level flicker in the data
RADIUS_NOISE_AR = 0.5           # ... and how much of it carries over to the next frame (flicker lasts 1-3 frames)
BALL_BACK_M = 0.0               # ball centre behind the leader's centre. 0: the data's reference points ARE the
                                # calibration reference; set > 0 only if you re-fit BLOB_C_MM_PX accordingly
CAM_FWD_M = 0.0                 # camera ahead of the follower's centre (same remark)
BLOB_FORMA = "esfera"           # "esfera": silhouette independent of viewing angle; "disco": flat disc on the
                                # leader's back whose area shrinks with cos(viewing angle)
BLOB_AREA_MAX = None            # if set (e.g. 133): the reported area saturates there (hypothesis: the official
                                # simulator stops growing the blob when the leader is closer than ~0.3 m)
K_CAL = 1.0                     # calibration factor of the size->distance conversion (1.0 = same as the data)
F_PX = (IMG_W_PX / 2.0) / math.tan(HALF_FOV)
BALL_RADIUS_M = BLOB_C_MM_PX / F_PX / 1000.0    # physical radius implied by the calibration (only for drawing)


def size_to_dist_m(area_px, k_cal=1.0):
    """Distance (m) the controller infers from the blob area (pixels) reported by the camera."""
    return k_cal * BLOB_C_MM_PX / math.sqrt(max(area_px, 1.0) / math.pi) / 1000.0


def _frac_inside(u, r, w):
    """Fraction of a disc (centre column u, radius r) that lies inside the image columns [0, w]."""
    def out(e):          # area of the disc beyond a vertical line at signed distance e from its centre
        if e >= r:
            return 0.0
        if e <= -r:
            return math.pi * r * r
        return r * r * math.acos(e / r) - e * math.sqrt(max(r * r - e * e, 0.0))
    a = math.pi * r * r
    return max(0.0, (a - out(u) - out(w - u)) / a)


BLOB_VIGENCIA_S = 0.4   # a detection still counts as "visible" (stale values) for this long

# --------------------------------------------------------------------------- #
# Task constants (same meaning as in your fitness function)
# --------------------------------------------------------------------------- #
DIST_CHOQUE_M = 0.12
DIST_TARGET_M = 0.3
BANDA_OBJETIVO_M = (0.2, 0.45)
NORM_BANDA_OBJETIVO = tuple((b - DIST_TARGET_M) / DIST_TARGET_M for b in BANDA_OBJETIVO_M)
X_TOLERANCIA = 0.15
PASO_S = 0.29                                 # measured step period of the real env (probar_robot.py: 0.275-0.313 s)
PASOS_PERDIDO_MAX = 4                         # unseen steps before the episode counts as lost
LOST_TIMEOUT_S = PASOS_PERDIDO_MAX * PASO_S   # 4 unseen steps, kept in seconds so any dt behaves alike

N_INPUTS = 7
N_OUTPUTS = 2


# --------------------------------------------------------------------------- #
# Leader configuration
# --------------------------------------------------------------------------- #
def _rango(x, nombre):
    """A number, or a (min, max) pair -> a validated (min, max) tuple."""
    if isinstance(x, (int, float)):
        lo = hi = float(x)
    else:
        x = tuple(float(v) for v in x)
        if len(x) == 1:
            lo = hi = x[0]
        elif len(x) == 2:
            lo, hi = x
        else:
            raise ValueError(f"{nombre}: expected one number or (min, max), got {x}")
    if lo < 0 or hi < lo:
        raise ValueError(f"{nombre}: need 0 <= min <= max, got ({lo}, {hi})")
    return (lo, hi)

# Leader driving modes (--modo N in train.py). Speeds in m/s; (min, max) = drawn per episode.
modos = {
    1: {"v_recta": (0.04, 0.04), "v_curva": (0.04, 0.04), "p_parada": 0.0, "t_parada": (1.0, 2.5), "parada_cada_s": 4.0, "aceleracion": 0.1},
    2: {"v_recta": (0.04, 0.04), "v_curva": (0.16, 0.16), "p_parada": 0.0, "t_parada": (1.0, 2.5), "parada_cada_s": 4.0, "aceleracion": 0.1},
    3: {"v_recta": (0.04, 0.16), "v_curva": (0.04, 0.16), "p_parada": 0.0, "t_parada": (1.0, 2.5), "parada_cada_s": 4.0, "aceleracion": 0.1},
    4: {"v_recta": (0.1, 0.1), "v_curva": (0.1, 0.1), "p_parada": 0.2, "t_parada": (1.0, 2.5), "parada_cada_s": 4.0, "aceleracion": 0.1},
    # 5: slow leader that often stops for a long time: the follower must brake, stay put without steering around
    #    in place, and set off again smoothly
    5: {"v_recta": (0.04, 0.10), "v_curva": (0.04, 0.10), "p_parada": 0.5, "t_parada": (4.0, 12.0), "parada_cada_s": 6.0, "aceleracion": 0.1},
}


@dataclass
class LeaderConfig:
    """How the guiding robot drives. Speeds are in m/s. A number means a fixed value; a (min, max)
    pair means a value drawn uniformly at the start of every episode.

    v_recta        speed on the straight sections
    v_curva        speed on the curves
    p_parada       probability that the leader stops at each stop check (0 = never stops)
    t_parada       how long it stays stopped, in seconds (number or (min, max))
    parada_cada_s  seconds between stop checks (a check is skipped while it is already stopped)
    aceleracion    m/s^2 limit on how fast the leader changes speed (0 = instantaneous changes)

    Use LeaderConfig.desde_modo(n) to get one of the presets in `modos`.
    """
    v_recta: tuple = (0.04, 0.04)
    v_curva: tuple = (0.04, 0.04)
    p_parada: float = 0.
    t_parada: tuple = (1.0, 2.5)
    parada_cada_s: float = 4.0
    aceleracion: float = 0.1

    @classmethod
    def desde_modo(cls, modo):
        if modo not in modos:       # an unknown mode used to fall back silently to the defaults (= mode 1)
            raise ValueError(f"unknown leader mode {modo!r}; available: {sorted(modos)}")
        return cls(**modos[modo])

    def __post_init__(self):
        self.v_recta = _rango(self.v_recta, "v_recta")
        self.v_curva = _rango(self.v_curva, "v_curva")
        self.t_parada = _rango(self.t_parada, "t_parada")
        self.p_parada = float(self.p_parada)
        self.parada_cada_s = float(self.parada_cada_s)
        self.aceleracion = float(self.aceleracion)
        if not 0.0 <= self.p_parada <= 1.0:
            raise ValueError(f"p_parada must be in [0, 1], got {self.p_parada}")
        if self.p_parada > 0 and self.parada_cada_s < 0.1:
            raise ValueError("parada_cada_s must be >= 0.1 s when p_parada > 0")
        if self.aceleracion < 0:
            raise ValueError("aceleracion must be >= 0")

    def advertencias(self):
        """Things that make the task impossible or degenerate (printed once by the caller)."""
        out = []
        vmax = max(self.v_recta[1], self.v_curva[1])
        if vmax > 0.95 * V_MAX:
            out.append(f"the leader can reach {vmax:.3f} m/s but the follower tops out at {V_MAX:.3f} m/s: "
                       "it will lose the leader")
        if self.v_recta[1] == 0 and self.v_curva[1] == 0:
            out.append("the leader never moves")
        return out

    def resumen(self):
        f = lambda r: f"{r[0]:.3f}" if r[0] == r[1] else f"{r[0]:.3f}-{r[1]:.3f}"
        return (f"leader: straights {f(self.v_recta)} m/s, curves {f(self.v_curva)} m/s, "
                f"stops p={self.p_parada:g} every {self.parada_cada_s:g} s for {f(self.t_parada)} s, "
                f"accel {self.aceleracion:g} m/s^2")


# --------------------------------------------------------------------------- #
# Leader path (centre line of the green lane)
# --------------------------------------------------------------------------- #
def _rounded_rect_px(left, right, top, bottom, r, n=24):
    """Closed polyline (px, y down) travelling counter-clockwise (up the right edge first), and a flag
    per segment telling whether it belongs to a curve (True) or to a straight (False)."""
    def arc(cx, cy, a0, a1):
        a = np.radians(np.linspace(a0, a1, n))
        return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])

    pts = np.vstack([
        arc(right - r, top + r, 0, -90),
        arc(left + r, top + r, -90, -180),
        arc(left + r, bottom - r, 180, 90),
        arc(right - r, bottom - r, 90, 0),
    ])
    # segments inside an arc are curve segments; the one joining two arcs (index n-1 of each block,
    # and the closing one) is a straight
    seg_curva = (np.arange(4 * n) % n) != (n - 1)
    return np.vstack([pts, pts[:1]]), seg_curva


class Path:
    """Closed path in world metres (y up) parametrised by arc length."""

    def __init__(self, pts_px, seg_curva):
        p = np.asarray(pts_px, dtype=float)
        self.pts_px = p
        self.pts = np.column_stack([p[:, 0] / PX_PER_M, (WORLD_H_PX - p[:, 1]) / PX_PER_M])
        seg = np.hypot(*np.diff(self.pts, axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])
        self.seg_curva = np.asarray(seg_curva, dtype=bool)
        self._a = self.pts[:-1]
        self._ab = self.pts[1:] - self.pts[:-1]
        self._ab2 = np.maximum((self._ab ** 2).sum(axis=1), 1e-12)

    def _seg(self, s):
        return min(int(np.searchsorted(self.s, s % self.length, side="right")) - 1, len(self.s) - 2)

    def distance_to(self, x, y):
        """Shortest distance (m) from the point (x, y) to the path polyline."""
        p = np.array([x, y])
        ap = p - self._a
        t = np.clip((ap * self._ab).sum(axis=1) / self._ab2, 0.0, 1.0)
        d = p - (self._a + self._ab * t[:, None])
        return float(np.sqrt((d ** 2).sum(axis=1).min()))

    def es_curva(self, s):
        return bool(self.seg_curva[self._seg(s)])

    def at(self, s):
        s = s % self.length
        x = np.interp(s, self.s, self.pts[:, 0])
        y = np.interp(s, self.s, self.pts[:, 1])
        i = self._seg(s)
        th = math.atan2(self.pts[i + 1, 1] - self.pts[i, 1], self.pts[i + 1, 0] - self.pts[i, 0])
        return float(x), float(y), th


def _make_path():
    g = GREEN_RECT
    pts, seg_curva = _rounded_rect_px(g["left"], g["right"], g["top"], g["bottom"], g["radius"])
    return Path(pts, seg_curva)


PATH = _make_path()


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class FollowEnv:
    """
    obs (7 floats): [blob_x, dist, visible, last_blob_x, last_dist, prev_a0, prev_a1]
        blob_x in [-1, 1] (+ = leader on the right), dist = (d - target)/target in [-1, 1]
        (-1 close, +1 far). When the leader is not visible blob_x = dist = 0 and the
        'last_*' inputs hold the last seen values (gives the net a little memory).
        A detection stays "visible" with its last (stale) values for BLOB_VIGENCIA_S.
    action: 2 floats in [-1, 1], as in RoboboSeguimientoEnv:
        a0 -> forward speed v in [V_MIN, V_MAX] m/s,  a1 -> angular speed w = a1 * W_MAX rad/s.
    The episode ends on: crash (choque), lost leader for LOST_TIMEOUT_S (perdido), the follower's centre
    leaving the green lane (fuera_camino), or after max_time seconds.

    leader: LeaderConfig (speeds, stops). margen_carril_m: extra tolerance added to the lane half width.
    """

    def __init__(self, dt=PASO_S, max_time=30.0, leader=None, margen_carril_m=0.0, aleatorizar=False, modo=1, p_saturacion=0.0):
        self.p_saturacion = float(p_saturacion)   # with randomisation: probability that the blob area saturates in an episode
        self.aleatorizar = bool(aleatorizar)     # domain randomisation of the plant/sensor (training robustness)
        self.dt = float(dt)
        self.max_time = float(max_time)
        self.max_steps = max(1, int(round(max_time / dt)))
        self.cfg = leader if leader is not None else LeaderConfig.desde_modo(modo)
        self.lane_half_m = LANE_HALF_M + float(margen_carril_m)
        self.path = PATH
        self.prev_action = (0.0, 0.0)
        self.action = (0.0, 0.0)
        self.reset(0)

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, s0=None):
        """s0: leader's starting arc position (m) along the lap; None -> the default start (length - 1.2 m)."""
        # three independent streams: the world (start pose, leader speeds, stops) must not depend on how many
        # camera-noise samples the follower's behaviour happens to draw
        ss_mundo, ss_ruido, ss_planta = np.random.SeedSequence(seed).spawn(3)
        self.rng_planta = np.random.default_rng(ss_planta)
        rng = self.rng_lider = np.random.default_rng(ss_mundo)
        self.rng = np.random.default_rng(ss_ruido)
        c = self.cfg

        self.v_recta_ep = rng.uniform(*c.v_recta)
        self.v_curva_ep = rng.uniform(*c.v_curva)

        # ---- per-episode plant / sensor parameters (nominal unless domain randomisation is on) ----
        self.v_max, self.w_max, self.w_bias = V_MAX, W_MAX, 0.0
        self.motor_tau, self.blob_noise, = MOTOR_TAU, BLOB_NOISE
        self.sensor_delay, self.p_drop, self.dt_jitter = 0, 0.0, 0.0
        self.r_noise_px, self.k_cal, self.quant_offs = RADIUS_NOISE_PX, K_CAL, BLOB_OFFS_PX
        self.area_max = BLOB_AREA_MAX
        self._r_err = 0.0
        self.sway_a, self.sway_w, self.sway_ph = 0.0, 1.0, 0.0     # leader's lateral weaving (m, rad/s, rad)
        if self.aleatorizar:
            rp = self.rng_planta
            k = rp.uniform(0.75, 1.35)                    # leader speed (real one is not exactly the nominal)
            self.v_recta_ep *= k
            self.v_curva_ep *= k
            self.v_max = V_MAX * rp.uniform(0.85, 1.15)   # actuator gains differ from the measured ones
            self.w_max = W_MAX * rp.uniform(0.80, 1.20)
            self.w_bias = rp.uniform(-0.04, 0.04)         # rad/s: left/right motor imbalance (drift on straights)
            self.motor_tau = rp.uniform(0.04, 0.22)
            self.blob_noise = rp.uniform(0.01, 0.05)
            self.sensor_delay = int(rp.integers(0, 2))    # camera latency of 0 or 1 control steps
            self.p_drop = rp.uniform(0.0, 0.06)           # random missed detections
            self.dt_jitter = 0.08                         # control period varies +-8 % from step to step
            self.r_noise_px = rp.uniform(0.12, 0.45)      # blob-size jitter (the data suggests ~0.25 px)
            self.k_cal = rp.uniform(0.88, 1.12)           # size->distance calibration error (fit uncertainty is ~8 %)
            self.quant_offs = BLOB_OFFS_PX + rp.uniform(-0.15, 0.15)
            # some episodes (p_saturacion): the reported blob area stops growing at some value (the recording suggests ~133)
            self.area_max = rp.uniform(125.0, 200.0) if rp.random() < self.p_saturacion else None
            self.sway_a = rp.uniform(0.0, 0.012)          # the leader does not follow the line perfectly
            self.sway_w = 2 * math.pi / rp.uniform(4.0, 10.0)
            self.sway_ph = rp.uniform(0.0, 2 * math.pi)
        self._hist = deque(maxlen=self.sensor_delay + 1)
        self.s = (self.path.length - 1.2) if s0 is None else float(s0) % self.path.length   # leader arc position
        self.v_lider = self.v_curva_ep if self.path.es_curva(self.s) else self.v_recta_ep
        self.parado_hasta = -1.0
        self.prox_chequeo = rng.uniform(0.0, c.parada_cada_s) if c.p_parada > 0 else math.inf

        if s0 is None:
            # default start (on the straight): exactly as before
            d0 = DIST_TARGET_M + 0.02  # initial distance to the leader
            fx, fy, fth = self.path.at(self.s - d0)
            lat = rng.uniform(-0.03, 0.03)
            self.x = fx - math.sin(fth) * lat
            self.y = fy + math.cos(fth) * lat
            lx0, ly0, _ = self.path.at(self.s)
            # start roughly facing the leader, so it is inside the camera view even on a corner
            self.th = math.atan2(ly0 - self.y, lx0 - self.x) + rng.uniform(-0.2, 0.2)
        else:
            # start anywhere on the lap (used for training/evaluation around the whole track). On a 0.31 m radius
            # corner the old start (0.32 m behind, aimed at the leader, up to 3 cm off the centre) cuts ~45 mm of
            # the 62 mm lane before the robot can react, which is unwinnable and only adds luck to the fitness.
            # Here: closer to the leader, almost on the centre line and heading along the lane (+-9 deg).
            d0 = DIST_TARGET_M - 0.06
            fx, fy, fth = self.path.at(self.s - d0)
            lat = rng.uniform(-0.015, 0.015)
            self.x = fx - math.sin(fth) * lat
            self.y = fy + math.cos(fth) * lat
            self.th = fth + rng.uniform(-0.15, 0.15)

        self.v = self.w = 0.0                  # executed forward / angular speed (after motor lag)
        self.prev_u = (0.0, 0.0)
        self.last_x = self.last_d = 0.0
        self.en_banda = 0
        self.lost_t = 0.0                      # time since the leader was last visible
        self.dt_k = self.dt
        self.t_det = -1e9                      # time of the last camera detection
        self.bx_det = self.dm_det = 0.0        # its (noisy) blob_x and camera distance in metres
        self._d_prev = None
        self.t = 0.0
        self.k = 0
        self.info = {}
        obs, self.info, _ = self._observe(primero=True)
        return obs

    # ------------------------------------------------------------------ #
    def _mover_lider(self, h):
        c = self.cfg
        if self.t >= self.prox_chequeo:                       # time for a stop check
            self.prox_chequeo += c.parada_cada_s
            if self.t >= self.parado_hasta and self.rng_lider.random() < c.p_parada:
                self.parado_hasta = self.t + self.rng_lider.uniform(*c.t_parada)
        if self.t < self.parado_hasta:
            objetivo = 0.0
        else:
            objetivo = self.v_curva_ep if self.path.es_curva(self.s) else self.v_recta_ep
        if c.aceleracion > 0:
            dv = c.aceleracion * h
            self.v_lider += min(dv, max(-dv, objetivo - self.v_lider))
        else:
            self.v_lider = objetivo
        self.s += self.v_lider * h

    def step(self, action):
        self.prev_action = self.action
        a0 = min(1.0, max(-1.0, float(action[0])))
        a1 = min(1.0, max(-1.0, float(action[1])))
        v_cmd = V_MIN + (a0 + 1.0) * 0.5 * (self.v_max - V_MIN)
        w_cmd = W_SIGN * a1 * self.w_max + self.w_bias
        dt_k = self.dt * (1.0 + self.rng_planta.uniform(-self.dt_jitter, self.dt_jitter)) if self.dt_jitter else self.dt
        self.dt_k = dt_k
        n = max(1, math.ceil(dt_k / SUBSTEP))
        h = dt_k / n
        k = h / (self.motor_tau + h)
        for _ in range(n):
            self.v += k * (v_cmd - self.v)
            self.w += k * (w_cmd - self.w)
            self.x += self.v * math.cos(self.th) * h
            self.y += self.v * math.sin(self.th) * h
            self.th += self.w * h
            self._mover_lider(h)
            self.t += h
        self.k += 1
        self.prev_u = (a0, a1)
        obs, info, done = self._observe()
        self.action = (a0, a1)
        info["prev_action"] = self.prev_action
        info["action"] = (a0, a1)
        info["v"], info["w"] = v_cmd, w_cmd
        self.info = info
        return obs, info, done

    # ------------------------------------------------------------------ #
    def leader_pose(self):
        """(x, y, heading) of the leader's centre. With randomisation it weaves a few mm around the line, as a
        real differential robot does, and its heading follows the weave."""
        x, y, th = self.path.at(self.s)
        if self.sway_a > 0.0:
            ph = self.sway_w * self.t + self.sway_ph
            lat = self.sway_a * math.sin(ph)
            v = max(self.v_lider, 1e-3)
            th += math.atan(self.sway_a * self.sway_w * math.cos(ph) / v)   # heading of a weaving trajectory
            x -= math.sin(th) * lat
            y += math.cos(th) * lat
        return x, y, th

    def ball_pose(self):
        """Centre of the ball (rear of the leader) and the leader's heading."""
        lx, ly, lth = self.leader_pose()
        return lx - BALL_BACK_M * math.cos(lth), ly - BALL_BACK_M * math.sin(lth), lth

    @staticmethod
    def project_ball(cx, cy, cth, bx, by, lth):
        """View of the ball from a camera at (cx, cy) heading cth.
        Returns (rho, u_px, r_eff_px, u_visible_px): distance, image column of the centre, equivalent radius of the
        part of the blob that is inside the image, and the column of the visible part's centre. None if not seen."""
        dx, dy = bx - cx, by - cy
        rho = math.hypot(dx, dy)
        if rho < 0.05:
            return None
        bearing = _wrap(math.atan2(dy, dx) - cth)
        if abs(bearing) >= math.radians(85.0):
            return None
        u = IMG_W_PX / 2.0 - F_PX * math.tan(bearing)
        r = BLOB_C_MM_PX / (rho * 1000.0)                       # ideal radius in pixels
        frac = _frac_inside(u, r, float(IMG_W_PX))
        if BLOB_FORMA == "disco":                               # flat disc, normal pointing backwards from the leader
            ang = _wrap(math.atan2(-dy, -dx) - (lth + math.pi)) # angle between the disc normal and the line of sight
            frac *= max(0.05, abs(math.cos(ang)))
        if frac <= 0.0:
            return None
        lo, hi = max(0.0, u - r), min(float(IMG_W_PX), u + r)
        return rho, u, r * math.sqrt(frac), 0.5 * (lo + hi)

    def _observe(self, primero=False):
        lx, ly, _ = self.leader_pose()
        bx_w, by_w, lth = self.ball_pose()
        d = math.hypot(lx - self.x, ly - self.y)               # ground truth centre-to-centre (collisions)
        cierre = 0.0 if self._d_prev is None else (self._d_prev - d) / max(self.dt_k, 1e-6)   # closing speed (m/s)
        self._d_prev = d
        cam_x = self.x + CAM_FWD_M * math.cos(self.th)
        cam_y = self.y + CAM_FWD_M * math.sin(self.th)
        rho_true = math.hypot(bx_w - cam_x, by_w - cam_y)      # ground truth camera -> ball distance
        self._hist.append((cam_x, cam_y, self.th, bx_w, by_w, lth))
        hx, hy, hth, hbx, hby, hlth = self._hist[0]            # what the camera saw (older if it has latency)

        # camera: what the official simulator reports is a blob AREA that only takes discrete values
        proj = self.project_ball(hx, hy, hth, hbx, hby, hlth)
        self.blob_area = 0.0
        if proj is not None and proj[0] <= CAM_RANGE_M:
            # the radius jitter is drawn every frame so that its correlation in time does not depend on dropouts
            self._r_err = RADIUS_NOISE_AR * self._r_err + self.rng.normal(0.0, self.r_noise_px * math.sqrt(1 - RADIUS_NOISE_AR ** 2))
            rho, u_c, r_eff, u_vis = proj
            n = round((r_eff + self._r_err - self.quant_offs) / BLOB_STEP_PX)
            if n >= BLOB_N_MIN and not (self.p_drop > 0 and self.rng_planta.random() < self.p_drop):
                area = round(math.pi * (BLOB_STEP_PX * n + self.quant_offs) ** 2)
                if self.area_max is not None:
                    area = min(area, self.area_max)
                self.blob_area = float(area)
                u_meas = round(u_vis + self.rng.normal(0.0, self.blob_noise * IMG_W_PX / 2.0))
                self.bx_det = min(1.0, max(-1.0, (u_meas - IMG_W_PX / 2.0) / (IMG_W_PX / 2.0)))
                self.dm_det = size_to_dist_m(area, self.k_cal)      # <- the controller's distance, from the size
                self.t_det = self.t
        # ...and stays "visible" (with stale values) for BLOB_VIGENCIA_S, like the real env
        visible = (self.t - self.t_det) <= BLOB_VIGENCIA_S + 1e-9

        if visible:
            bx = self.bx_det
            dn = min(1.0, max(-1.0, (self.dm_det - DIST_TARGET_M) / DIST_TARGET_M))
            self.last_x, self.last_d = bx, dn
            self.lost_t = 0.0
            # same definition as RoboboSeguimientoEnv: camera distance inside the target band
            if BANDA_OBJETIVO_M[0] <= self.dm_det <= BANDA_OBJETIVO_M[1]:
                self.en_banda += 1
            else:
                self.en_banda = 0
        else:
            bx = dn = 0.0
            if not primero:
                self.lost_t += self.dt_k
            self.en_banda = 0

        desvio = self.path.distance_to(self.x, self.y)
        fuera = desvio > self.lane_half_m                      # the follower's centre left the green lane
        choque = d < DIST_CHOQUE_M
        perdido = self.lost_t >= LOST_TIMEOUT_S - 1e-9
        done = choque or perdido or self.k >= self.max_steps

        info = {
            "blob_x": bx,
            "dist": dn,
            "visible": visible,
            "choque": choque,
            "perdido": perdido and not choque,
            "desvio_m": desvio,                    # distance from the follower to the lane centre line (m)
            "fuera_camino": fuera and not choque,
            "en_banda_consecutivos": self.en_banda,
            "pose_sim": (self.x, self.y, self.th),
            "distancia_real_mm": d * 1000.0,          # centre-to-centre
            "dist_camara_mm": rho_true * 1000.0,      # camera -> ball (what the size-based estimate tries to measure)
            "dist_estimada_mm": self.dm_det * 1000.0 if visible else 0.0,
            "blob_area_px": self.blob_area if visible else 0.0,
            "blob_diam_px": 2.0 * math.sqrt(self.blob_area / math.pi) if visible else 0.0,
            "v_lider": self.v_lider,
            "v_seguidor": self.v,
            "cierre_m_s": cierre,
            "lider_parado": self.t < self.parado_hasta,
        }
        obs = [bx, dn, 1.0 if visible else 0.0, self.last_x, self.last_d,
               self.prev_u[0], self.prev_u[1]]
        return obs, info, done


# --------------------------------------------------------------------------- #
# pygame viewer (imported lazily, never used by the training workers)
# --------------------------------------------------------------------------- #
class Viewer:
    GREEN = (0, 255, 0)

    def __init__(self, track_img=None, robot_img=None, title="follow_sim"):
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((WORLD_W_PX, WORLD_H_PX))
        self.font = pygame.font.SysFont(None, 20)
        self.bg = self._make_background(track_img)
        self.robot = self._load_robot(robot_img)
        self.trail = deque(maxlen=400)

    def _make_background(self, track_img):
        pg = self.pg
        if track_img:
            try:
                return pg.transform.smoothscale(pg.image.load(track_img).convert(),
                                                (WORLD_W_PX, WORLD_H_PX))
            except Exception as e:                       # noqa: BLE001
                print(f"[viewer] could not load {track_img}: {e}; drawing the lane instead")
        bg = pg.Surface((WORLD_W_PX, WORLD_H_PX))
        bg.fill((255, 255, 255))
        pts = [tuple(map(int, p)) for p in PATH.pts_px]
        pg.draw.lines(bg, self.GREEN, True, pts, LANE_WIDTH_PX)
        for p in pts:
            pg.draw.circle(bg, self.GREEN, p, LANE_WIDTH_PX // 2)
        return bg

    def _load_robot(self, robot_img):
        pg = self.pg
        size = (int(ROBOT_WID_M * PX_PER_M), int(ROBOT_LEN_M * PX_PER_M))
        if robot_img:
            try:
                return pg.transform.smoothscale(pg.image.load(robot_img).convert_alpha(), size)
            except Exception as e:                       # noqa: BLE001
                print(f"[viewer] could not load {robot_img}: {e}; drawing a box instead")
        s = pg.Surface(size, pg.SRCALPHA)
        pg.draw.rect(s, (60, 60, 60), s.get_rect(), border_radius=4)
        pg.draw.polygon(s, (230, 230, 230), [(size[0] // 2, 2), (4, 14), (size[0] - 4, 14)])
        return s

    def _px(self, x, y):
        return int(x * PX_PER_M), int(WORLD_H_PX - y * PX_PER_M)

    def _blit_robot(self, x, y, th, tag_color):
        pg = self.pg
        img = pg.transform.rotate(self.robot, math.degrees(th) - 90.0)
        c = self._px(x, y)
        self.screen.blit(img, img.get_rect(center=c))
        pg.draw.circle(self.screen, tag_color, c, 4)

    def draw(self, env, info, extra=""):
        pg = self.pg
        self.screen.blit(self.bg, (0, 0))
        # leader's path (lane centre line)
        pg.draw.lines(self.screen, (0, 0, 0), True,
                      [tuple(map(int, p)) for p in env.path.pts_px], 1)
        # camera field of view
        fx, fy, fth = env.x, env.y, env.th
        a = [self._px(fx + CAM_RANGE_M * math.cos(fth + s * HALF_FOV),
                      fy + CAM_RANGE_M * math.sin(fth + s * HALF_FOV)) for s in (-1, 1)]
        pg.draw.lines(self.screen, (90, 90, 255), False, [a[0], self._px(fx, fy), a[1]], 1)
        # trail + robots
        self.trail.append(self._px(fx, fy))
        if len(self.trail) > 1:
            pg.draw.lines(self.screen, (0, 0, 200), False, list(self.trail), 1)
        lx, ly, lth = env.leader_pose()
        self._blit_robot(lx, ly, lth, (255, 0, 0) if info["lider_parado"] else (255, 140, 0))
        bxw, byw, _ = env.ball_pose()                                   # the leader's green ball (rear)
        pg.draw.circle(self.screen, (0, 160, 0), self._px(bxw, byw), max(2, int(BALL_RADIUS_M * PX_PER_M)))
        near_edge = info["desvio_m"] > 0.75 * env.lane_half_m     # tag turns red close to the lane edge
        self._blit_robot(fx, fy, fth, (255, 0, 0) if near_edge else (0, 0, 255))
        # HUD
        lines = [f"t={env.t:5.1f}s  dt={env.dt}",
                 f"d={info['distancia_real_mm']:.0f}mm  x={info['blob_x']:+.2f}  "
                 f"vis={int(info['visible'])}  band={info['en_banda_consecutivos']}  "
                 f"dev={info['desvio_m'] * 1000:.0f}/{env.lane_half_m * 1000:.0f}mm",
                 f"leader {info['v_lider']:.3f} m/s{'  STOPPED' if info['lider_parado'] else ''}   {extra}"]
        lines.append(f"blob {info['blob_diam_px']:.0f}px  d_est={info['dist_estimada_mm']:.0f}mm  "
                     f"d_cam={info['dist_camara_mm']:.0f}mm")
        for i, txt in enumerate(lines):
            self.screen.blit(self.font.render(txt, True, (0, 0, 0), (255, 255, 255)), (4, 4 + 16 * i))
        self._draw_camera_view(info)
        pg.display.flip()

    def _draw_camera_view(self, info, scale=1.2):
        """Inset with what the follower's camera sees: the blob, at its column and with its pixel size."""
        pg = self.pg
        w, h = int(IMG_W_PX * scale), int(IMG_W_PX * 0.75 * scale)
        x0, y0 = 4, WORLD_H_PX - h - 4
        pg.draw.rect(self.screen, (235, 235, 235), (x0, y0, w, h))
        if info["visible"] and info["blob_diam_px"] > 0:
            cx = x0 + (info["blob_x"] + 1.0) * 0.5 * w
            pg.draw.circle(self.screen, (0, 200, 0), (int(cx), y0 + h // 2), max(1, int(info["blob_diam_px"] * scale / 2)))
        pg.draw.line(self.screen, (160, 160, 160), (x0 + w // 2, y0), (x0 + w // 2, y0 + h))
        pg.draw.rect(self.screen, (0, 0, 0), (x0, y0, w, h), 1)

    def close(self):
        self.pg.quit()


def watch(policy, dt=PASO_S, leader=None, margen_carril_m=0.0, seed=0, speed=1.0, max_time=60.0,
          track_img=None, robot_img=None, modo=1, aleatorizar=False):
    """Real-time view. Keys: SPACE pause, R reset, UP/DOWN speed, ESC quit."""
    import pygame
    env = FollowEnv(dt=dt, max_time=max_time, leader=leader, margen_carril_m=margen_carril_m,
                    modo=modo, aleatorizar=aleatorizar)
    viewer = Viewer(track_img, robot_img)
    clock = pygame.time.Clock()
    obs = env.reset(seed)
    viewer.draw(env, env.info)
    paused, run = False, True
    while run:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                run = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    seed += 1
                    obs = env.reset(seed)
                    viewer.trail.clear()
                elif e.key == pygame.K_UP:
                    speed *= 1.5
                elif e.key == pygame.K_DOWN:
                    speed /= 1.5
        if paused:
            clock.tick(30)
            continue
        obs, info, done = env.step(policy(obs))
        viewer.draw(env, info, f"speed x{speed:.2f}")
        clock.tick(speed / dt)
        if done:
            why = ("choque" if info["choque"] else "perdido" if info["perdido"]
                   else "fuera del carril" if info["fuera_camino"] else "tiempo agotado")
            print(f"episode ended ({why}) at t={env.t:.1f}s")
            seed += 1
            obs = env.reset(seed)
            viewer.trail.clear()
    viewer.close()


# --------------------------------------------------------------------------- #
# Hand-written P controller, only for sanity checks
# --------------------------------------------------------------------------- #
def p_controller(obs):
    """a0 sets the forward speed from the distance error, a1 turns towards the blob
    (blob_x > 0 = leader on the right -> negative angular speed)."""
    bx, dn, vis, last_x = obs[0], obs[1], obs[2], obs[3]
    clip = lambda v: max(-1.0, min(1.0, v))
    if not vis:
        return [-1.0, clip(-3.0 * last_x)]          # stop and look towards where it was last seen
    return [clip(3.0 * dn), clip(-3.0 * bx)]


if __name__ == "__main__":
    watch(p_controller)
