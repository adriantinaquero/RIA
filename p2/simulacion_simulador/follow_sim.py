"""
follow_sim.py - lightweight 2D simulator of a differential-drive robot that has to
follow a leader robot around a closed track, seen through a front camera.

* The physics/sensor core only needs the standard library + numpy, so it is safe and
  fast to use inside multiprocessing workers (pygame is imported lazily by the viewer).
* The control period `dt` is arbitrary; physics are integrated with sub-steps of <= 10 ms.
* `FollowEnv.step()` returns the same `info` dict your real simulator produces.

Run `python follow_sim.py` to watch a hand-written P controller (sanity check).
"""
import math
from collections import deque

import numpy as np

# --------------------------------------------------------------------------- #
# World (reference image is 450 x 500 px; PX_PER_M sets the real-world scale)
# --------------------------------------------------------------------------- #
PX_PER_M = 200.0
WORLD_W_PX, WORLD_H_PX = 450, 500
BAND_PX = 35                                   # drawn width of the green / red bands

# Centre line of the green band (px, y down) and the inset of the red (inner) band.
GREEN_RECT = dict(left=28, right=415, top=28, bottom=478, radius=60)
RED_INSET_PX = 37

# --------------------------------------------------------------------------- #
# Robot / sensor parameters
# --------------------------------------------------------------------------- #
ROBOT_LEN_M = 0.16
ROBOT_WID_M = 0.10

# Actions exactly as in RoboboSeguimientoEnv.step():
#   v = V_MIN + (a0 + 1) / 2 * (V_MAX - V_MIN)      (forward speed, never negative)
#   w = a1 * W_MAX                                   (angular speed, + = turn left)
V_MIN = 0.0             # m/s
V_MAX = 0.15            # m/s
W_MAX = 0.35            # rad/s
MOTOR_TAU = 0.08        # first-order lag of the executed v / w (s)
SUBSTEP = 0.01          # max integration sub-step (s)

HALF_FOV = math.radians(35.0)
CAM_RANGE_M = 1.2
BLOB_NOISE = 0.02       # std of blob_x, normalised units
DIST_NOISE = 0.02       # std of the camera distance estimate, as a fraction of DIST_TARGET_M
BLOB_VIGENCIA_S = 0.4   # a detection still counts as "visible" (stale values) for this long

# Leader forward speed (m/s). Must stay below the follower's V_MAX (0.15) or it can never keep up.
LEADER_V_RANGE = (0.05, 0.11)

# --------------------------------------------------------------------------- #
# Task constants (same meaning as in your fitness function)
# --------------------------------------------------------------------------- #
DIST_CHOQUE_M = 0.12
DIST_PERDIDO_M = 0.9    # not used to end episodes (the real env only counts unseen steps)
DIST_TARGET_M = 0.3
BANDA_OBJETIVO_M = (0.2, 0.45)
NORM_BANDA_OBJETIVO = tuple((b - DIST_TARGET_M) / DIST_TARGET_M for b in BANDA_OBJETIVO_M)
X_TOLERANCIA = 0.15
PASO_S = 0.2                                  # control period of the real env (default dt here)
PASOS_PERDIDO_MAX = 4                         # unseen steps before the episode counts as lost
LOST_TIMEOUT_S = PASOS_PERDIDO_MAX * PASO_S   # = 0.8 s, kept in seconds so any dt behaves alike

# Staying on the chosen path (distance from the follower to the leader's path centre line)
DEV_TOL_M = 0.0875      # dead zone = half the drawn band width (BAND_PX / 2 / PX_PER_M)
DEV_SOFT_M = 0.20       # soft penalty is at its maximum from here on
DEV_MAX_M = 0.30        # farther than this the episode ends ("fuera_camino")

N_INPUTS = 7
N_OUTPUTS = 2


# --------------------------------------------------------------------------- #
# Leader paths
# --------------------------------------------------------------------------- #
def _rounded_rect_px(left, right, top, bottom, r, n=24):
    """Closed polyline (px, y down). Travels up the right edge first (counter-clockwise)."""
    def arc(cx, cy, a0, a1):
        a = np.radians(np.linspace(a0, a1, n))
        return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])

    pts = np.vstack([
        arc(right - r, top + r, 0, -90),
        arc(left + r, top + r, -90, -180),
        arc(left + r, bottom - r, 180, 90),
        arc(right - r, bottom - r, 90, 0),
    ])
    return np.vstack([pts, pts[:1]])


class Path:
    """Closed path in world metres (y up) parametrised by arc length."""

    def __init__(self, pts_px):
        p = np.asarray(pts_px, dtype=float)
        self.pts_px = p
        self.pts = np.column_stack([p[:, 0] / PX_PER_M, (WORLD_H_PX - p[:, 1]) / PX_PER_M])
        seg = np.hypot(*np.diff(self.pts, axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])
        self._a = self.pts[:-1]
        self._ab = self.pts[1:] - self.pts[:-1]
        self._ab2 = np.maximum((self._ab ** 2).sum(axis=1), 1e-12)

    def distance_to(self, x, y):
        """Shortest distance (m) from the point (x, y) to the path polyline."""
        p = np.array([x, y])
        ap = p - self._a
        t = np.clip((ap * self._ab).sum(axis=1) / self._ab2, 0.0, 1.0)
        d = p - (self._a + self._ab * t[:, None])
        return float(np.sqrt((d ** 2).sum(axis=1).min()))

    def at(self, s):
        s = s % self.length
        x = np.interp(s, self.s, self.pts[:, 0])
        y = np.interp(s, self.s, self.pts[:, 1])
        i = min(int(np.searchsorted(self.s, s, side="right")) - 1, len(self.s) - 2)
        th = math.atan2(self.pts[i + 1, 1] - self.pts[i, 1], self.pts[i + 1, 0] - self.pts[i, 0])
        return float(x), float(y), th


def _make_paths():
    g = GREEN_RECT
    i = RED_INSET_PX
    green = _rounded_rect_px(g["left"], g["right"], g["top"], g["bottom"], g["radius"])
    red = _rounded_rect_px(g["left"] + i, g["right"] - i, g["top"] + i, g["bottom"] - i,
                           max(g["radius"] - i, 15))
    return {"verde": Path(green), "roja": Path(red)}


PATHS = _make_paths()


# --------------------------------------------------------------------------- #
# Leader speed profiles
# --------------------------------------------------------------------------- #
def make_profile(kind, v0, rng):
    if kind == "const":
        return lambda t: v0
    if kind == "stop_go":
        period = rng.uniform(6.0, 10.0)
        stop = rng.uniform(1.0, 2.5)
        return lambda t: 0.0 if (t % period) > period - stop else v0
    if kind == "sine":
        T = rng.uniform(8.0, 15.0)
        ph = rng.uniform(0.0, 2 * math.pi)
        return lambda t: v0 * (1.0 + 0.3 * math.sin(2 * math.pi * t / T + ph))
    raise ValueError(f"unknown profile {kind!r}")


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
    The episode is lost after LOST_TIMEOUT_S (0.8 s) without seeing the leader.
    """

    def __init__(self, dt=PASO_S, max_time=30.0, path="verde", profile="const", v0=None):
        self.dt = float(dt)
        self.max_time = float(max_time)
        self.max_steps = max(1, int(round(max_time / dt)))
        self.path_name, self.profile_name, self.v0_fixed = path, profile, v0
        self.reset(0)

    # ------------------------------------------------------------------ #
    def reset(self, seed=None):
        rng = self.rng = np.random.default_rng(seed)
        self.path = PATHS[self.path_name]
        self.v0 = self.v0_fixed if self.v0_fixed is not None else rng.uniform(*LEADER_V_RANGE)
        self.vfun = make_profile(self.profile_name, self.v0, rng)

        self.s = rng.uniform(0.0, self.path.length)         # leader arc position
        d0 = DIST_TARGET_M + rng.uniform(-0.05, 0.10)
        fx, fy, fth = self.path.at(self.s - d0)
        lat = rng.uniform(-0.03, 0.03)
        self.x = fx - math.sin(fth) * lat
        self.y = fy + math.cos(fth) * lat
        lx0, ly0, _ = self.path.at(self.s)
        # start roughly facing the leader, so it is inside the camera view even on a corner
        self.th = math.atan2(ly0 - self.y, lx0 - self.x) + rng.uniform(-0.2, 0.2)

        self.v = self.w = 0.0                  # executed forward / angular speed (after motor lag)
        self.prev_u = (0.0, 0.0)
        self.last_x = self.last_d = 0.0
        self.en_banda = 0
        self.lost_t = 0.0                      # time since the leader was last visible
        self.t_det = -1e9                      # time of the last camera detection
        self.bx_det = self.dm_det = 0.0        # its (noisy) blob_x and camera distance in metres
        self.t = 0.0
        self.k = 0
        self.info = {}
        obs, self.info, _ = self._observe(primero=True)
        return obs

    # ------------------------------------------------------------------ #
    def step(self, action):
        a0 = min(1.0, max(-1.0, float(action[0])))
        a1 = min(1.0, max(-1.0, float(action[1])))
        v_cmd = V_MIN + (a0 + 1.0) * 0.5 * (V_MAX - V_MIN)
        w_cmd = a1 * W_MAX
        n = max(1, math.ceil(self.dt / SUBSTEP))
        h = self.dt / n
        k = h / (MOTOR_TAU + h)
        for _ in range(n):
            self.v += k * (v_cmd - self.v)
            self.w += k * (w_cmd - self.w)
            self.x += self.v * math.cos(self.th) * h
            self.y += self.v * math.sin(self.th) * h
            self.th += self.w * h
            self.s += self.vfun(self.t) * h
            self.t += h
        self.k += 1
        self.prev_u = (a0, a1)
        obs, info, done = self._observe()
        info["action"] = (a0, a1)
        info["v"], info["w"] = v_cmd, w_cmd
        self.info = info
        return obs, info, done

    # ------------------------------------------------------------------ #
    def leader_pose(self):
        return self.path.at(self.s)

    def _observe(self, primero=False):
        lx, ly, _ = self.path.at(self.s)
        dx, dy = lx - self.x, ly - self.y
        d = math.hypot(dx, dy)
        bearing = _wrap(math.atan2(dy, dx) - self.th)          # + = leader to the left

        # camera: a detection happens when the leader is inside the field of view / range
        if abs(bearing) <= HALF_FOV and d <= CAM_RANGE_M:
            self.bx_det = min(1.0, max(-1.0, -bearing / HALF_FOV + self.rng.normal(0.0, BLOB_NOISE)))
            self.dm_det = d + self.rng.normal(0.0, DIST_NOISE * DIST_TARGET_M)
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
                self.lost_t += self.dt
            self.en_banda = 0

        desvio = self.path.distance_to(self.x, self.y)
        fuera = desvio > DEV_MAX_M
        choque = d < DIST_CHOQUE_M
        perdido = self.lost_t >= LOST_TIMEOUT_S - 1e-9
        done = choque or perdido or fuera or self.k >= self.max_steps

        info = {
            "blob_x": bx,
            "dist": dn,
            "visible": visible,
            "choque": choque,
            "perdido": perdido and not choque,
            "desvio_m": desvio,                    # distance to the chosen path (m)
            "fuera_camino": fuera and not choque,
            "en_banda_consecutivos": self.en_banda,
            "pose_sim": (self.x, self.y, self.th),
            "distancia_real_mm": d * 1000.0,
        }
        obs = [bx, dn, 1.0 if visible else 0.0, self.last_x, self.last_d,
               self.prev_u[0], self.prev_u[1]]
        return obs, info, done


# --------------------------------------------------------------------------- #
# pygame viewer (imported lazily, never used by the training workers)
# --------------------------------------------------------------------------- #
class Viewer:
    GREEN, RED = (0, 255, 0), (255, 0, 0)

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
                print(f"[viewer] could not load {track_img}: {e}; drawing the track instead")
        bg = pg.Surface((WORLD_W_PX, WORLD_H_PX))
        bg.fill((255, 255, 255))
        for name, col in (("verde", self.GREEN), ("roja", self.RED)):
            pts = [tuple(map(int, p)) for p in PATHS[name].pts_px]
            pg.draw.lines(bg, col, True, pts, BAND_PX)
            for p in pts:
                pg.draw.circle(bg, col, p, BAND_PX // 2)
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
        # active leader path
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
        self._blit_robot(lx, ly, lth, (255, 140, 0))
        off = info["desvio_m"] > DEV_TOL_M          # tag turns red when straying from the path
        self._blit_robot(fx, fy, fth, (255, 0, 0) if off else (0, 0, 255))
        # HUD
        lines = [f"t={env.t:5.1f}s  dt={env.dt}",
                 f"d={info['distancia_real_mm']:.0f}mm  x={info['blob_x']:+.2f}  "
                 f"vis={int(info['visible'])}  band={info['en_banda_consecutivos']}  "
                 f"dev={info['desvio_m'] * 1000:.0f}mm", extra]
        for i, txt in enumerate(lines):
            self.screen.blit(self.font.render(txt, True, (0, 0, 0), (255, 255, 255)), (4, 4 + 16 * i))
        pg.display.flip()

    def close(self):
        self.pg.quit()


def watch(policy, dt=0.05, path="verde", profile="const", seed=0, speed=1.0, max_time=60.0,
          track_img=None, robot_img=None):
    """Real-time view. Keys: SPACE pause, R reset, UP/DOWN speed, ESC quit."""
    import pygame
    env = FollowEnv(dt=dt, max_time=max_time, path=path, profile=profile)
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
        viewer.draw(env, info, f"speed x{speed:.2f}   leader v0={env.v0:.2f} m/s  {profile}/{path}")
        clock.tick(speed / dt)
        if done:
            why = ("choque" if info["choque"] else "perdido" if info["perdido"]
                   else "fuera del camino" if info["fuera_camino"] else "tiempo agotado")
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
    watch(p_controller, dt=PASO_S, profile="stop_go")
