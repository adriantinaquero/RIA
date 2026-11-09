"""
probar_ruedas.py - what does a MoveWheels speed in [-100, 100] mean in mm/s on RoboboSim?

The Robobo documentation only calls it a "speed factor" and never gives a unit, so this script
measures it with the same machinery as p1_environment.py: the move_wheels service to drive and the
robot_location topic (position in mm, rotation.y) to measure. Run it with RoboboSim and the bridge up:

    python probar_ruedas.py                                  # sweep 10 25 50 75 100, straight + in place
    python probar_ruedas.py --velocidades 5 10 20 40 80 100 --sentido adelante
    python probar_ruedas.py --sin-giro                       # only the straight-line sweep

For every speed s it runs
  * a straight run, move_wheels(s, s): mm/s from a line fit of travelled distance vs time
    (the first --descartar seconds are dropped so the acceleration phase does not bias the result)
  * an in-place turn, move_wheels(+s, -s): deg/s from a line fit of the accumulated yaw,
    and the wheel base this implies, L = 2 * v / omega (a cross-check of both measurements)
and at the end it fits  mm/s = k * speed  through the origin, reports how linear it is, and converts
the 221 mm/s you measured through the Twist (v = 0.15 m/s) command into the equivalent wheel speed.

CAUTION  The scene is reset before every run, which puts the follower about 0.3 m behind the leader.
By default the straight runs go BACKWARDS (away from the leader) and stop after --max-despl-mm.
Use --sentido adelante if there is a wall behind the follower, and make sure the area is free.
Times are wall-clock seconds; if RoboboSim does not run in real time, scale the results accordingly.
This script was written without access to RoboboSim or ROS: only its fitting logic has been tested
(against a simulated robot with known speeds). Positions are assumed to be in mm and rotation.y in degrees.
"""
import argparse
import csv
import math
import time

import numpy as np


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def medir(env, spin, r, l, duracion, max_despl_mm):
    """Drive move_wheels(r, l) and return [(t_s, distance_from_start_mm, accumulated_yaw_deg), ...]."""
    env.reset()
    t0 = time.time()
    while env._pose is None and time.time() - t0 < 3.0:
        spin(env)
    if env._pose is None:
        raise RuntimeError("no robot_location received: is the sim module running?")
    for _ in range(10):                       # let the pose settle after the scene reset
        spin(env)

    p0 = env._pose
    ultimo, yaw_prev, yaw_acum = p0, p0[2], 0.0
    muestras = []
    if not env.move_wheels(r, l):
        raise RuntimeError("the move_wheels service call failed")
    t_ini = time.time()
    while time.time() - t_ini < duracion:
        spin(env)
        p = env._pose
        if p is ultimo:                       # no new message since the last look
            continue
        ultimo = p
        yaw_acum += wrap_deg(p[2] - yaw_prev)
        yaw_prev = p[2]
        d = math.hypot(p[0] - p0[0], p[1] - p0[1])
        muestras.append((time.time() - t_ini, d, yaw_acum))
        if d > max_despl_mm:
            break
    env.stop_robot()
    for _ in range(10):
        spin(env)
    return muestras


def ajustar(muestras, descartar):
    """Slopes (distance mm/s, yaw deg/s) of the samples after the first `descartar` seconds."""
    m = [s for s in muestras if s[0] >= descartar]
    if len(m) < 4:
        return None, None, len(m)
    t = np.array([s[0] for s in m])
    v = float(np.polyfit(t, [s[1] for s in m], 1)[0])
    w = float(np.polyfit(t, [s[2] for s in m], 1)[0])
    return v, w, len(m)


def barrido(env, spin, velocidades, sentido, duracion, max_despl, descartar, giro):
    signo = -1.0 if sentido == "atras" else 1.0
    filas = []
    for s in velocidades:
        m = medir(env, spin, signo * s, signo * s, duracion, max_despl)
        v, _, n1 = ajustar(m, descartar)
        w = n2 = base = None
        if giro:
            m2 = medir(env, spin, s, -s, duracion, max_despl)
            _, w, n2 = ajustar(m2, descartar)
            if v is not None and w:
                base = 2.0 * abs(v) / math.radians(abs(w))
        filas.append(dict(velocidad=s, mm_s=v, giro_deg_s=w, base_mm=base, muestras=n1, muestras_giro=n2))
        print("  speed {:>4}: {}   {}   {}".format(
            s,
            "{:7.1f} mm/s ({:.2f} mm/s per unit, {} samples)".format(abs(v), abs(v) / s, n1) if v is not None
            else "straight run: too few samples ({}), lower the speed or raise --max-despl-mm".format(n1),
            "{:+7.1f} deg/s ({} samples)".format(w, n2) if w is not None else "",
            "implied wheel base {:.0f} mm".format(base) if base else ""))
    return filas


def informe(filas, ref_mm_s):
    ok = [f for f in filas if f["mm_s"] is not None]
    if len(ok) < 2:
        print("\nnot enough valid runs to fit a speed scale")
        return None
    s = np.array([f["velocidad"] for f in ok], dtype=float)
    v = np.array([abs(f["mm_s"]) for f in ok])
    k = float((s * v).sum() / (s * s).sum())                      # least squares through the origin
    resid = v - k * s
    desv = float(np.max(np.abs(resid) / np.maximum(v, 1e-9)) * 100)
    print("\nFit  mm/s = k * speed   ->  k = {:.3f} mm/s per unit   (speed 100 = {:.0f} mm/s if linear)".format(k, 100 * k))
    print("Largest deviation from a straight line through the origin: {:.1f} % of the measured value".format(desv))
    if desv > 10:
        print("  -> not linear (saturation or a dead zone): do not extrapolate, use the measured table")
    if ref_mm_s:
        print("The {:.0f} mm/s you measured with the Twist command v = 0.15 m/s equals a wheel speed of about {:.0f}".format(
            ref_mm_s, ref_mm_s / k))
    return k


def main():
    import rclpy
    from p1_environment import RoboboSeguimientoEnv

    p = argparse.ArgumentParser(description="Measure what a MoveWheels speed value means in mm/s")
    p.add_argument("--velocidades", type=float, nargs="+", default=[10, 25, 50, 75, 100])
    p.add_argument("--duracion", type=float, default=3.0, help="seconds per run")
    p.add_argument("--max-despl-mm", type=float, default=400.0, help="stop a straight run after this distance")
    p.add_argument("--descartar", type=float, default=0.5, help="seconds dropped at the start of every run")
    p.add_argument("--sentido", choices=["atras", "adelante"], default="atras")
    p.add_argument("--sin-giro", action="store_true", help="skip the in-place turns")
    p.add_argument("--ref-mm-s", type=float, default=221.0, help="speed measured with the Twist command (0 = skip)")
    p.add_argument("--csv", default="ruedas_medidas.csv")
    a = p.parse_args()

    env = RoboboSeguimientoEnv(pasos_max=100000)
    spin = lambda e: rclpy.spin_once(e.nodo, timeout_sec=0.02)
    try:
        print("Driving {} for {:.1f} s per speed (or {:.0f} mm); the first {:.1f} s of each run are dropped".format(
            a.sentido, a.duracion, a.max_despl_mm, a.descartar))
        filas = barrido(env, spin, a.velocidades, a.sentido, a.duracion, a.max_despl_mm,
                        a.descartar, not a.sin_giro)
    finally:
        env.close()

    informe(filas, a.ref_mm_s)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print("table saved to", a.csv)


if __name__ == "__main__":
    main()