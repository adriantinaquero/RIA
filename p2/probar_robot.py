"""
probar_robot.py - measures how RoboboSim answers to the (a0, a1) actions, so follow_sim.py can be
made to agree with it. Run it with the simulator and the bridge up (same setup as p1_environment).

    python probar_robot.py

For each test it prints the real step period, the forward speed and the yaw rate, next to what
follow_sim.py assumes. LOOK AT THE ROBOT in the simulator during the second test:

    a = (-1, +1)  ->  v = 0, w = +W_MAX.   In follow_sim.py that is a turn to the LEFT (counter-clockwise).
    If the robot visibly turns RIGHT, set W_SIGN = -1.0 in p1_environment.py (genomes trained with
    the current follow_sim.py then work unchanged) or in follow_sim.py (and retrain).

This script was written without access to the simulator or ROS, so it has not been run.
"""
import math
import time

import numpy as np

from p1_environment import PASO_S, V_MAX, W_MAX, RoboboSeguimientoEnv

PASOS = 15


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def prueba(env, accion, nombre):
    env.reset()
    env.step(np.array([-1.0, 0.0], dtype=np.float32))   # first step only makes sure the pose is received
    if env._pose is None:
        print(f"{nombre}: no pose received (is the sim module running?)")
        return
    p0, t0 = env._pose, time.time()
    for _ in range(PASOS):
        env.step(np.array(accion, dtype=np.float32))
    p1, t1 = env._pose, time.time()
    env.stop_robot()

    dt_real = (t1 - t0) / PASOS
    dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    dyaw = wrap_deg(p1[2] - p0[2])
    v = (V_MAX * (accion[0] + 1) / 2)
    print(f"\n{nombre}  action = {tuple(accion)}")
    print(f"  step period : {dt_real:.3f} s   (the simulator uses dt = {PASO_S} s)")
    print(f"  forward     : {dist / (t1 - t0):.3f} m/s measured   vs  {v:.3f} m/s commanded")
    print(f"  yaw         : {dyaw / (t1 - t0):+.1f} deg/s measured   vs  {math.degrees(accion[1] * W_MAX):+.1f} deg/s commanded"
          f"   (rotation.y increased = {'yes' if dyaw > 0 else 'no'})")


def main():
    env = RoboboSeguimientoEnv(pasos_max=1000)
    try:
        prueba(env, (+1.0, 0.0), "1) straight ahead at V_MAX")
        prueba(env, (-1.0, +1.0), "2) turn in place, w = +W_MAX  (WATCH: left or right?)")
        prueba(env, (-1.0, -1.0), "3) turn in place, w = -W_MAX")
    finally:
        env.close()


if __name__ == "__main__":
    main()
