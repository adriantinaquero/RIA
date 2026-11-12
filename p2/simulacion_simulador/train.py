"""
train.py - parallel NEAT training / evaluation / viewing for the follower robot.

Modes (same names as p2_entrenar.py, plus a few extras for the simulator):
    python train.py --entrenar --ruta-config config.txt --generaciones 100 --pasos-max 800
    python train.py --reanudar-entrenamiento --checkpoint 40 --pasos-faltantes 60
    python train.py --evaluar --ruta-modelo mejor_genoma.plk --episodios 5 --camino roja --perfil stop_go
    python train.py --ver --ruta-modelo mejor_genoma.plk          (pygame window)
    python train.py --comprobar      reference P controller through the same summary
    python train.py --aleatorio      random actions through the same summary

Logging written while training:
    evolucion.csv          one row per generation (see registro.py)
    genomas/gen_XXXX.pkl   best genome of every generation
    chk/<prefijo>-<gen>    NEAT checkpoint every generation (resume with --reanudar-entrenamiento)
"""
import argparse
import multiprocessing as mp
import os
import pickle
import random

import neat
import numpy as np

from follow_sim import (modos, DIST_TARGET_M, LANE_HALF_M, NORM_BANDA_OBJETIVO, X_TOLERANCIA,
                        PASO_S, PATH, FollowEnv, p_controller, watch)
from registro import RegistroGeneracion

# Preferred CAMERA distance (camera -> ball) for the REWARD, normalised: (d - 0.3)/0.3. A pursuing robot cuts the
# corners by an amount that grows with the square of its distance to the leader, and the lane is only 62 mm
# from the centre line. Following at 0.30 m uses all of that margin; at ~0.21 m it cuts far less. -0.30 -> 0.21 m,
# still inside the target band (0.20-0.45 m).
D_PREF = -0.10   # 0.27 m: a little inside the task's 0.30 m target (corners are cut less). If your sensor saturates
                 # (see --saturacion) the net cannot see closer than ~0.30 m: then use 0.0 (0.30 m)
SIGMA_CERCA, SIGMA_LEJOS, SIGMA_X = 0.20, 0.30, 0.20     # closer than preferred is punished harder than farther
W_DESVIO = 1.5        # penalty weight for the (normalised, squared) distance to the lane centre line
W_BORDE = 4.0         # extra penalty slope once past 80 % of the lane half width
W_SUAVE_W = 2.0       # quadratic penalty on the change of the steering command (anti-wobble)
W_SUAVE_W1 = 0.3      # linear penalty on the change of the steering command
W_SUAVE_V = 0.5       # quadratic penalty on the change of the speed command
W_CIERRE = 1.0        # penalty for closing in on the leader faster than ~2 cm/s (steady pace, no rushing)
W_GIRO_PARADO = 0.6   # penalty for steering while (almost) stopped with the leader already centred (no dithering)
W_ZIGZAG = 0.8        # penalty for flipping the steering sign from one step to the next
W_RUMBO = 0.3         # penalty for heading away from the lane direction (0.5 rad -> full penalty). 0 disables it


def _err_rumbo(info):
    """Heading of the follower minus the direction of the nearest lane segment (rad, ground truth)."""
    x, y, th = info["pose_sim"]
    ap = np.array([x, y]) - PATH._a
    t = np.clip((ap * PATH._ab).sum(axis=1) / PATH._ab2, 0.0, 1.0)
    d = ap - PATH._ab * t[:, None]
    i = int(np.argmin((d ** 2).sum(axis=1)))
    e = th - np.arctan2(PATH._ab[i, 1], PATH._ab[i, 0])
    return float((e + np.pi) % (2 * np.pi) - np.pi)


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #
def _r_dist(info):
    d_real = info["dist_camara_mm"] / 1000.0                  # camera -> ball, the quantity the real env measures
    d = (d_real - DIST_TARGET_M) / DIST_TARGET_M - D_PREF   # unclipped, from ground truth, centred on D_PREF
    s = SIGMA_CERCA if d < 0 else SIGMA_LEJOS
    return float(np.exp(-0.5 * (d / s) ** 2))

def r_paso(info):
    """Reward of one non-terminal step."""
    e_c = info["desvio_m"] / LANE_HALF_M                   # 0 = on the centre line, 1 = at the lane edge
    r_cam = -W_DESVIO * min(e_c, 1.5) ** 2 - W_BORDE * max(0.0, e_c - 0.8)
    if W_RUMBO > 0:
        r_cam -= W_RUMBO * min(1.0, (_err_rumbo(info) / 0.5) ** 2)
    d_a1 = abs(info["prev_action"][1] - info["action"][1])
    d_a0 = abs(info["prev_action"][0] - info["action"][0])
    r_suave = -min(1.0, W_SUAVE_W * d_a1 ** 2 + W_SUAVE_W1 * d_a1 + W_SUAVE_V * d_a0 ** 2)   # capped: never swamps the rest
    a1, p1 = info["action"][1], info["prev_action"][1]
    # steady behaviour: do not rush towards the leader, do not turn on the spot, do not zig-zag
    if info["distancia_real_mm"] < 450.0:
        r_suave -= W_CIERRE * min(1.0, max(0.0, (info["cierre_m_s"] - 0.02) / 0.06) ** 2)
    if info["v_seguidor"] < 0.03 and abs(info["blob_x"]) < 0.2:
        r_suave -= W_GIRO_PARADO * min(1.0, abs(a1))
    if a1 * p1 < 0:
        r_suave -= W_ZIGZAG * min(1.0, min(abs(a1), abs(p1)) / 0.3)

    if not info["visible"]:
        d_real = info["dist_camara_mm"] / 1000.0
        err = min(1.0, abs(d_real - DIST_TARGET_M) / DIST_TARGET_M)
        return -1.0 - 0.3 * err + r_cam + r_suave

    e_x = abs(info["blob_x"])
    r = _r_dist(info) * (0.3 + 0.7 * float(np.exp(-0.5 * (e_x / SIGMA_X) ** 2)))
    if e_x <= X_TOLERANCIA and NORM_BANDA_OBJETIVO[0] <= info["dist"] <= NORM_BANDA_OBJETIVO[1]:
        racha = min(info["en_banda_consecutivos"], 50) / 50.0
        r += 0.3 * (0.5 + 0.5 * racha)
    
    if info["fuera_camino"]:
        r -= 1

    if info["desvio_m"] <= 0.04:
        r += 0.5

    return r + r_cam + r_suave


def calidad(registro, pasos_eval):
    total, n = 0.0, len(registro)
    for info in registro:
        if info["choque"] or info["perdido"] or info["fuera_camino"]:
            total -= 1.0 * (pasos_eval - n + 1)            # charge the steps never played
            if info["choque"]:
                total -= 2.0
            break
        total += r_paso(info)
    return total / pasos_eval


# --------------------------------------------------------------------------- #
# Episodes and their statistics
# --------------------------------------------------------------------------- #
def stats_episodio(registro):
    n = len(registro)
    last = registro[-1]
    return {
        "pasos": n,
        "pct_banda": 100.0 * sum(1 for i in registro if i["en_banda_consecutivos"] >= 1) / n,
        "desvio_medio_m": float(np.mean([i["desvio_m"] for i in registro])),
        "desvio_max_m": float(max(i["desvio_m"] for i in registro)),
        "choque": bool(last["choque"]),
        "perdido": bool(last["perdido"]),
        "fuera": bool(last["fuera_camino"]),
    }


def desenlace(st):
    if st["choque"]:
        return "choque"
    if st["perdido"]:
        return "perdido"
    if st["fuera"]:
        return "fuera del camino"
    return "tiempo agotado"


def run_episode(politica, env, seed, detalle=False, s0=None):
    """politica: obs -> action (or None for random actions). Returns the list of info dicts."""
    obs = env.reset(seed, s0)
    registro = []
    while True:
        accion = np.random.uniform(-1, 1, 2) if politica is None else politica(obs)
        obs, info, done = env.step(accion)
        registro.append(info)
        if detalle:
            fin = ("  CHOQUE" if info["choque"] else "  PERDIDO" if info["perdido"]
                   else "  FUERA" if info["fuera_camino"] else "")
            r = r_paso(info) if not (info["choque"] or info["perdido"] or info["fuera_camino"]) else float("nan")
            print("paso {:3d}  vis={}  x={:+.3f}  dist={:+.3f}  d={:.3f}m  dev={:.3f}m  "
                  "v={:+.3f}  w={:+.3f}  r={:+.3f}{}".format(
                      env.k, int(info["visible"]), info["blob_x"], info["dist"],
                      info["distancia_real_mm"] / 1000.0, info["desvio_m"],
                      info["v"], info["w"], r, fin))
        if done:
            return registro


def _modo_ep(modo, i):
    """`modo` is an int or a tuple of ints; episode i uses modo[i % len(modo)]."""
    return modo if isinstance(modo, int) else modo[i % len(modo)]


def evaluate(genome, config, gen_seed, dt, n_iter, pasos_max, modo, aleat=True, p_sat=0.0):
    """0.6*mean + 0.4*worst over several scenarios; same seeds for every genome in a generation.
    Returns (fitness, stats) where stats summarises the behaviour over the episodes."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    scores, sts = [], []
    for i in range(n_iter):
        # ALWAYS the original start position. Episodes are long enough (pasos_max) for the leader to go
        # round the whole lap, so every episode contains all four corners.
        env = FollowEnv(dt=dt, max_time=pasos_max * dt, aleatorizar=aleat, modo=_modo_ep(modo, i), p_saturacion=p_sat)
        reg = run_episode(net.activate, env, gen_seed * 1000 + i)
        scores.append(calidad(reg, env.max_steps))
        sts.append(stats_episodio(reg))
    s = np.asarray(scores)
    stats = {
        "episodios": n_iter,
        "pasos": sum(st["pasos"] for st in sts),
        "pct_banda": float(np.mean([st["pct_banda"] for st in sts])),
        "desvio_medio_m": float(np.mean([st["desvio_medio_m"] for st in sts])),
        "choques": sum(st["choque"] for st in sts),
        "perdidos": sum(st["perdido"] for st in sts),
        "fuera": sum(st["fuera"] for st in sts),
    }
    # the worst scenario (usually the one with a corner) weighs a lot: a net that only survives the
    # straights must not out-rank one that also takes the curves
    return float(0.6 * s.mean() + 0.4 * s.min()), stats


def _job(args):
    return evaluate(*args)


def validate(genome, config, dt, pasos_max, modo=1, p_sat=0.0, seeds=(10_001, 10_002, 10_003, 10_004, 10_005, 10_006)):
    """Fixed test set, same for every candidate: the original start, half of the episodes on the nominal
    simulator and half with domain randomisation (the candidate must work on both)."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    scores = []
    for k, sd in enumerate(seeds):
        env = FollowEnv(dt=dt, max_time=pasos_max * dt, aleatorizar=(k % 2 == 1), modo=_modo_ep(modo, k), p_saturacion=p_sat)
        scores.append(calidad(run_episode(net.activate, env, sd * 100), env.max_steps))
    s = np.asarray(scores)
    return float(0.6 * s.mean() + 0.4 * s.min())


# --------------------------------------------------------------------------- #
# Parallel evaluator (a Pool so we can pass a per-generation seed to the workers)
# --------------------------------------------------------------------------- #
class PoolEvaluator:
    def __init__(self, workers, dt, n_iter, pasos_max, modo, detalle=False, aleat=True, p_sat=0.0):
        self.pool = mp.Pool(workers)
        self.dt, self.n_iter, self.pasos_max, self.modo, self.detalle, self.aleat = dt, n_iter, pasos_max, modo, detalle, aleat
        self.p_sat = p_sat
        self.contador = {"pasos": 0, "evaluaciones": 0}

    def __call__(self, genomas, config):
        gen_seed = random.randrange(1_000_000)
        jobs = [(g, config, gen_seed, self.dt, self.n_iter, self.pasos_max, self.modo, self.aleat, self.p_sat) for _, g in genomas]
        resultados = self.pool.map(_job, jobs, chunksize=1)
        pasos = 0
        for pos, ((_, g), (fit, st)) in enumerate(zip(genomas, resultados), 1):
            g.fitness, g.stats = fit, st
            pasos += st["pasos"]
            if self.detalle:
                print(f"Individuo {pos}: Fitness = {fit:.4f}  dev = {st['desvio_medio_m'] * 1000:.0f} mm  "
                      f"banda = {st['pct_banda']:.0f}%  choques/perdidos/fuera = "
                      f"{st['choques']}/{st['perdidos']}/{st['fuera']}")
        self.contador["pasos"] += pasos
        self.contador["evaluaciones"] += len(genomas) * self.n_iter
        print(f"Evaluaciones: {len(genomas) * self.n_iter}  pasos simulados: {pasos}  "
              f"(acumulado: {self.contador['evaluaciones']} evaluaciones, {self.contador['pasos']} pasos)")

    def close(self):
        self.pool.close()
        self.pool.join()


# --------------------------------------------------------------------------- #
# Summary of a policy (same layout as resumen() in p2_entrenar.py)
# --------------------------------------------------------------------------- #
def resumen(env, politica, episodios, titulo, detalle=False, seed0=1000):
    calidades, longitudes, desvios = [], [], []
    choques = perdidos = fuera = 0
    banda_total = 0.0

    print("\n" + titulo)
    print("-" * len(titulo))

    for ep in range(episodios):
        reg = run_episode(politica, env, seed0 + ep, detalle)
        st = stats_episodio(reg)
        q = calidad(reg, env.max_steps)
        calidades.append(q)
        longitudes.append(st["pasos"])
        desvios.append(st["desvio_medio_m"])
        banda_total += st["pct_banda"]
        choques += st["choque"]
        perdidos += st["perdido"]
        fuera += st["fuera"]
        print("episodio {:2d}   pasos {:3d}   calidad {:8.3f}   desvío medio {:5.0f} mm   "
              "máx {:5.0f} mm   % en banda {:5.1f}   {}".format(
                  ep + 1, st["pasos"], q, st["desvio_medio_m"] * 1000,
                  st["desvio_max_m"] * 1000, st["pct_banda"], desenlace(st)))

    print("\ncalidad media     {:.3f}  (desviación {:.3f})".format(
        float(np.mean(calidades)), float(np.std(calidades))))
    print("longitud media    {:.1f} pasos".format(float(np.mean(longitudes))))
    print("desvío medio      {:.0f} mm".format(1000 * float(np.mean(desvios))))
    print("% en banda medio  {:.1f}".format(banda_total / episodios))
    print("choques           {}/{}".format(choques, episodios))
    print("perdidos          {}/{}".format(perdidos, episodios))
    print("fuera del camino  {}/{}".format(fuera, episodios))


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def _ejecutar(poblacion, config, args, n_generaciones):
    os.makedirs("chk", exist_ok=True)               # the Checkpointer does not create folders
    poblacion.add_reporter(neat.StdOutReporter(True))
    estadisticas = neat.StatisticsReporter()
    poblacion.add_reporter(estadisticas)
    poblacion.add_reporter(neat.Checkpointer(generation_interval=1,
                                             filename_prefix=f"chk/{args.prefijo}-"))
    poblacion.add_reporter(RegistroGeneracion(args.archivo_final, "genomas"))

    ev = PoolEvaluator(args.workers, args.dt, args.iteraciones, args.pasos_max, args.modo,
                       detalle=args.detalle, aleat=not args.sin_aleatorizar, p_sat=args.saturacion)
    try:
        poblacion.run(ev, n_generaciones)
    except KeyboardInterrupt:
        print("\ninterrumpido - validando lo que hay")
    finally:
        ev.close()

    candidatos = estadisticas.best_genomes(5)
    if not candidatos:
        print("no hay genomas que guardar")
        return
    puntuados = [(validate(g, config, args.dt, args.pasos_max, args.modo, args.saturacion), g) for g in candidatos]
    for sc, g in sorted(puntuados, key=lambda t: -t[0]):
        print(f"candidato id={g.key}  entrenamiento={g.fitness:.3f}  validación={sc:.3f}")
    mejor_sc, mejor = max(puntuados, key=lambda t: t[0])
    with open(args.ruta_modelo, "wb") as f:
        pickle.dump(mejor, f)                      # just the genome, like p2_entrenar.py
    print(f"guardado {args.ruta_modelo} (validación {mejor_sc:.3f})")


def modo_entrenar(args):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, args.ruta_config)
    _ejecutar(neat.Population(config), config, args, args.generaciones)


def reanudar_entrenamiento(args):
    poblacion = neat.Checkpointer.restore_checkpoint(f"chk/{args.prefijo}-{args.checkpoint}")
    # reporters are not restored: they are added again, exactly as when starting.
    # (checkpoint N holds the population of generation N; RegistroGeneracion discards the
    #  CSV rows from N onwards so the re-run generations are not logged twice)
    _ejecutar(poblacion, poblacion.config, args, args.pasos_faltantes)


def _cargar_red(args):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, args.ruta_config)
    with open(args.ruta_modelo, "rb") as f:
        genoma = pickle.load(f)
    return neat.nn.FeedForwardNetwork.create(genoma, config)


def _crear_env(args, modo=None):
    return FollowEnv(dt=args.dt, max_time=args.pasos_max * args.dt,
                     aleatorizar=getattr(args, 'aleatorizar_eval', False),
                     modo=args.modo[0] if modo is None else modo)


def modo_evaluar(args):
    red = _cargar_red(args)
    for m in args.modo:                                   # one summary per leader mode
        resumen(_crear_env(args, m), red.activate, args.episodios,
                f"NEAT ({args.ruta_modelo}) - modo {m}", args.detalle)


def modo_ver(args):
    red = _cargar_red(args)
    watch(red.activate, dt=args.dt, seed=args.seed,
          speed=args.velocidad, max_time=args.pasos_max * args.dt,
          track_img=args.imagen_pista, robot_img=args.imagen_robot,
          modo=args.modo[0], aleatorizar=getattr(args, 'aleatorizar_eval', False))


def main():
    p = argparse.ArgumentParser(
        description="Entrenamiento NEAT (en paralelo) del seguimiento de un robot objetivo "
                    "manteniendo la distancia y sin salirse del camino.")

    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument("--comprobar", action="store_true", help="controlador P de referencia")
    modo.add_argument("--aleatorio", action="store_true", help="acciones aleatorias")
    modo.add_argument("--entrenar", action="store_true")
    modo.add_argument("--evaluar", action="store_true")
    modo.add_argument("--reanudar-entrenamiento", action="store_true")
    modo.add_argument("--ver", action="store_true", help="ventana pygame con el genoma guardado")

    p.add_argument("--dt", type=float, default=PASO_S, help="periodo de control (s); por defecto PASO_S de follow_sim.py")
    p.add_argument("--pasos-max", type=int, default=800,
                   help="longitud máxima de un episodio (pasos). 800 pasos x 0.29 s a 0.04 m/s = una vuelta completa")
    p.add_argument("--iteraciones", type=int, default=5, help="episodios por genoma y generación")
    p.add_argument("--episodios", type=int, default=5, help="episodios en --evaluar/--comprobar/--aleatorio")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--detalle", action="store_true",
                   help="una línea por individuo (entrenar) o por paso (evaluar/comprobar)")
    p.add_argument("--saturacion", type=float, default=0.0,
                   help="probabilidad (0-1) de que en un episodio el área del blob se sature (~133 px) al acercarse; "
                        "0 = nunca. Úsalo solo si el simulador oficial satura el tamaño del blob de cerca")
    p.add_argument("--sin-aleatorizar", action="store_true",
                   help="entrenar solo con el simulador nominal (sin aleatorizar planta/sensor)")
    p.add_argument("--aleatorizar-eval", action="store_true", help="evaluar/ver con el simulador aleatorizado")
    p.add_argument("--camino", default="verde", choices=["verde", "roja"], help="camino del líder (evaluar/ver)")
    p.add_argument("--perfil", default="const", choices=["const", "stop_go", "sine"])
    p.add_argument("--velocidad", type=float, default=1.0, help="velocidad de reproducción en --ver")
    p.add_argument("--imagen-pista", default=None)
    p.add_argument("--imagen-robot", default=None)

    p.add_argument("--ruta-modelo", default="mejor_genoma.plk", help="archivo del mejor genoma")
    p.add_argument("--ruta-config", default="config.txt")
    p.add_argument("--prefijo", default="p2", help="prefijo del nombre del checkpoint")
    p.add_argument("--archivo-final", default="evolucion.csv", help="CSV con el histórico de la evolución")
    p.add_argument("--checkpoint", type=int, help="generación del checkpoint desde el que reanudar")
    p.add_argument("--pasos-faltantes", type=int, help="generaciones adicionales al reanudar")
    p.add_argument("--generaciones", type=int, default=30)
    p.add_argument("--modo", default="1",
                   help="modo del líder (ver `modos` en follow_sim.py): 1 const 0.04 m/s, 2 curvas rápidas, 3 velocidad "
                        "aleatoria, 4 con paradas. Varios separados por comas (p. ej. 1,2,3,4): entrenar alterna un modo "
                        "por episodio, --evaluar da un resumen por modo, --ver usa el primero")

    args = p.parse_args()
    try:
        args.modo = tuple(int(m) for m in str(args.modo).split(","))
    except ValueError:
        p.error("--modo: enteros separados por comas")
    for m in args.modo:
        if m not in modos:
            p.error(f"--modo {m}: no existe; disponibles: {sorted(modos)}")
    if args.perfil != "const" or args.camino != "verde":
        print("AVISO: --perfil y --camino no hacen nada (el líder se configura con --modo; el camino es el de follow_sim.py)")
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.reanudar_entrenamiento:
        if args.checkpoint is None or args.pasos_faltantes is None:
            p.error("--reanudar-entrenamiento necesita --checkpoint y --pasos-faltantes")
        reanudar_entrenamiento(args)
    elif args.entrenar:
        modo_entrenar(args)
    elif args.evaluar:
        modo_evaluar(args)
    elif args.ver:
        modo_ver(args)
    elif args.comprobar:
        resumen(_crear_env(args), p_controller, args.episodios, "Controlador P de referencia", args.detalle)
    elif args.aleatorio:
        resumen(_crear_env(args), None, args.episodios, "Acciones aleatorias", args.detalle)


if __name__ == "__main__":
    main()
