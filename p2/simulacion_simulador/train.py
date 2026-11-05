"""
train.py - parallel NEAT training / evaluation / viewing for the follower robot.

Modes (same names as p2_entrenar.py, plus a few extras for the simulator):
    python train.py --entrenar --ruta-config config.txt --generaciones 100 --dt 0.2 --pasos-max 150
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

from follow_sim import (DEV_SOFT_M, DEV_TOL_M, DIST_TARGET_M, NORM_BANDA_OBJETIVO, X_TOLERANCIA,
                        FollowEnv, p_controller, watch)
from registro import RegistroGeneracion

SCENARIOS = [("verde", "const"), ("roja", "const"), ("verde", "stop_go"),
             ("roja", "sine"), ("verde", "sine"), ("roja", "stop_go")]

SIGMA_CERCA, SIGMA_LEJOS, SIGMA_X = 0.20, 0.50, 0.35
W_DESVIO = 0.8        # max per-step penalty for straying from the chosen path


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #
def _r_dist(info):
    d_real = info["distancia_real_mm"] / 1000.0
    d = (d_real - DIST_TARGET_M) / DIST_TARGET_M          # unclipped, from ground truth
    s = SIGMA_CERCA if d < 0 else SIGMA_LEJOS
    return float(np.exp(-0.5 * (d / s) ** 2))


def penal_desvio(info):
    """0 inside the band around the path (DEV_TOL_M), growing quadratically to W_DESVIO at DEV_SOFT_M."""
    x = (info["desvio_m"] - DEV_TOL_M) / (DEV_SOFT_M - DEV_TOL_M)
    return W_DESVIO * min(1.0, max(0.0, x)) ** 2


def r_paso(info):
    """Reward of one non-terminal step."""
    if not info["visible"]:
        d_real = info["distancia_real_mm"] / 1000.0
        err = min(1.0, abs(d_real - DIST_TARGET_M) / DIST_TARGET_M)
        return -0.2 - 0.3 * err - penal_desvio(info)

    e_x = abs(info["blob_x"])
    r = _r_dist(info) * (0.4 + 0.6 * float(np.exp(-0.5 * (e_x / SIGMA_X) ** 2)))
    if e_x <= X_TOLERANCIA and NORM_BANDA_OBJETIVO[0] <= info["dist"] <= NORM_BANDA_OBJETIVO[1]:
        racha = min(info["en_banda_consecutivos"], 50) / 50.0
        r += 0.3 * (0.5 + 0.5 * racha)
    return r - penal_desvio(info)


def calidad(registro, pasos_eval):
    total, n = 0.0, len(registro)
    for info in registro:
        if info["choque"] or info["perdido"] or info["fuera_camino"]:
            total -= 1.0 * (pasos_eval - n + 1)            # charge the steps never played
            if info["choque"]:
                total -= 3.0
            elif info["fuera_camino"]:
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


def run_episode(politica, env, seed, detalle=False):
    """politica: obs -> action (or None for random actions). Returns the list of info dicts."""
    obs = env.reset(seed)
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


def evaluate(genome, config, gen_seed, dt, n_iter, pasos_max):
    """mean - 0.5*std over several scenarios; same seeds for every genome in a generation.
    Returns (fitness, stats) where stats summarises the behaviour over the episodes."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    scores, sts = [], []
    for i in range(n_iter):
        path, profile = SCENARIOS[(gen_seed + i) % len(SCENARIOS)]
        env = FollowEnv(dt=dt, max_time=pasos_max * dt, path=path, profile=profile)
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
    return float(s.mean() - 0.5 * s.std()), stats


def _job(args):
    return evaluate(*args)


def validate(genome, config, dt, pasos_max, seeds=(10_001, 10_002)):
    """Fixed, larger test set (every scenario x several seeds) to pick the real best genome."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    scores = []
    for sd in seeds:
        for j, (path, profile) in enumerate(SCENARIOS):
            env = FollowEnv(dt=dt, max_time=pasos_max * dt, path=path, profile=profile)
            scores.append(calidad(run_episode(net.activate, env, sd * 100 + j), env.max_steps))
    s = np.asarray(scores)
    return float(s.mean() - 0.5 * s.std())


# --------------------------------------------------------------------------- #
# Parallel evaluator (a Pool so we can pass a per-generation seed to the workers)
# --------------------------------------------------------------------------- #
class PoolEvaluator:
    def __init__(self, workers, dt, n_iter, pasos_max, detalle=False):
        self.pool = mp.Pool(workers)
        self.dt, self.n_iter, self.pasos_max, self.detalle = dt, n_iter, pasos_max, detalle
        self.contador = {"pasos": 0, "evaluaciones": 0}

    def __call__(self, genomas, config):
        gen_seed = random.randrange(1_000_000)
        jobs = [(g, config, gen_seed, self.dt, self.n_iter, self.pasos_max) for _, g in genomas]
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

    ev = PoolEvaluator(args.workers, args.dt, args.iteraciones, args.pasos_max, args.detalle)
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
    puntuados = [(validate(g, config, args.dt, args.pasos_max), g) for g in candidatos]
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


def _crear_env(args):
    return FollowEnv(dt=args.dt, max_time=args.pasos_max * args.dt,
                     path=args.camino, profile=args.perfil)


def modo_evaluar(args):
    red = _cargar_red(args)
    resumen(_crear_env(args), red.activate, args.episodios,
            f"NEAT ({args.ruta_modelo}) - {args.perfil}/{args.camino}", args.detalle)


def modo_ver(args):
    red = _cargar_red(args)
    watch(red.activate, dt=args.dt, path=args.camino, profile=args.perfil, seed=args.seed,
          speed=args.velocidad, max_time=args.pasos_max * args.dt,
          track_img=args.imagen_pista, robot_img=args.imagen_robot)


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

    p.add_argument("--dt", type=float, default=0.2, help="periodo de control (s), cualquier valor")
    p.add_argument("--pasos-max", type=int, default=150, help="longitud máxima de un episodio (pasos)")
    p.add_argument("--iteraciones", type=int, default=3, help="episodios por genoma y generación")
    p.add_argument("--episodios", type=int, default=5, help="episodios en --evaluar/--comprobar/--aleatorio")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--detalle", action="store_true",
                   help="una línea por individuo (entrenar) o por paso (evaluar/comprobar)")
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

    args = p.parse_args()
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