import csv, os, pickle, statistics, time
import neat

CONTADOR = {"pasos": 0, "evaluaciones": 0} # lo actualiza evaluar_genoma()

class RegistroGeneracion(neat.reporting.BaseReporter):
    CAMPOS = ["generacion", "cal_mejor", "cal_media", "cal_desv", "especies",
            "nodos_mejor", "conexiones_mejor", "evaluaciones", "pasos",
            "segundos"]
    
    def __init__(self, ruta_csv, carpeta_genomas):
        self.ruta, self.carpeta = ruta_csv, carpeta_genomas
        os.makedirs(carpeta_genomas, exist_ok=True)
        self.base = {"evaluaciones": 0, "pasos": 0, "segundos": 0.0}
        if os.path.exists(ruta_csv): # se reanuda una evolución
            with open(ruta_csv) as f:
                filas = list(csv.DictReader(f))
            if filas:
                self.base = {k: float(filas[-1][k]) for k in self.base}
        else:
            with open(ruta_csv, "w", newline="") as f:
                csv.writer(f).writerow(self.CAMPOS)
        self.inicio = dict(CONTADOR) # por si el proceso ya había evaluado
        self.t0, self.gen = time.time(), 0

    def start_generation(self, generation):
        self.gen = generation

    def post_evaluate(self, config, population, species, best_genome):
        cal = [g.fitness for g in population.values()]
        nodos, conexiones = best_genome.size()
        fila = [self.gen, best_genome.fitness,
                statistics.mean(cal), statistics.pstdev(cal),
                len(species.species), nodos, conexiones,
                int(self.base["evaluaciones"] + CONTADOR["evaluaciones"]
                - self.inicio["evaluaciones"]),
                int(self.base["pasos"] + CONTADOR["pasos"] - self.inicio["pasos"]),
                round(self.base["segundos"] + time.time() - self.t0, 1)]
        with open(self.ruta, "a", newline="") as f:
            csv.writer(f).writerow(fila)
        ruta = os.path.join(self.carpeta, f"mejor_gen{self.gen:03d}.pkl")
        with open(ruta, "wb") as f:
            pickle.dump(best_genome, f)