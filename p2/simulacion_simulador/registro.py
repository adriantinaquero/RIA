"""
registro.py - per-generation logging for neat-python.

    poblacion.add_reporter(RegistroGeneracion("evolucion.csv", "genomas"))

After every generation it
  * appends one row to the CSV (fitness statistics, species, complexity of the best genome,
    and - when the evaluator attached `genome.stats` - behaviour metrics such as the mean
    deviation from the path, % of steps in the target band, crashes / lost / off-path counts),
  * pickles the best genome of that generation to `<carpeta>/gen_XXXX.pkl`
    (same format as `pickle.dump(mejor, f)` in p2_entrenar.py: just the genome).

The CSV is opened in append mode, so resuming from a checkpoint keeps adding to the same file.
"""
import csv
import os
import pickle
import time

import neat
import numpy as np

CABECERA = [
    "generacion", "tiempo_gen_s", "tiempo_total_s",
    "fit_mejor", "fit_media", "fit_std", "fit_min",
    "n_especies", "mejor_id", "mejor_nodos", "mejor_conexiones",
    "evaluaciones", "pasos", "pct_en_banda", "desvio_medio_mm",
    "choques", "perdidos", "fuera_camino",
]


class RegistroGeneracion(neat.reporting.BaseReporter):
    def __init__(self, archivo_csv, carpeta_genomas):
        self.archivo = archivo_csv
        self.carpeta = carpeta_genomas
        os.makedirs(self.carpeta, exist_ok=True)
        d = os.path.dirname(os.path.abspath(self.archivo))
        os.makedirs(d, exist_ok=True)
        self._escribir_cabecera = (not os.path.exists(self.archivo)
                                   or os.path.getsize(self.archivo) == 0)
        self.generacion = 0
        self._recortado = False
        self._t_inicio = time.time()
        self._t_gen = time.time()

    def start_generation(self, generation):
        if not self._recortado:
            self._recortado = True
            self._recortar_csv(generation)
        self.generacion = generation
        self._t_gen = time.time()

    def _recortar_csv(self, desde):
        """When resuming from a checkpoint, drop the rows of generations that are about to be re-run."""
        if not os.path.exists(self.archivo) or os.path.getsize(self.archivo) == 0:
            return
        with open(self.archivo, newline="") as f:
            filas = [r for r in csv.DictReader(f)]
        conservar = [r for r in filas if int(r["generacion"]) < desde]
        if len(conservar) != len(filas):
            with open(self.archivo, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CABECERA)
                w.writeheader()
                w.writerows(conservar)

    def post_evaluate(self, config, population, species, best_genome):
        genomas = list(population.values())
        fits = np.array([g.fitness for g in genomas if g.fitness is not None], dtype=float)
        stats = [g.stats for g in genomas if getattr(g, "stats", None)]

        def media(k):
            return float(np.mean([s[k] for s in stats])) if stats else float("nan")

        def suma(k):
            return int(np.sum([s[k] for s in stats])) if stats else 0

        n_nodos = len(best_genome.nodes)
        n_con = sum(1 for c in best_genome.connections.values() if c.enabled)

        fila = {
            "generacion": self.generacion,
            "tiempo_gen_s": round(time.time() - self._t_gen, 3),
            "tiempo_total_s": round(time.time() - self._t_inicio, 1),
            "fit_mejor": round(float(fits.max()), 5),
            "fit_media": round(float(fits.mean()), 5),
            "fit_std": round(float(fits.std()), 5),
            "fit_min": round(float(fits.min()), 5),
            "n_especies": len(species.species),
            "mejor_id": best_genome.key,
            "mejor_nodos": n_nodos,
            "mejor_conexiones": n_con,
            "evaluaciones": suma("episodios"),
            "pasos": suma("pasos"),
            "pct_en_banda": round(media("pct_banda"), 2),
            "desvio_medio_mm": round(media("desvio_medio_m") * 1000, 1),
            "choques": suma("choques"),
            "perdidos": suma("perdidos"),
            "fuera_camino": suma("fuera"),
        }

        with open(self.archivo, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CABECERA)
            if self._escribir_cabecera:
                w.writeheader()
                self._escribir_cabecera = False
            w.writerow(fila)

        with open(os.path.join(self.carpeta, f"gen_{self.generacion:04d}.pkl"), "wb") as f:
            pickle.dump(best_genome, f)
