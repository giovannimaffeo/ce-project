import os
import pandas as pd

import utils

# Caminho base da combinação
combination_path = "outputs/evolve_both/evolve_both/hiperparams_fatorial_test/reference_run2/combination1"

# Caminhos dos CSVs
fitness_historic_paths = [
  os.path.join(combination_path, f"run{i}", "fitness_history.csv")
  for i in range(1, 6)
]

# Extrair os best_fitnesses finais de cada run
best_fitnesses = [
  pd.read_csv(path)["best_fitness"].iloc[-1]
  for path in fitness_historic_paths
]

# Parâmetros da combinação (você deve preencher manualmente ou ler de outro CSV se existir)
combination_variable_params = {
  "STRUCUTRE_MUTATION_RATE": 0.1
}

# Executar
utils.generate_combination_results(
  combination_variable_params=combination_variable_params,
  best_fitnesses=best_fitnesses,
  fitness_historic_paths=fitness_historic_paths,
  combination_output_dir=combination_path
)