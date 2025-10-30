import pandas as pd
from collections import Counter

# Leer dataset aumentado
df = pd.read_csv("data/dataset_competencias_aug.csv")

print("="*80)
print("ANALISIS DE DISTRIBUCION DEL DATASET")
print("="*80)
print(f"\nTotal de filas: {len(df)}")

# Contar ocurrencias de cada competencia
all_competencias = []
for comps in df["competencias"].fillna(""):
    labels = [c.strip() for c in str(comps).split(",") if c.strip()]
    all_competencias.extend(labels)

counter = Counter(all_competencias)
print(f"\nTotal de competencias unicas: {len(counter)}")
print(f"Total de etiquetas asignadas: {len(all_competencias)}")
print(f"Promedio de etiquetas por fila: {len(all_competencias) / len(df):.2f}")

print("\n" + "="*80)
print("DISTRIBUCION POR COMPETENCIA (ordenado por frecuencia)")
print("="*80)

sorted_comps = sorted(counter.items(), key=lambda x: x[1], reverse=True)
for comp, count in sorted_comps:
    bar = "#" * count
    print(f"{comp:40} {count:3} {bar}")

# Identificar competencias con pocas muestras
print("\n" + "="*80)
print("COMPETENCIAS CON MENOS DE 5 MUESTRAS (necesitan mas ejemplos)")
print("="*80)
low_count = [(c, n) for c, n in sorted_comps if n < 5]
if low_count:
    for comp, count in low_count:
        print(f"  - {comp:40} ({count} muestras)")
else:
    print("  Todas las competencias tienen 5+ muestras")

print("\n" + "="*80)

