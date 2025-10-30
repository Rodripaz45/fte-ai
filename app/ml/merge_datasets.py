from __future__ import annotations
import argparse
import os
import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["cv_texto", "talleres", "competencias"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {missing}")
    # Normalización básica de tipos
    df["cv_texto"] = df["cv_texto"].fillna("").astype(str)
    df["talleres"] = df["talleres"].fillna("").astype(str)
    df["competencias"] = df["competencias"].fillna("").astype(str)
    return df


def main(base_path: str, extra_path: str, out_path: str):
    print(f"[merge] leyendo base: {base_path}")
    base = read_csv(base_path)
    print(f"[merge] leyendo extra: {extra_path}")
    extra = read_csv(extra_path)

    merged = pd.concat([base, extra], ignore_index=True)

    # Deduplicación conservadora por texto + etiquetas
    before = len(merged)
    merged = merged.drop_duplicates(subset=["cv_texto", "talleres", "competencias"]).reset_index(drop=True)
    after = len(merged)
    print(f"[merge] filas antes: {before}  después de deduplicar: {after}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[merge] dataset guardado en: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusiona dataset base y extra y genera uno aumentado")
    parser.add_argument("--base", default="data/dataset_competencias.csv", help="Ruta al dataset base")
    parser.add_argument("--extra", default="data/dataset_competencias_extra.csv", help="Ruta al dataset extra")
    parser.add_argument("--out", default="data/dataset_competencias_aug.csv", help="Ruta de salida")
    args = parser.parse_args()
    main(args.base, args.extra, args.out)


