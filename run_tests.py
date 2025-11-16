#!/usr/bin/env python
"""
Script auxiliar para ejecutar las pruebas del microservicio.
Facilita la ejecución de diferentes tipos de pruebas.
"""
import sys
import subprocess
import argparse


def run_command(cmd):
    """Ejecuta un comando y muestra el resultado."""
    print(f"\n{'='*60}")
    print(f"Ejecutando: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Ejecutar pruebas del microservicio FTE-AI")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "model", "all"],
        default="all",
        help="Tipo de pruebas a ejecutar"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generar reporte de cobertura"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Modo verboso"
    )
    parser.add_argument(
        "--markers",
        help="Filtrar por marcadores (ej: 'unit and not slow')"
    )
    
    args = parser.parse_args()
    
    cmd = ["pytest"]
    
    if args.type == "unit":
        cmd.extend(["tests/unit/", "-m", "unit"])
    elif args.type == "integration":
        cmd.extend(["tests/integration/", "-m", "integration"])
    elif args.type == "model":
        cmd.extend(["tests/model_quality/", "-m", "model_quality"])
    else:
        cmd.append("tests/")
    
    if args.coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])
    
    if args.verbose:
        cmd.append("-v")
    
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    exit_code = run_command(cmd)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

