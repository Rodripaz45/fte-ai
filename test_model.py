from app.ml.model_loader import load_model
import numpy as np

# Texto del ingeniero electromecánico que enviaste
CV_TEXT = """
Ingeniero electromechanical junior con experiencia en HSE, gestión de calidad, supervisión de calibración, 
planificación logística y análisis de datos en empresas multinacionales del sector energético. Habilidades en 
Power BI, VBA y Microsoft Office 365 para desarrollar KPIs, digitalizar procesos y apoyar iniciativas de mejora 
continua. Enfoque sólido en cumplimiento, informes técnicos y trabajo en equipo, con un enfoque proactivo y 
organizado para entregar resultados de alta calidad en entornos dinámicos.

Licenciatura en Ingeniería Electromecánica con propuesta de diseño de un sistema de automatización neumática 
para el área de almacenamiento y transferencia de cemento a granel. Actualmente cursando una Maestría en 
Administración de Empresas y Gestión de Proyectos, con énfasis en planificación, control y gestión de calidad, 
así como en riesgos financieros, integración, costos, recursos humanos, innovación, estrategias y marketing.

Experiencia como intérprete en español-inglés, realizando interpretaciones de llamadas de audio, principalmente 
en servicios médicos y procesos bancarios. Participación en un programa de desarrollo de excelencia en ingeniería 
con enfoque en cementación y pruebas de laboratorio, así como en la planificación logística de operaciones de 
perforación. Experiencia en análisis de bases de datos y desarrollo de KPIs relacionados con la integridad de 
tuberías.

Coordinador de comunicación en ferias de proyectos humanitarios, organizando eventos con numerosos participantes 
y gestionando propuestas de proyectos. Experiencia en liderazgo y organización de proyectos relacionados con la 
juventud, derechos civiles y bienestar comunitario. Habilidades en gestión de datos sensibles de clientes y en 
el uso de herramientas de software como AutoCAD, SolidWorks y SAP.

Competencias en mediciones eléctricas, mantenimiento mecánico y realización de pruebas de laboratorio de cemento. 
Coordinación de operaciones logísticas y gestión de calidad bajo normas ISO. Auditorías de HSE y SQ, así como 
gestión de datos sensibles para servicios de calidad. Idiomas: español nativo, inglés avanzado y francés básico.
"""

def main():
    print("=" * 80)
    print("PRUEBA DEL MODELO ENTRENADO")
    print("=" * 80)
    
    # Cargar modelo
    pipeline, classes, metadata = load_model()
    print(f"\nMetadata del modelo:")
    print(f"  - Umbral óptimo: {metadata.get('best_threshold', 'N/A')}")
    print(f"  - F2 macro: {metadata.get('cv_macro_f2', 'N/A'):.3f}")
    print(f"  - Cardinalidad de etiquetas: {metadata.get('label_cardinality', 'N/A'):.2f}")
    print(f"  - Clases totales: {len(classes)}")
    
    # Preprocesar texto
    text = CV_TEXT.strip().lower()
    
    # Predicción
    try:
        proba = pipeline.predict_proba([text])[0]
    except Exception:
        logits = pipeline.decision_function([text])[0]
        proba = 1 / (1 + np.exp(-logits))
    
    threshold = float(metadata.get("best_threshold", 0.35))
    
    # Resultados por encima del umbral
    results_above = []
    for cls, p in zip(classes, proba):
        if p >= threshold:
            results_above.append((cls, p))
    results_above.sort(key=lambda x: -x[1])
    
    print(f"\n{'='*80}")
    print(f"RESULTADOS CON UMBRAL {threshold:.2f}")
    print(f"{'='*80}")
    print(f"Competencias detectadas: {len(results_above)}")
    if results_above:
        for i, (cls, p) in enumerate(results_above, 1):
            print(f"  {i:2}. {cls:35} -> {p*100:5.1f}%")
    else:
        print("  (!) Ninguna competencia supero el umbral")
    
    # Top 10 probabilidades (sin umbral)
    all_probs = list(zip(classes, proba))
    all_probs.sort(key=lambda x: -x[1])
    print(f"\n{'='*80}")
    print(f"TOP 10 PROBABILIDADES (sin filtrar por umbral)")
    print(f"{'='*80}")
    for i, (cls, p) in enumerate(all_probs[:10], 1):
        marker = "+" if p >= threshold else " "
        print(f" {marker} {i:2}. {cls:35} -> {p*100:5.1f}%")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()

