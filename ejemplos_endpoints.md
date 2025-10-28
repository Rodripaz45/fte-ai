# Ejemplos de Uso de Endpoints - FTE-AI

## Endpoints Disponibles

### 1. POST `/analyze/profile` - Análisis de Perfil de Participante

Analiza el perfil de un participante basándose en su CV y asistencia a talleres.

#### Estructura del Request:
```json
{
  "participanteId": "string (requerido)",
  "incluirCV": "boolean (default: true)",
  "incluirTalleres": "boolean (default: true)",
  "useML": "boolean (default: false)",
  "talleres": [
    {
      "tema": "string",
      "asistencia_pct": "float (0.0 - 1.0)"
    }
  ],
  "cvTexto": "string (opcional)"
}
```

#### Ejemplo 1: Con talleres y CV (modo reglas)
Archivo: `examples_profile_1.json`
```json
{
  "participanteId": "P001-2024",
  "incluirCV": true,
  "incluirTalleres": true,
  "useML": false,
  "talleres": [
    {
      "tema": "Excel",
      "asistencia_pct": 0.85
    },
    {
      "tema": "python",
      "asistencia_pct": 0.70
    },
    {
      "tema": "Comunicación",
      "asistencia_pct": 0.90
    }
  ],
  "cvTexto": "Experiencia en análisis de datos con Python, SQL y Excel. Desarrollo de scripts para automatización de procesos. Conocimientos en Power BI para dashboards y reportes ejecutivos. Buena comunicación verbal y escrita. Liderazgo de equipos pequeños."
}
```

#### Ejemplo 2: Solo CV con ML habilitado
Archivo: `examples_profile_2.json`
```json
{
  "participanteId": "P002-2024",
  "incluirCV": true,
  "incluirTalleres": false,
  "useML": true,
  "cvTexto": "Ingeniero de sistemas con 5 años de experiencia en desarrollo web. Dominio de Django, Flask y FastAPI. Administración de bases de datos PostgreSQL y MySQL. Experiencia en cloud computing AWS. Buen manejo de Git y CI/CD. Conocimientos en machine learning básico con scikit-learn."
}
```

### 2. POST `/analyze/job` - Análisis de Requisitos de Puesto

Identifica las competencias necesarias para un puesto de trabajo basándose en su descripción.

#### Estructura del Request:
```json
{
  "puestoTexto": "string (requerido)",
  "topK": "integer (default: 6, rango: 1-20)"
}
```

#### Ejemplo 1: Puesto de atención al cliente
Archivo: `examples_job_1.json`
```json
{
  "puestoTexto": "Necesitamos alguien para atención al cliente que pueda manejar caja, procesar ventas, hacer seguimiento postventa y lidiar con reclamos. Que sepa usar Excel para reportes y que tenga buena comunicación.",
  "topK": 6
}
```

#### Ejemplo 2: Puesto de asistente contable
Archivo: `examples_job_2.json`
```json
{
  "puestoTexto": "Buscamos asistente contable con conocimientos en conciliaciones bancarias, facturación electrónica, uso de Excel avanzado con tablas dinámicas. Manejamos planilla de sueldos y necesitamos que sea organizado para la documentación administrativa.",
  "topK": 8
}
```

### 3. GET `/health` - Health Check

No requiere body. Responde:
```json
{
  "status": "ok",
  "message": "FTE-AI service is running"
}
```

## Cómo Usar los Ejemplos

### Con curl:
```bash
# Análisis de perfil (Ejemplo 1)
curl -X POST http://localhost:8000/analyze/profile \
  -H "Content-Type: application/json" \
  -d @examples_profile_1.json

# Análisis de perfil (Ejemplo 2)
curl -X POST http://localhost:8000/analyze/profile \
  -H "Content-Type: application/json" \
  -d @examples_profile_2.json

# Análisis de puesto (Ejemplo 1)
curl -X POST http://localhost:8000/analyze/job \
  -H "Content-Type: application/json" \
  -d @examples_job_1.json

# Análisis de puesto (Ejemplo 2)
curl -X POST http://localhost:8000/analyze/job \
  -H "Content-Type: application/json" \
  -d @examples_job_2.json

# Health check
curl -X GET http://localhost:8000/health
```

### Con Python (requests):
```python
import requests

# Análisis de perfil
with open('examples_profile_1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
response = requests.post('http://localhost:8000/analyze/profile', json=data)
print(response.json())

# Análisis de puesto
with open('examples_job_1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
response = requests.post('http://localhost:8000/analyze/job', json=data)
print(response.json())
```

### Con Postman o Insomnia:
1. Importa el JSON del archivo correspondiente
2. URL: `http://localhost:8000/analyze/profile` o `http://localhost:8000/analyze/job`
3. Método: POST
4. Headers: `Content-Type: application/json`
5. Body: copia el contenido del archivo JSON

## Respuestas Esperadas

### Análisis de Perfil (con useML=false):
```json
{
  "participanteId": "P001-2024",
  "competencias": [
    {
      "competencia": "Analítica de Datos",
      "nivel": 85.0,
      "confianza": 0.85,
      "fuente": ["talleres", "cv"]
    },
    {
      "competencia": "Programación",
      "nivel": 70.0,
      "confianza": 0.75,
      "fuente": ["talleres"]
    }
  ],
  "meta": {
    "mode": "rules",
    "W_TALLERES": 0.6,
    "W_CV": 0.4
  }
}
```

### Análisis de Job:
```json
{
  "competencias": [
    {
      "nombre": "Atención al Cliente",
      "nivel": 0.75,
      "fuente": ["keywords"]
    },
    {
      "nombre": "Ventas",
      "nivel": 0.65,
      "fuente": ["keywords"]
    }
  ],
  "meta": {
    "mode": "ml+keywords",
    "threshold": "0.35"
  }
}
```

## Notas Importantes

- El campo `asistencia_pct` en talleres debe estar entre 0.0 y 1.0
- El campo `topK` debe estar entre 1 y 20
- Si `useML=true` en análisis de perfil, usa el modelo ML para predicción
- Si `useML=false`, usa reglas basadas en keywords y talleres
- El endpoint de job usa ML si está disponible, sino usa keywords como fallback

