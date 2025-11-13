"""
Script para enriquecer el dataset con ejemplos basados en teorías de evaluación de currículums.
"""
import pandas as pd
import os

# Nuevos datos organizados por teoría
nuevos_registros = [
    # ===== 1. TEORÍA DEL AJUSTE PERSONA-PUESTO =====
    {
        "cv_texto": "Gestioné planilla de sueldos y liquidaciones para 50 empleados en empresa manufacturera, conciliando aportes y declaraciones juradas.",
        "talleres": "planilla de sueldos, contabilidad laboral",
        "competencias": "RRHH, Contabilidad"
    },
    {
        "cv_texto": "Implementé sistema de facturación electrónica y conciliaciones bancarias diarias para PYME comercial.",
        "talleres": "facturación electrónica, conciliaciones bancarias",
        "competencias": "Contabilidad, Ofimática"
    },
    {
        "cv_texto": "Supervisé almacén de productos químicos cumpliendo normativas de seguridad e higiene y control de inventario.",
        "talleres": "seguridad e higiene, control de stock",
        "competencias": "Logística, Seguridad e Higiene"
    },
    {
        "cv_texto": "Desarrollé scripts en Python para automatizar reportes contables y análisis de variaciones presupuestarias.",
        "talleres": "python, contabilidad, analisis de datos",
        "competencias": "Contabilidad, Analisis de Datos"
    },
    {
        "cv_texto": "Gestioné compras de materias primas y coordiné logística de importación con despachantes aduaneros.",
        "talleres": "procurement, comercio exterior",
        "competencias": "Compras, Comercio Exterior"
    },
    {
        "cv_texto": "Implementé protocolos ISO 9001 en línea de producción y capacitaciones en calidad para operarios.",
        "talleres": "iso 9001, gestion de calidad",
        "competencias": "Calidad, Producción"
    },
    {
        "cv_texto": "Coordiné mantenimiento preventivo de maquinaria industrial y gestión de repuestos críticos.",
        "talleres": "mantenimiento preventivo, cmms",
        "competencias": "Mantenimiento, Logística"
    },
    {
        "cv_texto": "Gestioné cartera de clientes B2B con CRM, negociando contratos marco y seguimiento post venta.",
        "talleres": "crm, negociacion, servicio al cliente",
        "competencias": "Ventas, Atención al Cliente"
    },
    {
        "cv_texto": "Elaboré presupuestos anuales y reportes financieros para dirección con análisis de rentabilidad por línea.",
        "talleres": "control de gestión, finanzas corporativas",
        "competencias": "Finanzas, Administración"
    },
    {
        "cv_texto": "Implementé tableros en Power BI para seguimiento de KPIs logísticos y control de inventarios.",
        "talleres": "power bi basico, logistica",
        "competencias": "Analisis de Datos, Logística"
    },
    
    # ===== 2. TEORÍA DE COMPETENCIAS LABORALES =====
    {
        "cv_texto": "Lideré equipo de 8 personas en área de ventas, capacitando en técnicas consultivas y logrando incremento del 25% en facturación.",
        "talleres": "liderazgo, ventas consultivas, capacitacion",
        "competencias": "Ventas, RRHH"
    },
    {
        "cv_texto": "Desarrollé habilidades de negociación compleja cerrando acuerdos con proveedores estratégicos reduciendo costos en 15%.",
        "talleres": "negociacion, procurement",
        "competencias": "Compras, Negociación"
    },
    {
        "cv_texto": "Demostré capacidad de análisis crítico identificando ineficiencias en procesos logísticos y proponiendo mejoras implementadas.",
        "talleres": "analisis de datos, logistica",
        "competencias": "Analisis de Datos, Logística"
    },
    {
        "cv_texto": "Gestioné múltiples proyectos simultáneos con metodologías ágiles, cumpliendo plazos y presupuestos asignados.",
        "talleres": "metodologias agiles, planificacion",
        "competencias": "Gestion de Proyectos"
    },
    {
        "cv_texto": "Comunicé resultados financieros a dirección mediante presentaciones ejecutivas y reportes claros.",
        "talleres": "comunicacion efectiva, finanzas corporativas",
        "competencias": "Finanzas, Comunicación"
    },
    {
        "cv_texto": "Trabajé bajo presión en cierres contables mensuales manteniendo precisión y cumplimiento de deadlines.",
        "talleres": "contabilidad, gestion del tiempo",
        "competencias": "Contabilidad"
    },
    {
        "cv_texto": "Adapté procesos de atención al cliente a modalidad remota implementando herramientas digitales.",
        "talleres": "servicio al cliente, herramientas digitales",
        "competencias": "Atención al Cliente"
    },
    {
        "cv_texto": "Colaboré con equipos multidisciplinarios en implementación de ERP coordinando áreas contables y logísticas.",
        "talleres": "erp, trabajo en equipo",
        "competencias": "Gestion de Proyectos, Contabilidad"
    },
    {
        "cv_texto": "Resolví conflictos con clientes insatisfechos aplicando técnicas de escucha activa y negociación.",
        "talleres": "escucha activa, negociacion, servicio al cliente",
        "competencias": "Atención al Cliente, Negociación"
    },
    {
        "cv_texto": "Capacité a nuevos empleados en procesos administrativos y uso de sistemas internos.",
        "talleres": "induccion, capacitacion",
        "competencias": "RRHH, Educación"
    },
    
    # ===== 3. TEORÍA DEL CAPITAL HUMANO =====
    {
        "cv_texto": "Certificado en Power BI avanzado, implementé dashboards ejecutivos y capacitaciones internas en herramientas de BI.",
        "talleres": "power bi avanzado, capacitacion",
        "competencias": "Analisis de Datos, Educación"
    },
    {
        "cv_texto": "Completé diplomado en gestión de calidad ISO 9001 aplicando conocimientos en auditorías internas de planta.",
        "talleres": "iso 9001, auditorias, certificaciones",
        "competencias": "Calidad"
    },
    {
        "cv_texto": "Realicé curso de Python para análisis de datos y automatización, desarrollando scripts para reportes financieros.",
        "talleres": "python, analisis de datos, certificaciones",
        "competencias": "Analisis de Datos"
    },
    {
        "cv_texto": "Certificado en metodologías ágiles Scrum, facilitando ceremonias y mejorando entregas de proyectos en 30%.",
        "talleres": "metodologias agiles, scrum, certificaciones",
        "competencias": "Gestion de Proyectos"
    },
    {
        "cv_texto": "Completé formación en comercio exterior y normativa aduanera, gestionando importaciones y documentación.",
        "talleres": "comercio exterior, normativa, certificaciones",
        "competencias": "Comercio Exterior"
    },
    {
        "cv_texto": "Realicé especialización en gestión de recursos humanos, implementando procesos de selección y desarrollo.",
        "talleres": "gestion del talento, seleccion por competencias, certificaciones",
        "competencias": "RRHH"
    },
    {
        "cv_texto": "Certificado en Lean Manufacturing, aplicando herramientas kaizen y reduciendo desperdicios en 20%.",
        "talleres": "lean manufacturing, kaizen, certificaciones",
        "competencias": "Producción, Calidad"
    },
    {
        "cv_texto": "Completé curso de Excel avanzado y Power Query, automatizando reportes contables y análisis financieros.",
        "talleres": "excel avanzado, power query, certificaciones",
        "competencias": "Ofimática"
    },
    {
        "cv_texto": "Realicé formación en seguridad e higiene laboral, implementando programas de prevención y capacitaciones.",
        "talleres": "seguridad e higiene, certificaciones",
        "competencias": "Seguridad e Higiene"
    },
    {
        "cv_texto": "Certificado en gestión de proyectos PMI, liderando implementaciones de sistemas y mejoras de procesos.",
        "talleres": "pm tools, certificaciones, planificacion",
        "competencias": "Gestion de Proyectos"
    },
    {
        "cv_texto": "Completé diplomado en marketing digital, ejecutando campañas en redes sociales y análisis de métricas.",
        "talleres": "marketing digital, certificaciones",
        "competencias": "Marketing"
    },
    {
        "cv_texto": "Realicé curso de SQL avanzado y bases de datos, desarrollando consultas complejas para análisis de negocio.",
        "talleres": "sql, bases de datos, certificaciones",
        "competencias": "Analisis de Datos"
    },
    
    # ===== 4. TEORÍA DE LA DECISIÓN RACIONAL =====
    {
        "cv_texto": "Incrementé ventas en 35% mediante estrategias de prospección y gestión de cartera con CRM.",
        "talleres": "ventas consultivas, crm",
        "competencias": "Ventas"
    },
    {
        "cv_texto": "Reduje tiempos de preparación de pedidos en 18% optimizando layout de almacén y procesos de picking.",
        "talleres": "logistica, gestion de almacenes",
        "competencias": "Logística"
    },
    {
        "cv_texto": "Mejoré NPS de atención al cliente de 65 a 82 puntos implementando protocolos de servicio.",
        "talleres": "servicio al cliente, comunicacion efectiva",
        "competencias": "Atención al Cliente"
    },
    {
        "cv_texto": "Reduje costos de compras en 12% mediante negociación estratégica y análisis de proveedores.",
        "talleres": "procurement, negociacion, analisis de costos",
        "competencias": "Compras"
    },
    {
        "cv_texto": "Optimicé flujo de caja reduciendo días de cobranza promedio de 45 a 28 días.",
        "talleres": "finanzas corporativas, analisis financiero",
        "competencias": "Finanzas"
    },
    {
        "cv_texto": "Reduje scrap en línea de producción de 3.2% a 1.8% aplicando controles de calidad estadísticos.",
        "talleres": "gestion de calidad, control estadistico",
        "competencias": "Calidad, Producción"
    },
    {
        "cv_texto": "Incrementé productividad de equipo en 22% mediante capacitaciones y mejoras de procesos.",
        "talleres": "gestion del talento, capacitacion",
        "competencias": "RRHH, Producción"
    },
    {
        "cv_texto": "Reduje tiempos de respuesta de mesa de ayuda de 4 horas a 1.5 horas promedio.",
        "talleres": "mesa de ayuda, herramientas it",
        "competencias": "Soporte Técnico"
    },
    {
        "cv_texto": "Mejoré precisión de inventarios de 92% a 98% implementando conteos cíclicos y reconciliaciones.",
        "talleres": "control de stock, logistica",
        "competencias": "Logística"
    },
    {
        "cv_texto": "Incrementé tasa de conversión de leads en 28% optimizando proceso de ventas y seguimiento.",
        "talleres": "ventas consultivas, crm",
        "competencias": "Ventas"
    },
    {
        "cv_texto": "Reduje tiempos de cierre contable mensual de 10 días a 6 días mediante automatizaciones.",
        "talleres": "contabilidad, automatizacion",
        "competencias": "Contabilidad, Ofimática"
    },
    {
        "cv_texto": "Mejoré cumplimiento de entregas OTIF de 85% a 94% optimizando planificación logística.",
        "talleres": "logistica, planificacion de rutas",
        "competencias": "Logística"
    },
    
    # ===== 5. TEORÍA DEL ENFOQUE CONDUCTISTA =====
    {
        "cv_texto": "Logré reducir quejas de clientes en 40% implementando protocolos de seguimiento post venta y capacitaciones.",
        "talleres": "servicio al cliente, gestion de reclamos",
        "competencias": "Atención al Cliente"
    },
    {
        "cv_texto": "Demostré capacidad de liderazgo gestionando equipo de 12 personas en área de producción con rotación cero.",
        "talleres": "liderazgo, gestion del talento",
        "competencias": "RRHH, Producción"
    },
    {
        "cv_texto": "Implementé sistema de control de calidad que redujo devoluciones de productos en 25% en primer trimestre.",
        "talleres": "gestion de calidad, iso 9001",
        "competencias": "Calidad"
    },
    {
        "cv_texto": "Gestioné crisis operativa durante peak de demanda manteniendo niveles de servicio y satisfacción del cliente.",
        "talleres": "logistica, servicio al cliente",
        "competencias": "Logística, Atención al Cliente"
    },
    {
        "cv_texto": "Logré cerrar negociación compleja con proveedor estratégico reduciendo costos anuales en $50,000 USD.",
        "talleres": "negociacion, procurement",
        "competencias": "Compras, Negociación"
    },
    {
        "cv_texto": "Demostré adaptabilidad migrando procesos administrativos a modalidad remota sin interrupciones operativas.",
        "talleres": "administracion, herramientas digitales",
        "competencias": "Administración"
    },
    {
        "cv_texto": "Implementé programa de seguridad que redujo accidentes laborales en 60% mediante capacitaciones y controles.",
        "talleres": "seguridad e higiene, capacitacion",
        "competencias": "Seguridad e Higiene"
    },
    {
        "cv_texto": "Gestioné proyecto crítico de implementación de ERP cumpliendo plazos y presupuesto asignado.",
        "talleres": "erp, planificacion, metodologias agiles",
        "competencias": "Gestion de Proyectos"
    },
    {
        "cv_texto": "Logré mejorar satisfacción de empleados de 6.2 a 8.1 puntos mediante programas de clima laboral.",
        "talleres": "clima laboral, gestion del talento",
        "competencias": "RRHH"
    },
    {
        "cv_texto": "Demostré proactividad identificando oportunidad de mejora en procesos logísticos ahorrando $30,000 anuales.",
        "talleres": "logistica, analisis de datos",
        "competencias": "Logística, Analisis de Datos"
    },
    
    # ===== 6. TEORÍA DEL ENFOQUE PSICOMÉTRICO =====
    {
        "cv_texto": "Certificado PMP con 5 años de experiencia gestionando proyectos de implementación de sistemas.",
        "talleres": "pm tools, certificaciones, metodologias agiles",
        "competencias": "Gestion de Proyectos"
    },
    {
        "cv_texto": "Contador público con especialización en NIIF, gestionando cierres contables y reportes corporativos.",
        "talleres": "contabilidad, normas niif, certificaciones",
        "competencias": "Contabilidad"
    },
    {
        "cv_texto": "Ingeniero industrial con certificación Lean Six Sigma, aplicando metodologías en mejora de procesos.",
        "talleres": "lean manufacturing, certificaciones, gestion de calidad",
        "competencias": "Producción, Calidad"
    },
    {
        "cv_texto": "Analista de datos certificado en Tableau y Power BI, desarrollando dashboards ejecutivos y reportes.",
        "talleres": "power bi basico, visualizacion, certificaciones",
        "competencias": "Analisis de Datos"
    },
    {
        "cv_texto": "Especialista en RRHH con certificación en selección por competencias, gestionando procesos de reclutamiento.",
        "talleres": "seleccion por competencias, reclutamiento, certificaciones",
        "competencias": "RRHH"
    },
    {
        "cv_texto": "Técnico en seguridad e higiene certificado, implementando programas de prevención y cumplimiento normativo.",
        "talleres": "seguridad e higiene, certificaciones",
        "competencias": "Seguridad e Higiene"
    },
    {
        "cv_texto": "Certificado en comercio exterior y aduanas, gestionando importaciones y documentación internacional.",
        "talleres": "comercio exterior, normativa, certificaciones",
        "competencias": "Comercio Exterior"
    },
    {
        "cv_texto": "Especialista en marketing digital con certificaciones en Google Analytics y Facebook Ads.",
        "talleres": "marketing digital, analytics, certificaciones",
        "competencias": "Marketing"
    },
    {
        "cv_texto": "Certificado en gestión de calidad ISO 9001, realizando auditorías internas y externas.",
        "talleres": "iso 9001, auditorias, certificaciones",
        "competencias": "Calidad"
    },
    {
        "cv_texto": "Analista financiero con certificación en análisis de estados financieros y modelado.",
        "talleres": "analisis financiero, modelos financieros, certificaciones",
        "competencias": "Finanzas"
    },
    
    # ===== 7. COMBINACIONES MULTI-COMPETENCIA (ENFOQUE SISTÉMICO) =====
    {
        "cv_texto": "Gestioné implementación de sistema de gestión integrado (calidad, seguridad y ambiente) coordinando auditorías y capacitaciones.",
        "talleres": "iso 9001, seguridad e higiene, capacitacion",
        "competencias": "Calidad, Seguridad e Higiene, RRHH"
    },
    {
        "cv_texto": "Coordiné lanzamiento de producto nuevo involucrando marketing, ventas, logística y atención al cliente.",
        "talleres": "marketing digital, ventas consultivas, logistica, servicio al cliente",
        "competencias": "Marketing, Ventas, Logística, Atención al Cliente"
    },
    {
        "cv_texto": "Implementé sistema de control de gestión integrando finanzas, compras y logística con dashboards ejecutivos.",
        "talleres": "control de gestión, finanzas corporativas, procurement, power bi basico",
        "competencias": "Finanzas, Compras, Logística, Analisis de Datos"
    },
    {
        "cv_texto": "Gestioné proceso de certificación ISO coordinando calidad, producción, mantenimiento y RRHH.",
        "talleres": "iso 9001, gestion de calidad, mantenimiento preventivo, gestion del talento",
        "competencias": "Calidad, Producción, Mantenimiento, RRHH"
    },
    {
        "cv_texto": "Lideré proyecto de digitalización de procesos administrativos involucrando IT, administración y capacitación.",
        "talleres": "herramientas it, administracion, capacitacion",
        "competencias": "Soporte Técnico, Administración, Educación"
    },
    {
        "cv_texto": "Coordiné estrategia comercial integrando marketing digital, ventas B2B y postventa con métricas unificadas.",
        "talleres": "marketing digital, ventas consultivas, servicio al cliente, analytics",
        "competencias": "Marketing, Ventas, Atención al Cliente"
    },
    {
        "cv_texto": "Gestioné cadena de suministro completa desde compras internacionales hasta distribución final.",
        "talleres": "procurement, comercio exterior, logistica, control de stock",
        "competencias": "Compras, Comercio Exterior, Logística"
    },
    {
        "cv_texto": "Implementé sistema de gestión de talento integrando reclutamiento, capacitación, clima y desarrollo.",
        "talleres": "reclutamiento, capacitacion, clima laboral, gestion del talento",
        "competencias": "RRHH, Educación"
    },
    {
        "cv_texto": "Coordiné proceso de expansión comercial involucrando legal, ventas, marketing y logística.",
        "talleres": "contratos, ventas consultivas, marketing digital, logistica",
        "competencias": "Legal, Ventas, Marketing, Logística"
    },
    {
        "cv_texto": "Gestioné transformación digital de procesos contables y financieros con integración de sistemas.",
        "talleres": "contabilidad, finanzas corporativas, erp, automatizacion",
        "competencias": "Contabilidad, Finanzas, Gestion de Proyectos"
    },
    
    # ===== 8. COMPETENCIAS TRANSVERSALES =====
    {
        "cv_texto": "Facilité reuniones de negociación entre áreas comerciales y operativas logrando acuerdos win-win.",
        "talleres": "negociacion, comunicacion efectiva, trabajo en equipo",
        "competencias": "Negociación, Comunicación"
    },
    {
        "cv_texto": "Desarrollé habilidades de comunicación asertiva presentando resultados a dirección y equipos multidisciplinarios.",
        "talleres": "comunicacion efectiva, presentaciones",
        "competencias": "Comunicación"
    },
    {
        "cv_texto": "Gestioné conflictos interpersonales en equipo aplicando técnicas de mediación y escucha activa.",
        "talleres": "escucha activa, trabajo en equipo, comunicacion",
        "competencias": "Comunicación"
    },
    {
        "cv_texto": "Negocié contratos complejos con proveedores internacionales considerando aspectos legales y comerciales.",
        "talleres": "negociacion, contratos, comercio exterior",
        "competencias": "Negociación, Legal, Comercio Exterior"
    },
    {
        "cv_texto": "Comunicé cambios organizacionales a equipos mediante presentaciones claras y sesiones de preguntas.",
        "talleres": "comunicacion efectiva, gestion del cambio",
        "competencias": "Comunicación, RRHH"
    },
    {
        "cv_texto": "Facilité procesos de negociación colectiva coordinando con áreas legales y recursos humanos.",
        "talleres": "negociacion, contratos, gestion del talento",
        "competencias": "Negociación, Legal, RRHH"
    },
    
    # ===== 9. CASOS ESPECÍFICOS: SALUD CON INFORMACIÓN PERSONAL MEZCLADA =====
    {
        "cv_texto": "Estudios realizados: Secretaria Administrativa en Cruz Roja Boliviana, Primeros Auxilios y Enfermería Básica, Segundo Curso de Enfermería. Universidad Nacional de Siglo XX Licenciada en Derecho.",
        "talleres": "primeros auxilios, enfermería básica, cruz roja",
        "competencias": "Salud, Legal"
    },
    {
        "cv_texto": "Formación en Cruz Roja: Primeros Auxilios y Enfermería Básica. Experiencia en atención primaria y campañas de salud comunitaria.",
        "talleres": "cruz roja, primeros auxilios, enfermería",
        "competencias": "Salud"
    },
    {
        "cv_texto": "Certificado en Primeros Auxilios por Cruz Roja Boliviana. Segundo Curso de Enfermería completado. Atención a pacientes en emergencias.",
        "talleres": "primeros auxilios, enfermería, cruz roja boliviana",
        "competencias": "Salud"
    },
    {
        "cv_texto": "Estudios: Secretaria Administrativa en Cruz Roja, Primeros Auxilios y Enfermería Básica. Experiencia en atención de emergencias y cuidado de pacientes.",
        "talleres": "secretaria administrativa, primeros auxilios, enfermería básica",
        "competencias": "Salud, Administración"
    },
    {
        "cv_texto": "Cruz Roja Boliviana: Primeros Auxilios y Enfermería Básica. Segundo Curso de Enfermería. Licenciada en Derecho. Experiencia en atención médica y asesoría legal.",
        "talleres": "cruz roja, enfermería, derecho",
        "competencias": "Salud, Legal"
    },
    {
        "cv_texto": "Formación en salud: Primeros Auxilios, Enfermería Básica y Segundo Curso de Enfermería en Cruz Roja. Atención primaria y campañas de prevención.",
        "talleres": "primeros auxilios, enfermería básica, salud comunitaria",
        "competencias": "Salud"
    },
    {
        "cv_texto": "Estudios realizados en Cruz Roja Boliviana incluyendo Secretaria Administrativa, Primeros Auxilios y Enfermería Básica. Segundo Curso de Enfermería completado.",
        "talleres": "cruz roja boliviana, primeros auxilios, enfermería",
        "competencias": "Salud"
    },
    {
        "cv_texto": "Experiencia en Cruz Roja: Secretaria Administrativa, Primeros Auxilios y Enfermería Básica. Segundo Curso de Enfermería. Atención de pacientes y gestión administrativa.",
        "talleres": "cruz roja, primeros auxilios, enfermería, administración",
        "competencias": "Salud, Administración"
    },
    
    # ===== 10. MÁS VARIACIONES DE SALUD =====
    {
        "cv_texto": "Técnico en enfermería con certificación en primeros auxilios. Experiencia en atención de pacientes y aplicación de protocolos de bioseguridad.",
        "talleres": "enfermería, primeros auxilios, bioseguridad",
        "competencias": "Salud"
    },
    {
        "cv_texto": "Voluntaria en Cruz Roja realizando primeros auxilios en eventos masivos. Capacitación en enfermería básica y atención de emergencias.",
        "talleres": "cruz roja, primeros auxilios, enfermería básica",
        "competencias": "Salud"
    },
    {
        "cv_texto": "Formación en salud: Primeros Auxilios, Enfermería Básica y Segundo Curso de Enfermería. Experiencia en vacunatorios y campañas de salud.",
        "talleres": "primeros auxilios, enfermería, vacunación",
        "competencias": "Salud"
    },
    {
        "cv_texto": "Certificada en Primeros Auxilios y Enfermería Básica por Cruz Roja. Segundo Curso de Enfermería. Atención primaria en salud comunitaria.",
        "talleres": "cruz roja, primeros auxilios, enfermería básica, salud comunitaria",
        "competencias": "Salud"
    },
    {
        "cv_texto": "Estudios en Cruz Roja Boliviana: Secretaria Administrativa, Primeros Auxilios y Enfermería Básica. Segundo Curso de Enfermería. Licenciada en Derecho con experiencia en salud.",
        "talleres": "cruz roja, enfermería, derecho, salud",
        "competencias": "Salud, Legal"
    },
    
    # ===== 11. CASOS CON INFORMACIÓN PERSONAL MEZCLADA (FORMATO REALISTA) =====
    {
        "cv_texto": "Nombre: María González. Estudios: Secretaria Administrativa en Cruz Roja, Primeros Auxilios y Enfermería Básica. Segundo Curso de Enfermería. Universidad: Licenciada en Derecho. Experiencia en atención médica.",
        "talleres": "primeros auxilios, enfermería, cruz roja",
        "competencias": "Salud, Legal"
    },
    {
        "cv_texto": "Datos personales: Celular, Correo Electrónico. Estudios: Cruz Roja Boliviana - Primeros Auxilios y Enfermería Básica, Segundo Curso de Enfermería. Universidad: Licenciada en Derecho.",
        "talleres": "cruz roja boliviana, primeros auxilios, enfermería",
        "competencias": "Salud, Legal"
    },
    {
        "cv_texto": "Lugar de Nacimiento: Oruro. Estudios realizados: Escuela Mariano Baptista, Liceo de Señoritas, Instituto Superior de Comercio. Cruz Roja: Secretaria Administrativa, Primeros Auxilios y Enfermería Básica, Segundo Curso de Enfermería. Universidad: Licenciada en Derecho.",
        "talleres": "cruz roja, primeros auxilios, enfermería básica, derecho",
        "competencias": "Salud, Legal"
    },
    {
        "cv_texto": "Nacionalidad: Boliviana. Domicilio: Oruro. Estudios: Cruz Roja Boliviana - Primeros Auxilios y Enfermería Básica, Segundo Curso de Enfermería. Universidad Nacional de Siglo XX: Licenciada en Derecho.",
        "talleres": "cruz roja boliviana, enfermería, primeros auxilios",
        "competencias": "Salud, Legal"
    },
    {
        "cv_texto": "Celular, Correo Electrónico, Nombre y Apellidos. Estudios: Secretaria Administrativa en Cruz Roja Boliviana, Primeros Auxilios y Enfermería Básica, Segundo Curso de Enfermería. Universidad: Licenciada en Derecho.",
        "talleres": "cruz roja, primeros auxilios, enfermería básica",
        "competencias": "Salud, Legal"
    },
    {
        "cv_texto": "Celular Correo Electrónico Nombre y Apellidos Lugar de Nacimiento Nacionalidad Domicilio Estudios realizados Universitario: Escuela Mariano Baptista: Liceo de Señoritas Oruro: Instituto Superior de Comercio Secretaria Administrativa: Cruz Roja Boliviana Primeros Auxilios y Enfermería Básica: Segundo Curso de Enfermería Universidad Nacional de Siglo XX Licenciada en Derecho Universidad Mayor de San.",
        "talleres": "cruz roja boliviana, primeros auxilios, enfermería básica, enfermería",
        "competencias": "Salud, Legal"
    },
]

def enriquecer_dataset():
    """Agrega los nuevos registros al dataset existente."""
    # Rutas
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    dataset_original = os.path.join(data_dir, "dataset_competencias.csv")
    dataset_enriquecido = os.path.join(data_dir, "dataset_competencias_enriquecido.csv")
    
    # Cargar dataset original
    print(f"Leyendo dataset original: {dataset_original}")
    df_original = pd.read_csv(dataset_original, encoding='utf-8')
    print(f"   Registros originales: {len(df_original)}")
    
    # Crear DataFrame con nuevos registros
    df_nuevos = pd.DataFrame(nuevos_registros)
    print(f"   Nuevos registros a agregar: {len(df_nuevos)}")
    
    # Combinar
    df_final = pd.concat([df_original, df_nuevos], ignore_index=True)
    print(f"   Total de registros finales: {len(df_final)}")
    
    # Guardar
    df_final.to_csv(dataset_enriquecido, index=False, encoding='utf-8', quoting=1)
    print(f"OK - Dataset enriquecido guardado en: {dataset_enriquecido}")
    
    # Actualizar también el archivo original automáticamente
    df_final.to_csv(dataset_original, index=False, encoding='utf-8', quoting=1)
    print(f"OK - Dataset original actualizado: {dataset_original}")
    
    # Estadísticas
    print("\nEstadisticas del dataset enriquecido:")
    print(f"   - Total de registros: {len(df_final)}")
    print(f"   - Registros originales: {len(df_original)}")
    print(f"   - Registros nuevos: {len(df_nuevos)}")
    
    # Contar competencias
    todas_competencias = []
    for comp_str in df_final['competencias']:
        comps = [c.strip() for c in str(comp_str).split(',')]
        todas_competencias.extend(comps)
    
    from collections import Counter
    conteo = Counter(todas_competencias)
    print(f"\nTop 10 competencias mas frecuentes:")
    for comp, count in conteo.most_common(10):
        print(f"   - {comp}: {count}")

if __name__ == "__main__":
    enriquecer_dataset()

