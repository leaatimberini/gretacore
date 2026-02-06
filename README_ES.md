# GRETA CORE

**Estado**: Fase 3 - Serie de Auditorías B3.xx (Hasta B3.60)

---

## Autoría y Liderazgo

**GRETA CORE** es un proyecto de ingeniería independiente concebido, fundado y liderado por:

- **Leandro Emanuel Timberini**
  - Fundador y Arquitecto Principal de Sistemas
  - Todas las decisiones arquitectónicas se originan en esta autoría
  - Visión a largo plazo y principios fundacionales definidos por el fundador

---

## Descripción del Proyecto

GRETA CORE es una iniciativa de ingeniería a largo plazo enfocada en la construcción de un **stack de cómputo mínimo, de alto rendimiento y de estilo CUDA para hardware AMD**, diseñado específicamente para Modelos de Lenguaje de Gran Escala (LLMs).

El proyecto existe para romper el lock-in actual de CUDA atacando el problema en su raíz: **el software**.

---

## Progreso Fase 3 (Serie de Auditorías B3.xx)

| Hito | Estado | Descripción |
|------|--------|-------------|
| B3.52 | ✅ PASS | Fix de direccionamiento KV cache |
| B3.55-B3.58 | ✅ PASS | Aislamiento de causa raíz (RoPE/Q-proj/RMSNorm) |
| B3.59 | ✅ PASS | Auditoría Embedding + StageDebugInput |
| B3.60 | ✅ PASS | Bisect Attention Block (pipeline Layer0 verificado) |

**Documentación**:
- [Índice de Progreso](docs/PROGRESS.md)
- [Índice de Reportes AMD](docs/AMD/INDEX.md)

---

## Motivación

El ecosistema moderno de inteligencia artificial está dominado por una única plataforma de cómputo. Esta dominancia ha creado:

- Barreras de entrada artificiales
- Costos de hardware inflados
- Innovación limitada

GRETA CORE aborda este problema desde una perspectiva **software-first**, buscando liberar todo el potencial del hardware AMD mediante un stack de cómputo enfocado y orientado al rendimiento.

---

## Filosofía

Todos los principios se originan en la visión del fundador:

- **Software por sobre hardware** - Controlar el stack, no solo el silicio
- **Control total del stack** - Desde el kernel hasta la inferencia
- **Minimalismo sobre bloat** - Cada línea debe justificar su existencia
- **Rendimiento por sobre abstracción** - Abstracciones de costo cero únicamente
- **Disciplina de ingeniería a largo plazo** - Décadas, no trimestres

---

## Qué es GRETA CORE

- Un runtime de cómputo personalizado para hardware AMD (MI300X, MI200, RDNA)
- Un stack de ejecución LLM kernel-first
- Una experiencia de desarrollo tipo CUDA sin replicar CUDA
- Una iniciativa de investigación e ingeniería a largo plazo
- Una instalación que incluye torch, triton y jax (sin instalaciones extra)

---

## Qué NO es GRETA CORE

- No es un fork de CUDA
- No es un wrapper delgado sobre frameworks existentes
- No es una plataforma de cómputo GPU de propósito general
- No es un proyecto de optimización a corto plazo

---

## Destacados de Arquitectura

### Stack de Runtime

```
src/rt/
├── allocator/      # Gestión de memoria
├── backend/       # Backends HIP, Vulkan
├── dispatch/      # Despacho de grafos
├── graph/         # Ejecución de grafos
├── stream/        # Gestión de streams
└── telemetry/     # Monitoreo de rendimiento
```

### Motor de Inferencia

```
src/inference/
├── block_scheduler/    # Scheduler a nivel de bloque
├── generator/          # Generación de tokens
├── layer_trace/        # Trazabilidad capa por capa
├── model_config/      # Configuración de modelos
├── stage_trace/       # Trazabilidad a nivel de etapa
├── tokenizer/         # Tokenización
├── trace/             # Trazabilidad general
└── weight_loader/     # Carga de pesos
```

---

## Hardware Soportado

- **AMD MI300X** - Objetivo principal de desarrollo
- **AMD MI200 series** - Soportado
- **AMD RDNA3+** - Compatible

---

## Estructura de Documentación

```
docs/
├── AMD/              # Reportes de auditoría específicos de AMD (serie B3.xx)
│   └── phases/       # Documentación de fases
├── en/              # Documentación en inglés
├── es/              # Documentación en español
├── strategy/        # Documentos de planificación estratégica
├── PROGRESS.md      # Seguimiento de progreso general
├── CHANGELOG.md     # Historial de versiones
└── WORKSPACE_RULES.md  # Guías de desarrollo
```

---

## Inicio Rápido

```bash
# Clonar el repositorio
git clone https://github.com/leaatimberini/gretacore.git
cd gretacore

# Ver documentación
cat docs/PROGRESS.md
cat docs/AMD/INDEX.md

# Ejecutar benchmarks
cd tools/benchmarks
./run_bench.py
```

---

## Contribuyendo

Esta es una **iniciativa de ingeniería a largo plazo** liderada por el fundador. Todas las contribuciones deben alinearse con la filosofía del proyecto de minimalismo, rendimiento y control total del stack.

**Áreas de enfoque**:
- Mejoras al código fuente
- Documentación técnica
- Benchmarks reproducibles
- Auditorías verificables

**Cambios no aceptados**:
- Bloat de funcionalidades
- Modificaciones no documentadas
- Cambios no relacionados con el motor de inferencia

---

## Licencia y Atribución

Todo el código, documentación y decisiones arquitectónicas son propiedad intelectual de **Leandro Emanuel Timberini** como Fundador y Arquitecto Principal de Sistemas.

---

## Contacto

- GitHub: [@leaatimberini](https://github.com/leaatimberini)
- Repositorio: https://github.com/leaatimberini/gretacore

---

*Última actualización: Febrero 2026*
*Versión: 0.1.0*
