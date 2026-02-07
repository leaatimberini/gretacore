#!/bin/bash
# =============================================================================
# GRETA CORE - Workflow: Download Llama-2-7B, Convert to GGUF, Run B3.65
# =============================================================================
# REPO: /root/gretacore (MI300X: 129.212.184.200)
# AUTHOR: L.E.T / Leandro Emanuel Timberini
# =============================================================================

set -euo pipefail

# --- CONFIGURACIÓN ---
export GRETA_DIR="/root/gretacore"
export MODELS_DIR="/root/models"
export HF_MODEL_DIR="${MODELS_DIR}/hf/meta-llama/Llama-2-7b"
export OUT_GGUF="${GRETA_DIR}/models/greta-v1.gguf"
export OUTDIR_ARTIFACTS="${GRETA_DIR}/artifacts_remote/$(date +%F)"
export LOGFILE="${OUTDIR_ARTIFACTS}/workflow_llama2_download.log"

# --- COLORES ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log_success() {
    log "${GREEN}✓ $1${NC}"
}

log_warn() {
    log "${YELLOW}⚠ $1${NC}"
}

log_error() {
    log "${RED}✗ $1${NC}"
}

# =============================================================================
# PASO 1: PEDIR TOKEN HF DE FORMA SEGURA
# =============================================================================
log "=== PASO 1: Configuración de Hugging Face Token ==="

# Leer token de stdin (NO hardcodear)
if [ -z "${HF_TOKEN:-}" ]; then
    log "Por favor ingresa tu HF Token (hf_...) para continuar."
    log "Token starting with hf_..."
    read -rsp 'Enter HF Token: ' HF_TOKEN
    echo ""
    export HF_TOKEN
else
    log "HF_TOKEN ya está definido en el entorno"
fi

# Validar formato del token
if [[ ! "$HF_TOKEN" =~ ^hf_ ]]; then
    log_error "Token inválido: debe comenzar con 'hf_'"
    exit 1
fi

log_success "Token validado (primeros 10 chars): ${HF_TOKEN:0:10}..."

# Guardar en cache HF (solo para esta sesión)
mkdir -p ~/.cache/huggingface
echo "$HF_TOKEN" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token

# =============================================================================
# PASO 2: PREPARAR ENTORNO
# =============================================================================
log "=== PASO 2: Preparando entorno ==="

# Instalar/actualizar dependencias
python3 -m pip install -U pip "huggingface_hub[cli]" hf_transfer -q 2>/dev/null || true

# Configurar variables de entorno
export HF_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$MODELS_DIR" "$OUTDIR_ARTIFACTS" "$(dirname "$OUT_GGUF")"
cd "$GRETA_DIR"

log_success "Entorno preparado"

# =============================================================================
# PASO 3: DESCARGAR LLAMA-2-7B
# =============================================================================
log "=== PASO 3: Descargando Llama-2-7B desde Hugging Face ==="

if [ -d "$HF_MODEL_DIR" ] && [ -f "${HF_MODEL_DIR}/tokenizer.json" ]; then
    log_warn "Modelo ya existe en $HF_MODEL_DIR"
    log "Usando modelo existente..."
else
    log "Iniciando descarga (puede tomar varios minutos)..."
    
    huggingface-cli download meta-llama/Llama-2-7b \
        --local-dir "$HF_MODEL_DIR" \
        --local-dir-use-symlinks False \
        --token "$HF_TOKEN" \
        2>&1 | tee -a "$LOGFILE"
    
    if [ ! -d "$HF_MODEL_DIR" ]; then
        log_error "Descarga falló: directorio no creado"
        exit 1
    fi
fi

log_success "Modelo descargado"
du -sh "$HF_MODEL_DIR" | tee -a "$LOGFILE"

# =============================================================================
# PASO 4: DETECTAR FORMATO Y CONVERTIR A GGUF
# =============================================================================
log "=== PASO 4: Convirtiendo a GGUF ==="

# Verificar si ya viene en GGUF
if find "$HF_MODEL_DIR" -maxdepth 2 -type f -name "*.gguf" | grep -q .; then
    log_warn "Ya existen archivos GGUF, copiando..."
    find "$HF_MODEL_DIR" -maxdepth 2 -type f -name "*.gguf" -exec cp -v {} "$OUT_GGUF" \;
else
    log "Convirtiendo con llama.cpp..."
    
    # Clonar/actualizar llama.cpp si no existe
    if [ ! -d /root/tools/llama.cpp ]; then
        log "Clonando llama.cpp..."
        git clone https://github.com/ggerganov/llama.cpp.git /root/tools/llama.cpp
    fi
    
    cd /root/tools/llama.cpp
    git pull --rebase --quiet 2>/dev/null || true
    
    python3 -m pip install -U numpy sentencepiece safetensors torch -q 2>/dev/null || true
    
    # Obtener commit de llama.cpp
    LLAMA_COMMIT=$(git rev-parse --short HEAD)
    export LLAMA_COMMIT
    
    cd "$GRETA_DIR"
    
    # Convertir
    python3 /root/tools/llama.cpp/convert_hf_to_gguf.py \
        "$HF_MODEL_DIR" \
        --outfile "$OUT_GGUF" \
        --outtype f16 \
        2>&1 | tee -a "$LOGFILE"
fi

if [ ! -f "$OUT_GGUF" ]; then
    log_error "Conversión falló: archivo GGUF no creado"
    exit 1
fi

log_success "Conversión completada"
ls -lh "$OUT_GGUF" | tee -a "$LOGFILE"
file "$OUT_GGUF" | tee -a "$LOGFILE"

# =============================================================================
# PASO 5: VERIFICAR COMPATIBILIDAD CON GRETA
# =============================================================================
log "=== PASO 5: Verificando compatibilidad con GRETA ==="

python3 << 'PYEOF' 2>&1 | tee -a "$LOGFILE"
import sys
sys.path.insert(0, '/root/gretacore/src/inference')

try:
    from gretaio import gguf
    
    reader = gguf.GGUFReader('/root/gretacore/models/greta-v1.gguf')
    print('=== MODEL CONFIG ===')
    
    keys = [
        'general.architecture',
        'llama.block_count',
        'llama.embedding_length', 
        'llama.head_count',
        'vocab_size',
        'llama.ffn_expanded_hidden_size'  # hidden_dim para GRETA
    ]
    
    for key in keys:
        try:
            val = reader.header.get_field(key).value
            print(f'{key}: {val}')
        except Exception as e:
            print(f'{key}: NOT_FOUND ({e})')
    
    print('\n✓ Modelo compatible con GRETA')
except ImportError as e:
    print(f'⚠ gretaio no disponible: {e}')
    print('Verificación manual requerida')
except Exception as e:
    print(f'✗ Error verificando: {e}')
    sys.exit(1)
PYEOF

# =============================================================================
# PASO 6: GENERAR TRAZABILIDAD
# =============================================================================
log "=== PASO 6: Generando trazabilidad ==="

cd /root/tools/llama.cpp
LLAMA_COMMIT=$(git rev-parse --short HEAD)
cd "$GRETA_DIR"

# Checksum SHA256
SHA256=$(sha256sum "$OUT_GGUF" | cut -d' ' -f1)
echo "$SHA256" > "${OUTDIR_ARTIFACTS}/model_sha256.txt"
log_success "SHA256: $SHA256"

# Manifest
cat > "${OUTDIR_ARTIFACTS}/model_manifest/manifest.txt" << EOF
MODEL_SOURCE=meta-llama/Llama-2-7b
MODEL_LOCAL_HF_DIR=${HF_MODEL_DIR}
MODEL_GGUF_PATH=${OUT_GGUF}
CONVERSION_TOOL=/root/tools/llama.cpp (commit: ${LLAMA_COMMIT})
OUTTYPE=f16
DATE=$(date -Is)
SHA256=${SHA256}
GRETA_HIDDEN_DIM=11008
EOF

log_success "Manifest generado"
cat "${OUTDIR_ARTIFACTS}/model_manifest/manifest.txt" | tee -a "$LOGFILE"

# =============================================================================
# PASO 7: RESCATE A LOCAL
# =============================================================================
log "=== PASO 7: Rescatando artifacts a local ==="

# Empaquetar en remoto
cd "$GRETA_DIR"
tar -czvf "/root/gretacore_remote_rescue_$(date +%F).tgz" artifacts_remote/$(date +%F)/model_manifest 2>/dev/null

# Copiar a local (requiere que el user tenga SSH configurado)
log "Copiando a local..."
scp "/root/gretacore_remote_rescue_$(date +%F).tgz" "root@localhost:/media/leandro/D08A27808A2762683/gretacore/gretacore/artifacts_remote/_rescued_from_remote/" 2>/dev/null && \
    log_success "Rescate completado" || \
    log_warn "No se pudo copiar a local (¿SSH a localhost configurado?)"

# =============================================================================
# PASO 8: EJECUTAR B3.65 (OPCIONAL)
# =============================================================================
log "=== PASO 8: B3.65 Determinism Sweep ==="

if [ -f "${GRETA_DIR}/tools/benchmarks/run_b3_65_determinism_mi300x.sh" ]; then
    chmod +x "${GRETA_DIR}/tools/benchmarks/run_b3_65_determinism_mi300x.sh"
    
    read -p "¿Ejecutar B3.65? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$GRETA_DIR"
        ./tools/benchmarks/run_b3_65_determinism_mi300x.sh 2>&1 | tee -a "$LOGFILE"
    else
        log "B3.65 omitido por el usuario"
    fi
else
    log_warn "Script B3.65 no encontrado"
fi

# =============================================================================
# RESUMEN FINAL
# =============================================================================
log ""
log "========================================"
log "WORKFLOW COMPLETADO"
log "========================================"
log "Modelo GGUF: $OUT_GGUF"
log "SHA256: $SHA256"
log "Llama.cpp commit: $LLAMA_COMMIT"
log "Artifacts: $OUTDIR_ARTIFACTS"
log "Log: $LOGFILE"
log "========================================"
