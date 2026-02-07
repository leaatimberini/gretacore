# Hugging Face Token Setup for GRETA

## ⚠️ IMPORTANTE: Token Requerido

Para descargar Llama-2-7B desde Hugging Face, necesitas un HF Token.

## Opción 1: Token Temporal (para esta sesión)

```bash
# En el remoto MI300X:
read -rsp 'Enter HF Token (hf_...): ' HF_TOKEN
export HF_TOKEN="$HF_TOKEN"
echo "$HF_TOKEN" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
```

## Opción 2: Token desde archivo seguro

```bash
# Crear archivo con permisos restrictivos
echo "hf_xxxxxxxxxxxxxxxxxxxx" > ~/.hf_token
chmod 600 ~/.hf_token

# El workflow lo leerá automáticamente
```

## Opción 3: Variable de entorno (menos seguro, no recomendado)

```bash
export HF_TOKEN="hf_tu_token_aqui"
```

## Obtener tu HF Token

1. Ve a https://huggingface.co/settings/tokens
2. Crea un nuevo token (Read access es suficiente)
3. Copia el token (empieza con `hf_`)

## Verificar

```bash
cat ~/.cache/huggingface/token
# Debe mostrar: hf_...
```
