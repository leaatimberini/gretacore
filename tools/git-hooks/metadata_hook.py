#!/usr/bin/env python3
"""
Git Metadata Hook para GRETA CORE
Agrega metadatos de autoría a documentos Markdown automáticamente.

Uso:
    python .git/metadata_hook.py [--dry-run]
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# Templates de metadatos
METADATA_TEMPLATE_EN = """
---
## Metadata
- **Author:** GRETA Core Team
- **Created:** {created_date}
- **Modified:** {modified_date}
- **Version:** {version}
---
"""

METADATA_TEMPLATE_ES = """
---
## Metadatos
- **Autor:** GRETA Core Team
- **Creado:** {created_date}
- **Modificado:** {modified_date}
- **Versión:** {version}
---
"""

def get_git_date(filepath):
    """Obtiene la fecha de creación/modificación desde git."""
    try:
        # Fecha de creación (primer commit que afectó el archivo)
        created = subprocess.run(
            ["git", "log", "--format=%ai", "--follow", "--reverse", filepath],
            capture_output=True, text=True
        )
        if created.returncode == 0 and created.stdout.strip():
            dates = created.stdout.strip().split('\n')
            return dates[0][:10]  # YYYY-MM-DD
    except Exception:
        pass
    return datetime.now().strftime("%Y-%m-%d")

def get_modified_date():
    """Obtiene la fecha de modificación actual."""
    return datetime.now().strftime("%Y-%m-%d")

def process_file(filepath):
    """Procesa un archivo individual."""
    created_date = get_git_date(filepath)
    modified_date = get_modified_date()
    
    # Determinar el template según el idioma del path
    if '/es/' in str(filepath) or '_ES.md' in str(filepath):
        metadata = METADATA_TEMPLATE_ES.format(
            created_date=created_date,
            modified_date=modified_date,
            version="0.1.0"
        )
    else:
        metadata = METADATA_TEMPLATE_EN.format(
            created_date=created_date,
            modified_date=modified_date,
            version="0.1.0"
        )
    
    return metadata

def main():
    """Main entry point."""
    dry_run = "--dry-run" in sys.argv
    
    # Obtener archivos modificados/nuevos
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
            capture_output=True, text=True
        )
        files = [f for f in result.stdout.strip().split('\n') if f.endswith('.md')]
    except Exception as e:
        print(f"Error getting staged files: {e}")
        sys.exit(1)
    
    if not files:
        print("No Markdown files to process.")
        return
    
    print(f"Processing {len(files)} Markdown files...")
    
    for filepath in files:
        if dry_run:
            print(f"  [DRY-RUN] Would process: {filepath}")
        else:
            print(f"  Processing: {filepath}")
            metadata = process_file(filepath)
            
            # Agregar metadatos al final del archivo
            try:
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write(metadata)
                # Marcar el archivo como modificado
                subprocess.run(["git", "add", filepath], check=False)
            except Exception as e:
                print(f"    Error processing {filepath}: {e}")
    
    if dry_run:
        print("\n[DRY-RUN] No changes made.")

if __name__ == "__main__":
    main()
