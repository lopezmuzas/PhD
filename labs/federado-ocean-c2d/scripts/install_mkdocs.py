"""Instala este laboratorio en un proyecto MkDocs existente.

Hace tres cosas que a mano se olvidan:

  1. Copia los 9 documentos a  <docs>/<capitulo>/7.5-flower-ocean-c2d/
  2. Genera la pagina indice   <docs>/<capitulo>/7.5-flower-ocean-c2d.md
  3. REESCRIBE los enlaces relativos. Las subpaginas bajan un nivel, asi que
     `../../02-deep-learning/...` pasa a ser `../../../02-deep-learning/...`.
     Es lo que rompe `mkdocs build --strict` y cuesta media hora encontrar.

Uso:

    python scripts/install_mkdocs.py \
        --mkdocs-root ~/mi-proyecto \
        --chapter 03-aprendizaje-federado/07-federado-descentralizado

    # ensayo sin escribir nada
    python scripts/install_mkdocs.py --mkdocs-root ~/mi-proyecto \
        --chapter 03-x/07-y --dry-run

El codigo NO se copia a docs/: se queda en <mkdocs-root>/labs/flower-ocean-c2d/
y las paginas lo incluyen con pymdownx.snippets. Asi la documentacion no puede
desincronizarse del codigo.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SLUG = "06-federado-sobre-data-spaces"

# Ficheros y carpetas del repo que van a labs/ (todo menos la doc)
CODE_ITEMS = [
    "src", "tests", "examples", "scripts", "docker",
    "Makefile", "pyproject.toml", "requirements.txt",
    "requirements-orchestrator.txt", "requirements-dev.txt", ".dockerignore",
]


def rewrite_links(text: str) -> str:
    """Ajusta los enlaces de las subpaginas, que bajan un nivel.

    - `../../02-deep-learning/x.md`  ->  `../../../02-deep-learning/x.md`
    - `7.3-offline-....md`           ->  `../7.3-offline-....md`
    Los enlaces entre subpaginas (`01-lab0-simulacion.md`) NO se tocan:
    siguen siendo hermanos.
    """
    # 1. subir un nivel los `../` que salen del capitulo
    text = re.sub(r"\]\((\.\./\.\./)", r"](../../../", text)

    # 2. los hermanos del capitulo (7.1, 7.2, 7.3...) quedan un nivel arriba
    text = re.sub(r"\]\((7\.\d+[^)/]*\.md)", r"](../\1)", text)

    return text


def build_hub(chapter_depth: int) -> str:
    """Genera la pagina indice a partir del README, ajustando rutas."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    # Los enlaces del README apuntan a docs/XX.md; en MkDocs van al subdirectorio
    hub = readme.replace("](docs/", f"]({SLUG}/")

    # El hub esta al mismo nivel que el 7.5 original: sus enlaces externos valen
    return hub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mkdocs-root", required=True,
                    help="carpeta que contiene mkdocs.yml")
    ap.add_argument("--chapter", required=True,
                    help="ruta del capitulo DENTRO de docs/, "
                         "ej. 03-federado/07-descentralizado")
    ap.add_argument("--docs-dir", default="docs")
    ap.add_argument("--labs-dir", default="labs",
                    help="donde va el codigo, FUERA de docs/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.mkdocs_root).expanduser().resolve()
    if not (root / "mkdocs.yml").exists():
        raise SystemExit(f"No encuentro mkdocs.yml en {root}")

    chapter = root / args.docs_dir / args.chapter
    pages = chapter / SLUG
    labs = root / args.labs_dir / "flower-ocean-c2d"

    planned: list[str] = []

    # --- 1. las 9 subpaginas -------------------------------------------------
    for src in sorted((REPO / "docs").glob("*.md")):
        dest = pages / src.name
        planned.append(f"  doc   {dest.relative_to(root)}")
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(rewrite_links(src.read_text(encoding="utf-8")),
                            encoding="utf-8")

    # --- 2. la pagina indice -------------------------------------------------
    hub_path = chapter / f"{SLUG}.md"
    planned.append(f"  hub   {hub_path.relative_to(root)}")
    if not args.dry_run:
        hub_path.parent.mkdir(parents=True, exist_ok=True)
        hub_path.write_text(build_hub(len(Path(args.chapter).parts)), encoding="utf-8")

    # --- 3. el codigo, FUERA de docs/ ---------------------------------------
    for item in CODE_ITEMS:
        src = REPO / item
        if not src.exists():
            continue
        dest = labs / item
        planned.append(f"  code  {dest.relative_to(root)}")
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, dest)

    print(("[ENSAYO] " if args.dry_run else "") + f"{len(planned)} elementos:")
    print("\n".join(planned))

    if not args.dry_run:
        print(f"\nAhora anade la navegacion a {root / 'mkdocs.yml'} "
              f"(ver docs/09-integracion-mkdocs.md) y ejecuta:")
        print(f"    cd {root} && mkdocs build --strict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
