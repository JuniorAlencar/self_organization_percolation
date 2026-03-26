#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Set

PROJECT_ROOT = Path(".").resolve()
SCAN_DIRS = ["tools", "jupyter", "python"]
OUTPUT_FILE = "requirements.txt"

# imports locais começam com src
LOCAL_NAMESPACE = "src"

# mapeamento import -> nome do pacote no pip
IMPORT_TO_PIP = {
    "yaml": "PyYAML",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "Crypto": "pycryptodome",
    "fitz": "PyMuPDF",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "googleapiclient": "google-api-python-client",
    "serial": "pyserial",
}

MANUAL_IGNORE = {
    "__future__",
}

INCLUDE_VERSIONS = False


def get_stdlib_modules() -> Set[str]:
    stdlib = set(sys.builtin_module_names)

    if hasattr(sys, "stdlib_module_names"):
        stdlib |= set(sys.stdlib_module_names)

    stdlib |= {
        "os", "sys", "math", "json", "re", "pathlib", "typing", "itertools",
        "functools", "collections", "subprocess", "shutil", "glob", "time",
        "datetime", "random", "statistics", "csv", "pickle", "tempfile",
        "threading", "multiprocessing", "logging", "argparse", "unittest",
        "ast", "hashlib", "base64", "fractions", "decimal", "dataclasses",
        "inspect", "textwrap", "io", "enum", "copy", "heapq", "bisect",
        "traceback", "warnings", "sqlite3", "gzip", "bz2", "lzma", "zipfile",
        "tarfile", "configparser", "urllib", "http", "email", "socket",
        "struct", "signal", "platform", "pprint",
    }
    return stdlib


def extract_imports_from_code(code: str) -> Set[str]:
    imports = set()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                full_name = alias.name.strip()

                # ignora imports locais do tipo import src.xxx
                if full_name == LOCAL_NAMESPACE or full_name.startswith(f"{LOCAL_NAMESPACE}."):
                    continue

                root = full_name.split(".")[0]
                imports.add(root)

        elif isinstance(node, ast.ImportFrom):
            # ignora import relativo
            if node.level and node.module is None:
                continue

            if node.module:
                module_name = node.module.strip()

                # ignora imports locais do tipo from src.xxx import y
                if module_name == LOCAL_NAMESPACE or module_name.startswith(f"{LOCAL_NAMESPACE}."):
                    continue

                root = module_name.split(".")[0]
                imports.add(root)

    return imports


def extract_imports_from_py_file(path: Path) -> Set[str]:
    try:
        code = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        code = path.read_text(encoding="latin-1")
    return extract_imports_from_code(code)


def extract_imports_from_ipynb(path: Path) -> Set[str]:
    imports = set()

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return imports

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        code = "".join(source) if isinstance(source, list) else str(source)
        imports |= extract_imports_from_code(code)

    return imports


def normalize_package_name(import_name: str) -> str:
    return IMPORT_TO_PIP.get(import_name, import_name)


def get_installed_version(package_name: str) -> str | None:
    try:
        from importlib.metadata import version
    except ImportError:
        return None

    try:
        return version(package_name)
    except Exception:
        return None


def main() -> None:
    stdlib_modules = get_stdlib_modules()
    found_imports = set()

    for folder in SCAN_DIRS:
        base = PROJECT_ROOT / folder
        if not base.exists():
            continue

        for path in base.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix == ".py":
                found_imports |= extract_imports_from_py_file(path)

            elif path.suffix == ".ipynb":
                found_imports |= extract_imports_from_ipynb(path)

    filtered = set()
    for name in found_imports:
        if not name:
            continue
        if name in MANUAL_IGNORE:
            continue
        if name in stdlib_modules:
            continue
        filtered.add(name)

    packages = sorted(normalize_package_name(name) for name in filtered)

    lines = []
    for pkg in packages:
        if INCLUDE_VERSIONS:
            ver = get_installed_version(pkg)
            lines.append(f"{pkg}=={ver}" if ver else pkg)
        else:
            lines.append(pkg)

    output_path = PROJECT_ROOT / OUTPUT_FILE
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Arquivo gerado: {output_path}")
    print(f"Total de pacotes: {len(lines)}")
    print("\nPacotes encontrados:")
    for pkg in lines:
        print(f" - {pkg}")


if __name__ == "__main__":
    main()