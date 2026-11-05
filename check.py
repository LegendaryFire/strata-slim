#!/usr/bin/env python3
"""Check that vendor/ has everything the image needs.

    ./check.py        run after ./extract.sh; exits 1 if something is missing

It reads the Python source in vendor/ (it never imports or runs it) and follows the imports, starting from
the programs the image runs: the server and the model prep tools. Then it reports:

  1. Upstream modules that are imported but were left out by manifest.txt.
       - imported when a program starts   -> ERROR   (the program would crash immediately)
       - imported only inside a function  -> warning (only that code path would fail)
  2. Third-party packages imported when a program starts that are not in runtime-packages.txt -> ERROR
  3. The config keys the server reads, written to vendor/.config-keys (launch.py warns about unknown keys)
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"

# The programs the image runs. Each one is a starting point for following imports.
PROGRAMS = [
    "serve/server.py",
    "tools/iq_pack.py",
    "tools/mtp_fetch.py",
    "tools/mtp_pack.py",
    "tools/mtp_rt.py",
    "tools/strata_tokenizer.py",
    "tools/dflash2_pack.py",                    # loaded by drafting_fetch to pack the DFlash2 download
    "strata_control/drafting_fetch.py",         # prepare.py calls it to get the DFlash2 drafter
]

# Where imports are looked up: the server and the tools both put the repo root and tools/ on sys.path.
SEARCH_DIRS = ["", "tools/"]

# Import names whose pip package has a different name.
PIP_NAME = {"PIL": "pillow", "yaml": "pyyaml"}

# Imports that the image provides without pip.
NOT_FROM_PIP = {"gguf"}          # llama.cpp's gguf-py, copied into the image


# ---- where modules live -----------------------------------------------------------------------------------------

class Modules:
    """Answers two questions about a module name such as "strata_control.procs":
    is it in vendor/ (and which file), and is it upstream code at all (in vendor/ or not)."""

    def __init__(self, upstream_files: set[str]):
        self.upstream_files = upstream_files      # every .py path in the upstream tag, from extract.sh

    @staticmethod
    def candidates(module: str) -> list[str]:
        """The paths "import module" could load, e.g. strata_control/procs.py or strata_control/procs/__init__.py."""
        path = module.replace(".", "/")
        return [f"{d}{path}.py" for d in SEARCH_DIRS] + [f"{d}{path}/__init__.py" for d in SEARCH_DIRS]

    def in_vendor(self, module: str) -> Path | None:
        for rel in self.candidates(module):
            if (VENDOR / rel).is_file():
                return VENDOR / rel
        return None

    def is_upstream(self, module: str) -> bool:
        return any(rel in self.upstream_files for rel in self.candidates(module))


# ---- reading imports --------------------------------------------------------------------------------------------

@dataclass
class Import:
    module: str          # e.g. "strata_control.procs"
    at_startup: bool     # False when the import sits inside a function
    optional: bool       # True when it is wrapped in try/except ImportError


class ImportFinder(ast.NodeVisitor):
    """Collects the imports of one file, noting whether each runs at startup and whether it is optional."""

    def __init__(self, modules: Modules, package: str):
        self.modules = modules
        self.package = package               # the file's package, e.g. "serve", for "from . import x"
        self.imports: list[Import] = []
        self.function_depth = 0              # > 0: inside a def or lambda
        self.optional_depth = 0              # > 0: inside "try:" whose except catches ImportError

    def visit_FunctionDef(self, node):
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_Try(self, node):
        optional = any(catches_import_error(h) for h in node.handlers)
        self.optional_depth += optional
        for child in node.body:
            self.visit(child)
        self.optional_depth -= optional
        for child in node.handlers + node.orelse + node.finalbody:
            self.visit(child)

    def visit_Import(self, node):
        # "import a.b.c"
        for alias in node.names:
            self.add(alias.name)

    def visit_ImportFrom(self, node):
        # "from a.b import c" or "from . import c"
        module = self.absolute(node.module, node.level)
        self.add(module)
        for alias in node.names:
            # "c" is either a name defined in a.b, or a submodule a/b/c.py. Only a submodule is an import.
            sub = f"{module}.{alias.name}" if module else alias.name
            if self.modules.in_vendor(sub) or self.modules.is_upstream(sub):
                self.add(sub)

    def absolute(self, module: str | None, level: int) -> str:
        """Turn a relative import ("from ..x import y" inside package a.b) into an absolute module name."""
        if level == 0:
            return module or ""
        parts = self.package.split(".") if self.package else []
        base = parts[: len(parts) - level + 1]
        return ".".join(base + ([module] if module else []))

    def add(self, module: str):
        self.imports.append(Import(module, at_startup=self.function_depth == 0, optional=self.optional_depth > 0))


def catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:                 # bare "except:"
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(getattr(t, "id", None) in ("ImportError", "ModuleNotFoundError", "Exception") for t in types)


def imports_of(path: Path, modules: Modules) -> list[Import]:
    rel = path.relative_to(VENDOR)
    package = ".".join(rel.parent.parts) if rel.parts[0] != "tools" else ""   # tools/ files are top-level modules
    finder = ImportFinder(modules, package)
    finder.visit(ast.parse(path.read_text(encoding="utf-8", errors="replace")))
    return finder.imports


# ---- following imports ------------------------------------------------------------------------------------------

@dataclass
class Findings:
    files_loaded: set[Path] = field(default_factory=set)
    missing_at_startup: list[str] = field(default_factory=list)            # "file: imports module"
    missing_later: dict[str, set[str]] = field(default_factory=dict)       # module -> files importing it in a function
    packages: dict[str, set[str]] = field(default_factory=dict)            # pip package -> files needing it at startup
    optional_packages: set[str] = field(default_factory=set)


def follow_imports(programs: list[Path], modules: Modules) -> Findings:
    """Starting from the programs, follow every import that runs at startup."""
    stdlib = set(sys.stdlib_module_names)
    found = Findings()
    queue = list(programs)
    while queue:
        path = queue.pop()
        if path in found.files_loaded:
            continue
        found.files_loaded.add(path)
        name = str(path.relative_to(VENDOR))

        for imp in imports_of(path, modules):
            top = imp.module.split(".")[0]
            if not top or top in stdlib or top == "__future__":
                continue

            vendor_file = modules.in_vendor(imp.module)
            if vendor_file:                             # upstream code we have: follow its imports too
                if imp.at_startup:
                    queue.append(vendor_file)
            elif modules.is_upstream(imp.module):       # upstream code that manifest.txt left out
                if imp.at_startup and not imp.optional:
                    found.missing_at_startup.append(f"{name}: imports {imp.module}")
                else:
                    found.missing_later.setdefault(imp.module, set()).add(name)
            elif imp.at_startup and top not in NOT_FROM_PIP and not modules.is_upstream(top):
                package = PIP_NAME.get(top, top).lower()  # a third-party package
                if imp.optional:
                    found.optional_packages.add(package)
                else:
                    found.packages.setdefault(package, set()).add(name)
    return found


# ---- config keys ------------------------------------------------------------------------------------------------

def config_keys() -> set[str]:
    """Every key the server reads from its run config: cfg.get("key") and cfg["key"] in serve/*.py."""
    pattern = re.compile(r'cfg(?:\.get\(|\[)"([a-z_][a-z0-9_]*)"')
    keys = set()
    for path in (VENDOR / "serve").glob("*.py"):
        keys |= set(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))
    return keys


# ---- main -------------------------------------------------------------------------------------------------------

def read_list(path: Path) -> set[str]:
    """The non-empty lines of a file, without # comments."""
    return {line.split("#")[0].strip() for line in path.read_text().splitlines()} - {""}


def main() -> int:
    if not (VENDOR / ".upstream-files").is_file():
        print("vendor/ is empty or old: run ./extract.sh <tag> first", file=sys.stderr)
        return 1
    modules = Modules(read_list(VENDOR / ".upstream-files"))
    errors: list[str] = []

    programs = []
    for program in PROGRAMS:
        if (VENDOR / program).is_file():
            programs.append(VENDOR / program)
        else:
            errors.append(f"{program} is not in vendor/ (add it to manifest.txt)")
    found = follow_imports(programs, modules)

    # 1. upstream modules
    for problem in found.missing_at_startup:
        errors.append(f"{problem}, which is not in vendor/ (add it to manifest.txt)")

    # 2. third-party packages
    installed = {name.lower() for name in read_list(HERE / "runtime-packages.txt")}
    for package, files in sorted(found.packages.items()):
        if package not in installed:
            errors.append(f"package {package} is needed at startup by {', '.join(sorted(files)[:3])}, "
                          "but is not in runtime-packages.txt")

    # 3. config keys
    keys = config_keys()
    (VENDOR / ".config-keys").write_text("\n".join(sorted(keys)) + "\n")

    print(f"[check] {len(found.files_loaded)} modules load when the {len(programs)} programs start")
    print(f"[check] third-party packages needed at startup: {', '.join(sorted(found.packages)) or 'none'}"
          + (f" (optional: {', '.join(sorted(found.optional_packages))})" if found.optional_packages else ""))
    print(f"[check] the server reads {len(keys)} config keys (saved to vendor/.config-keys)")
    if found.missing_later:
        print(f"[check] warning: {len(found.missing_later)} upstream modules are imported only inside functions and "
              "are not in vendor/. That is fine unless you use those code paths:")
        for module, files in sorted(found.missing_later.items()):
            print(f"          {module}  (from {', '.join(sorted(files)[:2])})")
    for error in errors:
        print(f"[check] ERROR: {error}", file=sys.stderr)
    if not errors:
        print("[check] OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
