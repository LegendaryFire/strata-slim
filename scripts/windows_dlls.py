#!/usr/bin/env python3
"""Copy the DLLs the Windows engine needs into its folder (scripts/build_windows.ps1 runs it).

    windows_dlls.py <engine folder> <search folder> [<search folder> ...]

Reads the import table of every .exe and .dll in the engine folder, and copies each imported DLL that is in a search
folder (the CUDA toolkit's bin, MSVC's redistributable folders) and not in the engine folder yet. A copied DLL can import
more, so it repeats until nothing new is found. What is in no search folder is Windows' own or the NVIDIA driver's
(nvcuda.dll), and is listed, not copied.
"""
from __future__ import annotations

import logging
import shutil
import struct
import sys
from pathlib import Path

log = logging.getLogger(__name__)


class PEError(Exception):
    """The file is not a 64-bit Windows program or library this script can read."""


def imported_dlls(path: Path) -> list[str]:
    """The DLL names in a PE32+ file's import table (normal and delay-loaded)."""
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise PEError(f"{path.name}: not a Windows program")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise PEError(f"{path.name}: no PE header")

    section_count = struct.unpack_from("<H", data, pe + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    optional = pe + 24
    if struct.unpack_from("<H", data, optional)[0] != 0x20B:
        raise PEError(f"{path.name}: not a 64-bit program")

    # data directories: 1 is the import table, 13 the delay-load import table
    directories = optional + 112
    import_rva = struct.unpack_from("<I", data, directories + 1 * 8)[0]
    delay_rva = struct.unpack_from("<I", data, directories + 13 * 8)[0]

    sections = []
    table = optional + optional_size
    for i in range(section_count):
        entry = table + i * 40
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from("<IIII", data, entry + 8)
        sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def offset(rva: int) -> int:
        for start, size, raw in sections:
            if start <= rva < start + size:
                return rva - start + raw
        raise PEError(f"{path.name}: address {rva:#x} is in no section")

    def name_at(rva: int) -> str:
        start = offset(rva)
        return data[start:data.index(b"\0", start)].decode("ascii", errors="replace")

    names = []
    if import_rva:
        entry = offset(import_rva)
        while True:
            name_rva = struct.unpack_from("<I", data, entry + 12)[0]    # IMAGE_IMPORT_DESCRIPTOR.Name, 20 bytes each
            if not name_rva:
                break
            names.append(name_at(name_rva))
            entry += 20
    if delay_rva:
        entry = offset(delay_rva)
        while True:
            name_rva = struct.unpack_from("<I", data, entry + 4)[0]     # IMAGE_DELAYLOAD_DESCRIPTOR.DllNameRVA, 32 bytes each
            if not name_rva:
                break
            names.append(name_at(name_rva))
            entry += 32
    return names


def find(name: str, search: list[Path]) -> Path | None:
    """The DLL in the first search folder that has it, matched without regard to case as Windows does."""
    for folder in search:
        for candidate in folder.iterdir():
            if candidate.name.lower() == name.lower() and candidate.is_file():
                return candidate
    return None


def bundle(engine: Path, search: list[Path]) -> None:
    left_out = set()
    while True:
        present = {f.name.lower() for f in engine.iterdir()}
        wanted = set()
        for binary in sorted(engine.iterdir()):
            if binary.suffix.lower() in (".exe", ".dll"):
                for name in imported_dlls(binary):
                    wanted.add(name)

        copied = False
        for name in sorted(wanted, key=str.lower):
            if name.lower() in present or name.lower() in left_out:
                continue
            found = find(name, search)
            if found is None:
                left_out.add(name.lower())
                continue
            shutil.copy2(found, Path(engine, found.name))
            log.info(f"bundled {found.name} (from {found.parent})")
            copied = True
        if not copied:
            break

    log.info(f"not bundled, expected from Windows or the NVIDIA driver: {', '.join(sorted(left_out))}")


if __name__ == "__main__":
    logging.basicConfig(format="[dlls] %(levelname)s %(message)s", level="INFO")
    if len(sys.argv) < 3:
        sys.exit("usage: windows_dlls.py <engine folder> <search folder> [<search folder> ...]")
    folders = [Path(a) for a in sys.argv[2:] if Path(a).is_dir()]
    try:
        bundle(Path(sys.argv[1]), folders)
    except PEError as e:
        log.error(e)
        sys.exit(1)
