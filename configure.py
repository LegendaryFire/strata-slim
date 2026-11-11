#!/usr/bin/env python3
"""Print strata-slim's settings for this machine: its GPUs, the VRAM they already use, and its RAM.

Run it on the host (not in the container), before the first `docker compose up -d`. Every variable is printed set,
ready to paste into .env (or any environment):

    python3 configure.py                       # print the settings
    python3 configure.py --output .env         # and write them to .env (an existing one is kept unless --force)
    python3 configure.py --gpus 0,1 --vision cpu --context 256k

The VRAM each card already uses (the desktop, other programs) is measured first, before anything is loaded: that is
the baseline the engine will find beside it, and it decides the reserve and whether the card is small. A running
strata engine is not part of that baseline, so its own VRAM is left out of the measurement.

The rules are upstream setup's (Strata's setup.py), for the Qwen3.8-Flash-Next models strata-slim serves. Only the
standard library is used, so it runs with any Python 3.11+ without installing anything.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

# upstream setup's numbers
DEFAULT_CONTEXT = 131072                  # 128K on every card
SMALL_CARD_CONTEXT = 32768
KV_STREAMING_FROM = 65536                 # from 64K up the KV cache can live in RAM
KV_RESIDENT = 32768                       # the positions per layer kept in VRAM when it does
KV_CELL_BYTES = {"int8": 1056, "k8v4": 816, "q4": 576, "fp16": 2112}
LOW_RAM_HEADROOM_GB = 10                  # RAM beside the experts: the OS, the engine's buffers, the server
UNSLOTH_RESIDENT_BESIDE_GB = 16           # every expert in RAM when they fit with this much more
UNSLOTH_RAM_LEFT_GB = 24                  # RAM beside the budget, in the RAM-budget mode
MIN_RAM_BUDGET_GIB = 8
SMALL_CARD_GB = 7.5                       # a card under 8 GB
HEADLESS_RESERVE_MIB = 550                # example.env's: a card nothing else uses
DISPLAY_RESERVE_MIB = 1500                # an NVIDIA card that drives a display
DESKTOP_RESERVE_MIB = 3072                # an AMD card that drives a Linux desktop
VISION_GPU_RESERVE_MIB = 700              # the image encoder on the GPU
BUSY_BASELINE_MIB = 300                   # more than this already in use: something else lives on the card

ENGINE_ARGS_BASE = "--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5"
AMD_ARCHS = ("gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201")    # the RDNA3 / RDNA4 cards the rocm image runs
PCIE_GENS = {2.5: "1.0", 5.0: "2.0", 8.0: "3.0", 16.0: "4.0", 32.0: "5.0", 64.0: "6.0"}    # GT/s -> generation

BASELINE_SAMPLES = 5
BASELINE_INTERVAL_S = 0.4

GIB = 2 ** 30
GB_PER_GIB = GIB / 1e9


class ConfigureError(Exception):
    pass


# ---- the models -----------------------------------------------------------------------------------------------

# the Qwen3.8-Flash-Next repositories with GGUF files prepare.py can pack
ISTA_REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
UNSLOTH_REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
GYRO_REPO = "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"
MMPROJ_REPO = ISTA_REPO


class Quant(StrEnum):
    """The Qwen3.8-Flash-Next quantizations known here, by their name on the Hub."""
    # ISTA-DASLab's GSQ-RCO
    ISTA_Q2_0 = "Q2_0"
    ISTA_IQ2_XS = "IQ2_XS"
    ISTA_IQ3_XXS = "IQ3_XXS"
    ISTA_IQ3_S = "IQ3_S"
    # Unsloth's
    UD_IQ1_M = "UD-IQ1_M"
    UD_Q2_0 = "UD-Q2_0"
    UD_IQ2_XS = "UD-IQ2_XS"
    UD_Q2_K_XL = "UD-Q2_K_XL"
    UD_IQ3_XXS = "UD-IQ3_XXS"
    UD_Q3_K_XL = "UD-Q3_K_XL"
    UD_IQ4_XS = "UD-IQ4_XS"
    UD_Q4_K_XL = "UD-Q4_K_XL"
    UD_Q5_K_XL = "UD-Q5_K_XL"
    UD_Q6_K_XL = "UD-Q6_K_XL"
    UD_Q8_0 = "Q8_0"
    # AgentionAI's Gyro
    GYRO_S = "Gyro-S-TQ1_0"
    GYRO_M = "Gyro-M-TQ2_0"


DEFAULT_QUANT = Quant.ISTA_Q2_0


def find_quant(name: str) -> Quant | None:
    """The known quantization of this name, in any case, or None."""
    for quant in Quant:
        if quant.lower() == name.lower():
            return quant
    return None


@dataclass(frozen=True)
class Model:
    """A quantization's repository and sizes: upstream setup's MODELS[...] "ram_gb" and "arena_gb"."""
    repo: str
    ram_gb: float               # the RAM it needs (Gyro: VRAM and RAM together)
    arena_gb: float             # its routed experts
    ram_budget: bool = False    # below the RAM it needs: the hottest N GiB of experts in RAM, the rest from the SSD
    pack_args: str = ""


MODELS = {
    Quant.ISTA_Q2_0: Model(ISTA_REPO, ram_gb=48, arena_gb=34.0),
    Quant.ISTA_IQ2_XS: Model(ISTA_REPO, ram_gb=48, arena_gb=35.5),
    Quant.ISTA_IQ3_XXS: Model(ISTA_REPO, ram_gb=60, arena_gb=42.9),
    Quant.ISTA_IQ3_S: Model(ISTA_REPO, ram_gb=62, arena_gb=50.3),
    Quant.UD_IQ1_M: Model(UNSLOTH_REPO, ram_gb=48, arena_gb=34.8, ram_budget=True, pack_args="--compat-bf16"),
    Quant.UD_Q2_0: Model(UNSLOTH_REPO, ram_gb=48, arena_gb=37.4, ram_budget=True, pack_args="--compat-bf16"),
    Quant.UD_IQ2_XS: Model(UNSLOTH_REPO, ram_gb=48, arena_gb=42.3, ram_budget=True, pack_args="--compat-bf16"),
    Quant.UD_Q2_K_XL: Model(UNSLOTH_REPO, ram_gb=48, arena_gb=46.0, ram_budget=True, pack_args="--compat-bf16"),
    Quant.UD_IQ3_XXS: Model(UNSLOTH_REPO, ram_gb=48, arena_gb=50.3, ram_budget=True, pack_args="--compat-bf16"),
    Quant.UD_Q3_K_XL: Model(UNSLOTH_REPO, ram_gb=48, arena_gb=56.2, ram_budget=True, pack_args="--compat-bf16"),
    Quant.UD_IQ4_XS: Model(UNSLOTH_REPO, ram_gb=48, arena_gb=60.4, ram_budget=True, pack_args="--compat-bf16"),
    Quant.UD_Q4_K_XL: Model(UNSLOTH_REPO, ram_gb=48, arena_gb=77.0, ram_budget=True, pack_args="--compat-bf16"),
    Quant.UD_Q5_K_XL: Model(UNSLOTH_REPO, ram_gb=108, arena_gb=98.4, ram_budget=True, pack_args="--compat-bf16"),
    Quant.UD_Q6_K_XL: Model(UNSLOTH_REPO, ram_gb=118, arena_gb=109.2, ram_budget=True, pack_args="--compat-bf16"),
    Quant.UD_Q8_0: Model(UNSLOTH_REPO, ram_gb=136, arena_gb=128.3, ram_budget=True, pack_args="--compat-bf16"),
    Quant.GYRO_S: Model(GYRO_REPO, ram_gb=27.6, arena_gb=25.8, pack_args="--compat-bf16"),
    Quant.GYRO_M: Model(GYRO_REPO, ram_gb=35.0, arena_gb=32.1, pack_args="--compat-bf16"),
}


# ---- the hardware ---------------------------------------------------------------------------------------------

class Vendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"


@dataclass(frozen=True)
class PcieLink:
    """The link a card can run at in its slot, and what the card itself supports."""
    gen: str
    width: int
    card_gen: str
    card_width: int

    def __str__(self) -> str:
        text = f"PCIe {self.gen} x{self.width}"
        is_limited = (self.gen, self.width) != (self.card_gen, self.card_width)
        if is_limited:
            text += f" (card: {self.card_gen} x{self.card_width})"
        return text


@dataclass
class Gpu:
    index: int
    name: str
    vendor: Vendor
    arch: str                     # 86 (compute capability) or gfx1100
    total_mib: int
    pci: Path                     # the card's PCI device folder in sysfs
    link: PcieLink | None
    display: bool = False
    baseline_mib: int = 0         # in use before strata loads anything
    strata_mib: int = 0           # held by a running strata engine, left out of the baseline

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.baseline_mib

    @property
    def total_gb(self) -> float:
        return self.total_mib / 1024


@dataclass(frozen=True)
class Ram:
    total_gb: float
    available_gb: float           # free for new programs, a running strata engine's share counted as free
    strata_gb: float              # held by a running strata engine


class Hardware:
    """Reads what the machine reports (nvidia-smi, sysfs, /proc). Nothing is loaded or started."""

    @classmethod
    def gpus(cls) -> list[Gpu]:
        return cls.nvidia_gpus() + cls.amd_gpus()

    @classmethod
    def nvidia_gpus(cls) -> list[Gpu]:
        """Every NVIDIA GPU, numbered as nvidia-smi numbers them (the numbers GPUS takes)."""
        if not shutil.which("nvidia-smi"):
            return []
        found = []
        for index, name, total, cc, display, bus_id in cls.nvidia_query("index,name,memory.total,compute_cap,"
                                                                         "display_active,pci.bus_id"):
            pci = Path("/sys/bus/pci/devices", bus_id[-12:].lower())    # 00000000:07:00.0 -> 0000:07:00.0
            found.append(Gpu(int(index), name, Vendor.NVIDIA, cc.replace(".", ""), int(total), pci,
                             cls.pcie_link(pci), display=display.lower() == "enabled"))
        return found

    @classmethod
    def amd_gpus(cls, sysfs: Path = Path("/sys")) -> list[Gpu]:
        """AMD GPUs from the kernel's KFD topology, numbered as HIP numbers them (the GPU nodes in order)."""
        nodes = Path(sysfs, "class/kfd/kfd/topology/nodes")
        if not nodes.is_dir():
            return []
        # the nodes are folders named 0, 1, 2, ...: in number order, as HIP numbers the GPUs
        node_numbers = sorted(int(folder.name) for folder in nodes.iterdir() if folder.name.isdigit())
        found = []
        for number in node_numbers:
            node = Path(nodes, str(number))
            try:
                props = cls.read_pairs(Path(node, "properties"), " ")
                version = int(props.get("gfx_target_version") or 0)
                is_gpu = version != 0 and int(props.get("simd_count") or 0) != 0
                if not is_gpu:
                    continue
                arch = f"gfx{version // 10000}{(version // 100) % 100:x}{version % 100:x}"
                device = Path(sysfs, f"class/drm/renderD{props['drm_render_minor']}/device")
                total = int(Path(device, "mem_info_vram_total").read_text()) // 2 ** 20
            except (OSError, ValueError, KeyError):
                continue
            try:
                name = Path(device, "product_name").read_text().strip()
            except OSError:
                name = ""
            found.append(Gpu(len(found), name or f"AMD Radeon ({arch})", Vendor.AMD, arch, total, device,
                             cls.pcie_link(device)))
        return found

    @staticmethod
    def pcie_link(device: Path) -> PcieLink | None:
        """The card's link to its slot, from its maximum speed: the current one drops to 2.5 GT/s while it idles.
        An AMD card sits behind a PCIe switch on its own board, so its link is that switch's upstream port (the
        topmost device of the card's vendor). The slot's port caps what the card supports."""
        def read(folder: Path, name: str) -> str:
            return Path(folder, name).read_text().strip()

        def speed(folder: Path) -> float:
            return float(read(folder, "max_link_speed").split()[0])     # "16.0 GT/s PCIe"

        try:
            card = device.resolve()
            vendor = read(card, "vendor")
            while Path(card.parent, "vendor").is_file() and read(card.parent, "vendor") == vendor:
                card = card.parent
            card_speed, card_width = speed(card), int(read(card, "max_link_width"))
        except (OSError, ValueError, IndexError):
            return None

        slot_speed, slot_width = card_speed, card_width
        try:
            slot_speed, slot_width = speed(card.parent), int(read(card.parent, "max_link_width"))
        except (OSError, ValueError, IndexError):
            pass    # a card on the root complex: no port above it

        link_speed = min(card_speed, slot_speed)
        if link_speed not in PCIE_GENS or card_speed not in PCIE_GENS:
            return None
        return PcieLink(PCIE_GENS[link_speed], min(card_width, slot_width), PCIE_GENS[card_speed], card_width)

    @classmethod
    def measure_baseline(cls, gpus: list[Gpu]) -> None:
        """The VRAM each card already uses, before strata loads anything: the highest of a few samples, so a desktop
        that briefly needs more is counted. A running strata engine's own VRAM is left out (NVIDIA lists it per
        process; AMD cannot)."""
        has_nvidia = any(g.vendor == Vendor.NVIDIA for g in gpus)
        for _ in range(BASELINE_SAMPLES):
            nvidia = cls.nvidia_vram_used() if has_nvidia else {}
            for gpu in gpus:
                if gpu.vendor == Vendor.NVIDIA:
                    used, strata = nvidia.get(gpu.index, (0, 0))
                else:
                    used, strata = cls.amd_vram_used(gpu), 0
                gpu.baseline_mib = max(gpu.baseline_mib, used - strata)
                gpu.strata_mib = max(gpu.strata_mib, strata)
            time.sleep(BASELINE_INTERVAL_S)

    @classmethod
    def nvidia_vram_used(cls) -> dict[int, tuple[int, int]]:
        """Per card: (MiB in use, the part of it a strata engine holds)."""
        index_of = {}
        used = {}
        for index, uuid, mib in cls.nvidia_query("index,uuid,memory.used"):
            index_of[uuid] = int(index)
            used[int(index)] = int(mib)

        strata = dict.fromkeys(used, 0)
        for uuid, process, mib in cls.nvidia_query("gpu_uuid,process_name,used_memory", "compute-apps"):
            is_strata = Path(process).name.startswith("strata")
            if is_strata and uuid in index_of:
                strata[index_of[uuid]] += int(mib)
        return {index: (used[index], strata[index]) for index in used}

    @staticmethod
    def amd_vram_used(gpu: Gpu) -> int:
        try:
            return int(Path(gpu.pci, "mem_info_vram_used").read_text()) // 2 ** 20
        except (OSError, ValueError):
            return 0

    @classmethod
    def ram(cls) -> Ram:
        """The RAM, and how much of it is free to use. A running strata engine's own RAM is counted as free, as its
        VRAM is: the engine will not find it beside itself once it restarts."""
        meminfo = {key: int(value.split()[0]) * 1024 / GIB
                   for key, value in cls.read_pairs(Path("/proc/meminfo"), ":").items()}
        if "MemTotal" not in meminfo:
            raise ConfigureError("cannot read the RAM size from /proc/meminfo")

        strata_gb = 0.0
        for status in cls.strata_processes("status"):
            fields = cls.read_pairs(status, ":")
            if "RssAnon" in fields:
                strata_gb += int(fields["RssAnon"].split()[0]) * 1024 / GIB

        total = meminfo["MemTotal"]
        available = meminfo.get("MemAvailable", total) + strata_gb
        return Ram(total, min(available, total), strata_gb)

    @classmethod
    def strata_running(cls) -> bool:
        """A strata engine on this machine (in a container too: the host sees its processes)."""
        return any(cls.strata_processes("comm"))

    @staticmethod
    def strata_processes(file: str) -> list[Path]:
        """/proc/<pid>/<file> of every strata process."""
        found = []
        for comm in Path("/proc").glob("[0-9]*/comm"):
            try:
                if comm.read_text().strip().startswith("strata"):
                    found.append(Path(comm.parent, file))
            except OSError:
                continue
        return found

    @staticmethod
    def nvidia_query(fields: str, what: str = "gpu") -> list[list[str]]:
        """nvidia-smi --query-<what>=<fields>: a row of values per line, the lines it cannot read left out."""
        argv = ["nvidia-smi", f"--query-{what}={fields}", "--format=csv,noheader,nounits"]
        try:
            text = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=True).stdout
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return []
        width = len(fields.split(","))
        rows = [[value.strip() for value in line.split(",")] for line in text.splitlines()]
        return [row for row in rows if len(row) == width]

    @staticmethod
    def read_pairs(file: Path, separator: str) -> dict[str, str]:
        """A file of "key<separator>value" lines; an unreadable file is empty."""
        try:
            lines = file.read_text().splitlines()
        except OSError:
            return {}
        pairs = {}
        for line in lines:
            key, found, value = line.partition(separator)
            if found:
                pairs[key.strip()] = value.strip()
        return pairs


# ---- the plan -------------------------------------------------------------------------------------------------

class Vision(StrEnum):
    OFF = "off"
    CPU = "cpu"
    GPU = "gpu"


@dataclass
class Plan:
    """Every setting that goes into the .env, and the notes printed beside them."""
    repo: str
    quant: str
    model: Model | None           # None: a quantization not known here
    gpus: list[Gpu]
    ram_gb: float
    context: int
    kv: str
    vision: Vision
    api_key: str | None
    vram_reserve_mib: int = HEADLESS_RESERVE_MIB
    draft_vocab: str | None = None
    kv_streaming: bool = False
    ram_budget_gib: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def engine_args(self) -> str:
        args = ENGINE_ARGS_BASE
        if self.kv_streaming:
            args += f" --kv-resident {KV_RESIDENT}"
        if self.ram_budget_gib is not None:
            args += f" --resident-budget-gib {self.ram_budget_gib}"
        return args


def parse_context(text: str) -> int:
    text = text.strip().lower()
    try:
        return int(text[:-1]) * 1024 if text.endswith("k") else int(text)
    except ValueError:
        raise ConfigureError(f"--context {text}: tokens, e.g. 32768, or with k: 64k")


def context_text(tokens: int) -> str:
    return f"{tokens // 1024}k" if tokens % 1024 == 0 else str(tokens)


def pick_gpus(found: list[Gpu], asked: str | None) -> list[Gpu]:
    """The cards asked for, or every supported card of the vendor with the most VRAM (one vendor per engine)."""
    if asked:
        by_index = {g.index: g for g in found}
        chosen = []
        for index in (int(x) for x in asked.split(",")):
            if index not in by_index:
                raise ConfigureError(f"--gpus {asked}: no GPU {index} (found {', '.join(str(i) for i in by_index)})")
            chosen.append(by_index[index])
        if len({g.vendor for g in chosen}) > 1:
            raise ConfigureError(f"--gpus {asked}: NVIDIA and AMD cards cannot share one engine")
        return chosen

    nvidia = [g for g in found if g.vendor == Vendor.NVIDIA]
    amd = [g for g in found if g.vendor == Vendor.AMD and g.arch in AMD_ARCHS]
    if not nvidia and not amd:
        raise ConfigureError("no supported GPU found: NVIDIA needs nvidia-smi, AMD an RX 7000 / 9000 card and the "
                             "amdgpu driver")
    return max(nvidia, amd, key=lambda cards: sum(g.total_mib for g in cards))


def plan_reserve(plan: Plan, asked_mib: int | None) -> None:
    """The VRAM kept free on each card: room for what already lives there to grow. The engine counts the baseline
    itself when it starts; the reserve is on top of it."""
    for gpu in plan.gpus:
        is_busy = gpu.baseline_mib > BUSY_BASELINE_MIB
        if gpu.vendor == Vendor.NVIDIA and gpu.display:
            plan.vram_reserve_mib = max(plan.vram_reserve_mib, DISPLAY_RESERVE_MIB)
            plan.notes.append(f"GPU {gpu.index} drives a display: {DISPLAY_RESERVE_MIB} MiB kept free so the desktop "
                              "does not freeze once the expert cache fills the card")
        elif gpu.vendor == Vendor.AMD and is_busy:
            plan.vram_reserve_mib = max(plan.vram_reserve_mib, DESKTOP_RESERVE_MIB)
            plan.notes.append(f"GPU {gpu.index} already uses {gpu.baseline_mib} MiB (a desktop?): {DESKTOP_RESERVE_MIB} "
                              "MiB kept free, or amdgpu moves the expert cache to RAM when the desktop grows")
        elif is_busy:
            plan.vram_reserve_mib = max(plan.vram_reserve_mib, DISPLAY_RESERVE_MIB)
            plan.notes.append(f"GPU {gpu.index} already uses {gpu.baseline_mib} MiB: {DISPLAY_RESERVE_MIB} MiB kept free "
                              "for the programs on it to grow")

    if plan.vision == Vision.GPU:
        plan.vram_reserve_mib = max(plan.vram_reserve_mib, VISION_GPU_RESERVE_MIB)
    if asked_mib is not None:
        plan.vram_reserve_mib = asked_mib


def plan_small_card(plan: Plan, context_asked: bool) -> None:
    """A single card with little VRAM left: a shorter context and the smallest draft head leave the expert cache
    some room."""
    smallest = min(plan.gpus, key=lambda g: g.free_mib)
    left_gb = (smallest.free_mib - plan.vram_reserve_mib) / 1024
    is_small = len(plan.gpus) == 1 and left_gb < SMALL_CARD_GB
    if not is_small:
        return
    if not context_asked:
        plan.context = SMALL_CARD_CONTEXT
    plan.draft_vocab = "en"
    plan.notes.append(f"GPU {smallest.index} has {left_gb:.1f} GB left after what it already uses and the reserve: "
                      f"{context_text(plan.context)} context and the English draft vocabulary, so the expert cache "
                      "has room")


def kv_ram_gb(context: int, kv: str) -> float:
    """The RAM a streamed KV cache takes: 12 attention layers and the draft layer."""
    return context * 13 * KV_CELL_BYTES[kv] / 1e9


def plan_kv_streaming(plan: Plan) -> None:
    """From 64K the KV cache can live in RAM, only 32K positions per layer in VRAM, which frees VRAM for experts.
    It needs the RAM beside the model."""
    is_long = plan.context >= KV_STREAMING_FROM
    if not is_long or plan.model is None:
        return
    kv_gb = kv_ram_gb(plan.context, plan.kv)
    need_gb = min(plan.model.ram_gb + kv_gb + 1, plan.model.arena_gb + LOW_RAM_HEADROOM_GB + kv_gb)
    plan.kv_streaming = plan.ram_gb >= need_gb
    if plan.kv_streaming:
        plan.notes.append(f"KV streaming on: the {context_text(plan.context)} KV cache lives in RAM ({kv_gb:.1f} GB), "
                          "so more experts fit in VRAM")
    else:
        plan.notes.append(f"KV streaming off: it needs ~{need_gb:.0f} GB of RAM, this machine has {plan.ram_gb:.0f}")


def plan_ram_budget(plan: Plan) -> None:
    """Unsloth's files keep every expert in RAM when they fit with room beside them, else the hottest N GiB of them
    (--resident-budget-gib) and the rest read from the SSD."""
    model, ram = plan.model, plan.ram_gb
    if model is None:
        plan.notes.append(f"MODEL_QUANT={plan.quant}: its size is not known here, so no RAM budget is set (the engine "
                          "keeps every expert in RAM)")
        return

    if model.ram_budget and ram < model.arena_gb + UNSLOTH_RESIDENT_BESIDE_GB:
        streamed_gb = kv_ram_gb(plan.context, plan.kv) if plan.kv_streaming else 0.0
        budget = round(ram) - UNSLOTH_RAM_LEFT_GB - math.ceil(streamed_gb)
        plan.ram_budget_gib = max(MIN_RAM_BUDGET_GIB, min(budget, int(model.arena_gb / GB_PER_GIB)))
        plan.notes.append(f"{ram:.0f} GB of RAM does not hold {plan.quant}'s {model.arena_gb:.0f} GB of experts with "
                          f"room to spare: {plan.ram_budget_gib} GiB of them kept in RAM, the rest read from the SSD")
    elif not model.ram_budget and ram < model.arena_gb + LOW_RAM_HEADROOM_GB:
        plan.notes.append(f"{ram:.0f} GB of RAM does not hold {plan.quant}'s {model.arena_gb:.0f} GB of experts: a "
                          "smaller quantization fits better, or add --mmap-experts to ENGINE_ARGS to read them from "
                          "the pack")


def resolve_model(quant_asked: str, repo_asked: str | None) -> tuple[str, str, Model | None]:
    """(repository, quantization, Model): a known quantization as the Hub spells it, from its own repository unless
    --repo names another; any other needs --repo."""
    quant = find_quant(quant_asked)
    if quant is None and not repo_asked:
        raise ConfigureError(f"--quant {quant_asked} is not one known here ({', '.join(Quant)}): name its "
                             "repository with --repo")
    if quant is None:
        return repo_asked, quant_asked, None
    model = MODELS[quant]
    return repo_asked or model.repo, str(quant), model


def make_plan(args: argparse.Namespace, gpus: list[Gpu], ram_gb: float) -> Plan:
    repo, quant, model = resolve_model(args.quant, args.repo)
    context = parse_context(args.context) if args.context else DEFAULT_CONTEXT
    plan = Plan(repo=repo, quant=quant, model=model, gpus=gpus, ram_gb=ram_gb, context=context, kv=args.kv,
                vision=Vision(args.vision), api_key=args.api_key)
    plan_reserve(plan, args.vram_reserve_mib)
    plan_small_card(plan, context_asked=bool(args.context))
    plan_kv_streaming(plan)
    plan_ram_budget(plan)
    if len(gpus) > 1:
        plan.notes.append(f"{len(gpus)} GPUs: the layers are split across them, each loading only its own dense "
                          "weights (STRATA_STAGE_TRIM=1)")
    return plan


# ---- the output -----------------------------------------------------------------------------------------------

def say(text: str = "") -> None:
    """The report goes to stderr, so stdout holds only the settings."""
    print(text, file=sys.stderr)


def print_hardware(gpus: list[Gpu], ram: Ram) -> None:
    name_width = max(len(g.name) for g in gpus)
    say()
    say("GPUs")
    for gpu in gpus:
        link = str(gpu.link) if gpu.link else "PCIe link unknown"
        say(f"  {gpu.index}  {gpu.name:<{name_width}}  {gpu.total_gb:5.1f} GB  "
            f"idle {gpu.baseline_mib:>5} MiB used, {gpu.free_mib / 1024:5.1f} GB free  {link}")
    for gpu in gpus:
        if gpu.strata_mib:
            say(f"     GPU {gpu.index}: {gpu.strata_mib} MiB held by a running strata engine, counted as free")
    say("RAM")
    say(f"  {ram.total_gb:.1f} GB, {ram.available_gb:.1f} GB available")
    if ram.strata_gb >= 0.1:
        say(f"     {ram.strata_gb:.1f} GB held by a running strata engine, counted as available")
    say()


def build_hints(gpus: list[Gpu]) -> list[str]:
    """What compose.yaml needs for these cards (the image or build args), which the .env cannot set."""
    archs = ";".join(sorted({g.arch for g in gpus}))
    if gpus[0].vendor == Vendor.NVIDIA:
        return [f"image: ghcr.io/legendaryfire/strata-slim:cuda, or build with CUDA_ARCHITECTURES: \"{archs}\""]
    return [f"image: ghcr.io/legendaryfire/strata-slim:rocm, or build Dockerfile.rocm with HIP_ARCHITECTURES: \"{archs}\"",
            "pass /dev/kfd and /dev/dri as devices instead of the deploy section"]


def env_text(plan: Plan) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M")
    lines = [f"# strata-slim settings, written by configure.py on {stamp} for this machine:"]
    for gpu in plan.gpus:
        lines.append(f"#   GPU {gpu.index} {gpu.name}: {gpu.total_gb:.0f} GB, {gpu.baseline_mib} MiB in use before loading")
    lines.append(f"#   {plan.ram_gb:.0f} GB of RAM")
    lines.append("# Every variable there is, with what it does, is in example.env.")

    lines += ["", "# ---- the model", f"MODEL_REPO={plan.repo}", f"MODEL_QUANT={plan.quant}"]
    if plan.model and plan.model.pack_args:
        lines.append(f"PACK_ARGS={plan.model.pack_args}")
    lines.append("DRAFTER=mtp")
    if plan.draft_vocab:
        lines.append(f"DRAFT_VOCAB={plan.draft_vocab}")

    lines += ["", "# ---- the engine", f"CONTEXT={context_text(plan.context)}", f"KV={plan.kv}",
              f"VRAM_RESERVE_MIB={plan.vram_reserve_mib}", f"ENGINE_ARGS={plan.engine_args}"]

    lines += ["", "# ---- the GPUs", f"GPUS={','.join(str(g.index) for g in plan.gpus)}"]
    if len(plan.gpus) > 1:
        lines += ["CONFIG_LAYER_SPLIT=auto", "STRATA_STAGE_TRIM=1"]

    lines += ["", "# ---- pictures", f"VISION={plan.vision}"]
    if plan.vision != Vision.OFF:
        lines.append(f"MMPROJ_REPO={MMPROJ_REPO}")
    if plan.vision == Vision.CPU:
        lines.append(f"VISION_THREADS={max(1, min(8, (os.cpu_count() or 2) // 2))}")

    lines += ["", "# ---- the server", f"API_KEY={plan.api_key}" if plan.api_key else "# API_KEY=", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print strata-slim's settings for this machine's GPUs and RAM.")
    parser.add_argument("--output", "-o", type=Path, help="also write them to this file, e.g. .env")
    parser.add_argument("--force", action="store_true", help="overwrite the --output file when it exists")
    parser.add_argument("--gpus", help="card numbers to use, e.g. 0,1 (default: every supported card)")
    parser.add_argument("--quant", default=DEFAULT_QUANT,
                        help=f"quantization (default {DEFAULT_QUANT}): {', '.join(Quant)}, or another with --repo")
    parser.add_argument("--repo", help="Hugging Face repository (default: the one --quant is known from)")
    parser.add_argument("--context", help="context length, e.g. 64k (default 128k, 32k on a small card)")
    parser.add_argument("--kv", default="int8", choices=list(KV_CELL_BYTES), help="KV cache type (default int8)")
    parser.add_argument("--vram-reserve-mib", type=int,
                        help="VRAM kept free on each card (default: from what the cards already use)")
    parser.add_argument("--vision", default=Vision.OFF, choices=list(Vision), help="image input (default off)")
    parser.add_argument("--api-key", help="the key clients send (default: none)")
    args = parser.parse_args()

    # the arguments checked before measuring anything
    resolve_model(args.quant, args.repo)
    if args.context:
        parse_context(args.context)
    if args.output and args.output.exists() and not args.force:
        raise ConfigureError(f"{args.output} already exists: --force to overwrite it, or --output another file")

    # the hardware first: the baseline is measured before anything else, with nothing of strata's counted
    found = Hardware.gpus()
    if not found:
        raise ConfigureError("no GPU found (nvidia-smi for NVIDIA, /sys/class/kfd for AMD)")
    say("Measuring the VRAM the cards already use...")
    Hardware.measure_baseline(found)
    ram = Hardware.ram()
    print_hardware(found, ram)

    gpus = pick_gpus(found, args.gpus)
    plan = make_plan(args, gpus, ram.total_gb)
    unused = [str(g.index) for g in found if g not in gpus]
    if unused:
        plan.notes.append(f"not used: GPU {', '.join(unused)}")
    is_amd = gpus[0].vendor == Vendor.AMD
    if is_amd and Hardware.strata_running():
        plan.notes.append("a strata engine is running: AMD cannot tell its VRAM apart, so the baseline includes it. "
                          "Stop it (docker compose stop) and run this again for a true baseline")
    for note in plan.notes + build_hints(gpus):
        say(f"  - {note}")

    text = env_text(plan)
    say("\nThe settings, to paste into .env:\n")
    sys.stdout.write(text)
    sys.stdout.flush()
    if args.output:
        args.output.write_text(text)
        say(f"\nWrote {args.output}. Start it with: docker compose up -d")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigureError as e:
        print(f"configure.py: {e}", file=sys.stderr)
        sys.exit(2)
