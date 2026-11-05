"""Get the model files and prepare them for the engine. Every step is skipped when its result is already there.

  1. download   the model's GGUF files and the picture reader (mmproj) from Hugging Face
  2. pack       tools/iq_pack.py: the pack the engine reads beside the GGUF (dense tensors, expert index, tokenizer)
  3. mtp        tools/mtp_fetch.py, mtp_pack.py, mtp_rt.py: the MTP draft layer (speculative decoding), from the 31 MTP
                tensors of the Qwen3.8-Flash-Next BF16 checkpoint (about 5 GB, read with range requests)

The variables are in example.env.

Which version
  The first download writes <model folder>/.source.json: the repository, the commit (a branch resolved to a fixed
  commit) and each file's size and SHA-256. After that the folder is that version: later starts use the record, need
  no network when every file is there, and are not affected when the repository replaces files under the same names.
  A MODEL_REVISION that is another commit stops the start: a new version goes into a new folder, with a new pack
  (the pack is built from these exact files).
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import hf
from common import ROOT, ModelPaths, Progress, drafter, env, fail, first_shard, mmproj_near, say, size_text
from hf import RemoteFile
from strata_control import acquisition, drafting, drafting_fetch

TOOLS = ROOT / "tools"
DRAFT_VOCABS = {"cjk": "draft_vocab.bin", "en": "draft_vocab_en.bin", "cyrillic": "draft_vocab_cyrillic.bin",
                "fr": "draft_vocab_fr.bin"}
RECORD_FILE = ".source.json"


@dataclass
class Prepared:
    """What the steps found or made, for launch.py."""
    model: Path                # the GGUF the engine is given (shard 1)
    mmproj: Path | None        # the picture reader, when VISION is on


def run(paths: ModelPaths, vision: bool) -> Prepared:
    model_folder = paths.model if paths.model.suffix != ".gguf" else paths.model.parent
    if env("MODEL_REPO"):
        if paths.model.suffix == ".gguf":
            fail("MODEL_REPO is set: MODEL_GGUF must be the folder the files go into, not a file")
        download_model(model_folder)
    mmproj = get_mmproj(model_folder) if vision else None
    model = first_shard(paths.model)
    build_pack(model, paths.pack)
    if paths.mtp:
        build_mtp(paths.mtp)
    if drafter() in ("dflash2", "auto"):
        build_dflash2(paths.data_dir)
    say("model ready")
    return Prepared(model, mmproj)


# ---- which files ------------------------------------------------------------------------------------------------

def is_mmproj(f: RemoteFile) -> bool:
    return f.name.lower().startswith("mmproj") and f.name.endswith(".gguf")


def quant_of(f: RemoteFile) -> str:
    """The quantization a model file belongs to: its folder (UD-Q4_K_XL/...), or its name without the shard suffix."""
    if "/" in f.path:
        return f.path.split("/", 1)[0]
    return re.sub(r"-\d{5}-of-\d{5}$", "", f.name.removesuffix(".gguf"))


def files_by_quant(files: list[RemoteFile], repo: str, quant: str | None) -> tuple[str, list[RemoteFile]]:
    """The files of one quantization: MODEL_QUANT, or the only one the repository has."""
    groups: dict[str, list[RemoteFile]] = {}
    for f in files:
        if f.name.endswith(".gguf") and not is_mmproj(f) and not f.name.lower().startswith("mtp"):
            groups.setdefault(quant_of(f), []).append(f)
    choices = "\n".join(f"    {q:28} {len(fs)} file(s), {size_text(sum(f.size for f in fs))}"
                        for q, fs in sorted(groups.items()))

    if not quant:
        if len(groups) == 1:
            return next(iter(groups.items()))
        fail(f"{repo} has {len(groups)} quantizations; set MODEL_QUANT to one of them:\n{choices}")

    exact = [q for q in groups if q.lower() == quant.lower()]
    suffix = [q for q in groups if q.lower().endswith("-" + quant.lower())]
    matches = exact or suffix
    if len(matches) != 1:
        found = "matches none" if not matches else f"matches {len(matches)} ({', '.join(matches)})"
        fail(f"MODEL_QUANT={quant} {found} of {repo}'s quantizations:\n{choices}")
    return matches[0], groups[matches[0]]


def choose_model_files(repo: str, commit: str) -> tuple[str, list[RemoteFile]]:
    """(quantization, files) at a commit, by MODEL_QUANT."""
    return files_by_quant(hf.list_files(repo, commit), repo, env("MODEL_QUANT"))


def choose_mmproj(repo: str, commit: str) -> RemoteFile:
    candidates = [f for f in hf.list_files(repo, commit) if is_mmproj(f)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        fail(f"{repo} has no mmproj*.gguf file; set MMPROJ_REPO to a repository with one, or VISION=off")
    fail(f"{repo} has {len(candidates)} mmproj files ({', '.join(f.path for f in candidates)}): "
         "download the one you want beside the model folder, or set MMPROJ to it")


# ---- the record of a download (.source.json) ----------------------------------------------------------------------

@dataclass
class Record:
    repo: str
    commit: str
    quant: str
    files: list[RemoteFile]

    @staticmethod
    def read(folder: Path) -> Record | None:
        path = folder / RECORD_FILE
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        files = [RemoteFile(f["path"], f["size"], f["sha256"]) for f in data["files"]]
        return Record(data["repo"], data["commit"], data.get("quant"), files)

    def write(self, folder: Path) -> None:
        data = {"repo": self.repo, "commit": self.commit, "quant": self.quant, "recorded": time.strftime("%F %T"),
                "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in self.files]}
        (folder / RECORD_FILE).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def marked_files(folder: Path) -> dict[str, str | None]:
    """The GGUF files in a folder with a .done mark: name -> the SHA-256 in the mark (or None)."""
    names = [mark.name.removesuffix(".done") for mark in folder.glob("*.gguf.done")]
    return {name: hf.marked_sha(folder / name) for name in names}


def folder_holds_other_files(folder: Path, files: list[RemoteFile]) -> bool:
    """True when the folder already has finished files that are not exactly these (another version or quant)."""
    wanted = {f.name: f.sha256 for f in files}
    for name, sha in marked_files(folder).items():
        if name not in wanted or (sha and wanted[name] and sha != wanted[name]):
            return True
    return False


# ---- step 1: download -------------------------------------------------------------------------------------------

def download_model(folder: Path) -> None:
    repo = env("MODEL_REPO")
    record = Record.read(folder)
    if record:
        complete_recorded(folder, record, repo)
    else:
        first_download(folder, repo)


def complete_recorded(folder: Path, record: Record, repo: str) -> None:
    """The folder has a record: check the variables still ask for that version, then fetch any missing file."""
    if record.repo != repo:
        fail(f"{folder} holds {record.repo}, not {repo}: use another folder for another model")
    quant = env("MODEL_QUANT")
    if quant and record.quant and quant.lower() != record.quant.lower():
        fail(f"{folder} holds {record.quant}, not {quant}: use another folder (and pack) for another quantization")
    pinned = env("MODEL_REVISION")
    if pinned and pinned != record.commit and hf.resolve_commit(repo, pinned) != record.commit:
        fail(f"{folder} holds {repo}@{record.commit[:12]}, not MODEL_REVISION={pinned}. Use a new folder (and pack) "
             "for another version, or unset MODEL_REVISION to keep this one.")

    missing = [f for f in record.files if not hf.is_done(folder / f.name, f)]
    if not missing:
        say(f"{folder.name}: {len(record.files)} file(s) of {repo}@{record.commit[:12]} present")
        return
    say(f"{folder.name}: {len(missing)} of {len(record.files)} file(s) missing, fetching them at the recorded commit")
    for f in missing:
        hf.download(repo, record.commit, f, folder / f.name)


def first_download(folder: Path, repo: str) -> None:
    """No record yet: resolve the revision, adopt files already there, download the rest, write the record."""
    pinned = env("MODEL_REVISION")
    commit = hf.resolve_commit(repo, pinned or "main")
    quant, files = choose_model_files(repo, commit)

    if folder_holds_other_files(folder, files):
        fail(f"{folder}: the files there are not {repo}@{commit[:12]}'s {quant}. Use an empty folder, or set "
             "MODEL_REVISION to the commit they came from.")

    say(f"{repo}@{commit[:12]}{' ' + quant if quant else ''}: {len(files)} file(s), "
        f"{size_text(sum(f.size for f in files))} -> {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        hf.download(repo, commit, f, folder / f.name)
    Record(repo, commit, quant, files).write(folder)


def get_mmproj(model_folder: Path) -> Path:
    """The picture reader: MMPROJ, else the one in or beside the model folder; downloaded when it is missing."""
    given = env("MMPROJ")
    if given and Path(given).exists():
        return Path(given)
    if not given and (nearby := mmproj_near(model_folder)):
        return nearby

    repo = env("MMPROJ_REPO") or env("MODEL_REPO")
    if not repo:
        fail("VISION is on and no mmproj file was found: set MMPROJ to it, or MMPROJ_REPO to download one")
    commit = hf.resolve_commit(repo, "main")
    f = choose_mmproj(repo, commit)
    path = Path(given) if given else model_folder.parent / f.name     # beside the model folders, as upstream keeps it
    say(f"picture reader: {repo}@{commit[:12]} {f.path} -> {path}")
    hf.download(repo, commit, f, path)
    return path


# ---- steps 2 and 3: the pack and the MTP draft layer ---------------------------------------------------------------

def tool(name: str, *args: str, tries: int = 1) -> None:
    """Run one of upstream's tools; a tool that resumes (mtp_fetch) can be given more than one try."""
    cmd = [sys.executable, str(TOOLS / name), *args]
    for attempt in range(1, tries + 1):
        say("$ " + " ".join(shlex.quote(c) for c in cmd))
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode == 0:
            return
        if attempt < tries:
            say(f"{name} stopped (exit {result.returncode}); again in 10 s ({attempt + 1} of {tries})")
            time.sleep(10)
    fail(f"{name} failed (exit {result.returncode})")


def build_pack(model: Path, pack: Path) -> None:
    if (pack / "native_experts.txt").is_file() and (pack / "tokenizer" / "vocab.json").is_file():
        return
    say(f"building the pack {pack} from {model.name}")
    pack.parent.mkdir(parents=True, exist_ok=True)
    tool("iq_pack.py", "--gguf", str(model), "--out", str(pack), *shlex.split(env("PACK_ARGS") or ""))


def build_mtp(rt: Path) -> None:
    if not (rt / "experts.bin").is_file():
        root, experts = rt.parent, env("MTP_EXPERTS") or "q2_0"
        say(f"building the MTP draft layer in {root} (about 5 GB of tensors from the Qwen3.8-Flash-Next checkpoint)")
        root.mkdir(parents=True, exist_ok=True)
        tool("mtp_fetch.py", "fetch", "--out", str(root), tries=3)
        tool("mtp_fetch.py", "verify", "--out", str(root))
        packed = root / f"mtp-{experts}.gguf"
        tool("mtp_pack.py", "--src", str(root), "--experts", experts, "--out", str(packed))
        tool("mtp_rt.py", "--gguf", str(packed), "--out", str(rt))
    set_draft_vocab(rt)


def build_dflash2(data_dir: Path) -> None:
    """The DFlash2 drafter in <data_dir>/dflash2/, made by upstream's own fetch (strata_control/drafting_fetch.py):
    its pinned files downloaded and checked, packed to Q4_0, the pack checked against the exact bytes upstream measured,
    and the draft vocabulary beside it. The engine flags for it are set by the server at each start ("drafting")."""
    current: dict = {"step": "", "bar": None}

    def progress(step: str, done: int, total: int) -> None:
        """drafting_fetch reports (step, bytes done, bytes total): one progress line per step that has a size."""
        if step != current["step"]:
            if current["bar"]:
                current["bar"].finish()
            current["step"], current["bar"] = step, None
            if total > 1 and done < total:
                current["bar"] = Progress(f"DFlash2: {step.rstrip('…. ')}", total, done=done)
            else:
                say(f"DFlash2: {step}")
        if current["bar"]:
            current["bar"].update(done)

    try:
        drafting_fetch.ensure(data_dir, progress=progress)
        if drafting.engine_reads_selector_q4(ROOT / "engine" / "strata"):
            drafting_fetch.convert_selector(data_dir, progress=progress)    # 174 MiB less VRAM, same output
    except (drafting_fetch.DraftingError, acquisition.AcquisitionRefused) as e:
        fail(f"DFlash2: {e}")
    if current["bar"]:
        current["bar"].finish()
    if (choice := env("DRAFT_VOCAB")):
        drafting.provide_vocab(data_dir, choice.lower())


def set_draft_vocab(rt: Path) -> None:
    """Put the draft token subset into the MTP folder: the default (cjk) when there is none, or the one DRAFT_VOCAB names."""
    if not rt.is_dir():
        return
    target = rt / "draft_vocab.bin"
    choice = (env("DRAFT_VOCAB") or "").lower()
    if target.exists() and not choice:
        return
    choice = choice or "cjk"
    if choice not in DRAFT_VOCABS:
        fail(f"DRAFT_VOCAB={choice}: one of {', '.join(DRAFT_VOCABS)}")
    shutil.copyfile(ROOT / "data" / DRAFT_VOCABS[choice], target)
    say(f"draft layer token subset: {choice}")
