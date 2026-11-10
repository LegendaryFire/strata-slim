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
import logging
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import hf
from common import ROOT, Progress, SetupError, engine_program, first_shard, mmproj_near, size_text
from hf import HuggingFace, RemoteFile
from settings import DraftVocab, Drafter, ModelPaths, Settings
from strata_control import acquisition, drafting, drafting_fetch

log = logging.getLogger(__name__)

TOOLS = Path(ROOT, "tools")
DRAFT_VOCAB_FILES = {
    DraftVocab.CJK: "draft_vocab.bin",
    DraftVocab.EN: "draft_vocab_en.bin",
    DraftVocab.CYRILLIC: "draft_vocab_cyrillic.bin",
    DraftVocab.FR: "draft_vocab_fr.bin",
}
RECORD_FILE = ".source.json"


# ---- the record of a download (.source.json) ----------------------------------------------------------------------

@dataclass
class Record:
    repo: str
    commit: str
    quant: str
    files: list[RemoteFile]

    @staticmethod
    def read(folder: Path) -> Record | None:
        path = Path(folder, RECORD_FILE)
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        files = []
        for f in data["files"]:
            files.append(RemoteFile(f["path"], f["size"], f["sha256"]))
        return Record(data["repo"], data["commit"], data.get("quant"), files)

    def write(self, folder: Path) -> None:
        files = []
        for f in self.files:
            files.append({"path": f.path, "size": f.size, "sha256": f.sha256})
        data = {"repo": self.repo, "commit": self.commit, "quant": self.quant, "recorded": time.strftime("%F %T"),
                "files": files}
        record_path = Path(folder, RECORD_FILE)
        with record_path.open("w") as f:
            json.dump(data, f, indent=1)


def tool(name: str, *args: str, tries: int = 1) -> None:
    """Run one of upstream's tools; a tool that resumes (mtp_fetch) can be given more than one try."""
    cmd = [sys.executable, str(Path(TOOLS, name)), *args]
    for attempt in range(1, tries + 1):
        log.info(f"$ {shlex.join(cmd)}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode == 0:
            return
        if attempt < tries:
            log.warning(f"{name} stopped (exit {result.returncode}); again in 10 s ({attempt + 1} of {tries})")
            time.sleep(10)
    raise SetupError(f"{name} failed (exit {result.returncode})")


# ---- the steps --------------------------------------------------------------------------------------------------

class Preparer:
    """Downloads and builds what the engine needs. run() returns (the GGUF for the engine, the mmproj or None)."""

    def __init__(self, settings: Settings, paths: ModelPaths):
        self.settings = settings
        self.paths = paths
        self.hub = HuggingFace(settings.hf_endpoint, settings.hf_token)

    def run(self) -> tuple[Path, Path | None]:
        model_folder = self.paths.model
        if model_folder.suffix == ".gguf":
            model_folder = model_folder.parent

        if self.settings.model_repo:
            if self.paths.model.suffix == ".gguf":
                raise SetupError("MODEL_REPO is set: MODEL_GGUF must be the folder the files go into, not a file")
            self.download_model(model_folder)

        mmproj = None
        if self.settings.vision_on:
            mmproj = self.get_mmproj(model_folder)

        model = first_shard(self.paths.model)
        self.build_pack(model)
        if self.paths.mtp:
            self.build_mtp()
        if self.settings.drafter in (Drafter.DFLASH2, Drafter.AUTO):
            self.build_dflash2()

        log.info("model ready")
        return model, mmproj

    # ---- step 1: download ---------------------------------------------------------------------------------------

    def download_model(self, folder: Path) -> None:
        record = Record.read(folder)
        if record:
            self.complete_recorded(folder, record)
        else:
            self.first_download(folder)

    def complete_recorded(self, folder: Path, record: Record) -> None:
        """The folder has a record: check the settings still ask for that version, then fetch any missing file."""
        repo = self.settings.model_repo
        if record.repo != repo:
            raise SetupError(f"{folder} holds {record.repo}, not {repo}: use another folder for another model")

        quant = self.settings.model_quant
        if quant and record.quant and quant.lower() != record.quant.lower():
            raise SetupError(f"{folder} holds {record.quant}, not {quant}: "
                             "use another folder (and pack) for another quantization")

        pinned = self.settings.model_revision
        if pinned and pinned != record.commit and self.hub.resolve_commit(repo, pinned) != record.commit:
            raise SetupError(f"{folder} holds {repo}@{record.commit[:12]}, not MODEL_REVISION={pinned}. Use a new "
                             "folder (and pack) for another version, or unset MODEL_REVISION to keep this one.")

        missing = []
        for f in record.files:
            if not hf.is_done(Path(folder, f.name), f):
                missing.append(f)
        if not missing:
            log.info(f"{folder.name}: {len(record.files)} file(s) of {repo}@{record.commit[:12]} present")
            return

        log.info(f"{folder.name}: {len(missing)} of {len(record.files)} file(s) missing, "
                 "fetching them at the recorded commit")
        for f in missing:
            self.hub.download(repo, record.commit, f, Path(folder, f.name))

    def first_download(self, folder: Path) -> None:
        """No record yet: resolve the revision, adopt files already there, download the rest, write the record."""
        repo = self.settings.model_repo
        commit = self.hub.resolve_commit(repo, self.settings.model_revision or "main")

        # the model's GGUF files by quantization: their folder (UD-Q4_K_XL/...), or the name without the shard suffix
        groups: dict[str, list[RemoteFile]] = {}
        for f in self.hub.list_files(repo, commit):
            if not f.name.endswith(".gguf") or f.name.lower().startswith(("mmproj", "mtp")):
                continue
            if "/" in f.path:
                q = f.path.split("/", 1)[0]
            else:
                q = re.sub(r"-\d+-of-\d+$", "", f.name.removesuffix(".gguf"))   # Model-Q8_0-00001-of-00003 -> Model-Q8_0
            groups.setdefault(q, []).append(f)

        choices = ""
        for q, group in sorted(groups.items()):
            total = 0
            for f in group:
                total += f.size
            choices += f"\n    {q:28} {len(group)} file(s), {size_text(total)}"

        # MODEL_QUANT picks one: an exact name, else a suffix (Q4_K_XL finds UD-Q4_K_XL). Unset: the only one there is
        asked = self.settings.model_quant
        matches = []
        if not asked:
            if len(groups) != 1:
                raise SetupError(f"{repo} has {len(groups)} quantizations; set MODEL_QUANT to one of them:{choices}")
            matches = list(groups)
        else:
            for q in groups:
                if q.lower() == asked.lower():
                    matches.append(q)
            if not matches:
                for q in groups:
                    if q.lower().endswith(f"-{asked.lower()}"):
                        matches.append(q)
            if len(matches) != 1:
                found = "matches none" if not matches else f"matches {len(matches)} ({', '.join(matches)})"
                raise SetupError(f"MODEL_QUANT={asked} {found} of {repo}'s quantizations:{choices}")
        quant = matches[0]
        files = groups[quant]

        # finished (.done) GGUF files already in the folder are adopted, but only when they are exactly these:
        # another version's or quant's files stop the start
        wanted = {}
        for f in files:
            wanted[f.name] = f.sha256
        for mark in folder.glob("*.gguf.done"):
            name = mark.name.removesuffix(".done")
            sha = hf.marked_sha(Path(folder, name))
            if name not in wanted or (sha and wanted[name] and sha != wanted[name]):
                raise SetupError(f"{folder}: the files there are not {repo}@{commit[:12]}'s {quant}. Use an empty "
                                 "folder, or set MODEL_REVISION to the commit they came from.")

        total = 0
        for f in files:
            total += f.size
        log.info(f"{repo}@{commit[:12]} {quant}: {len(files)} file(s), {size_text(total)} -> {folder}")

        folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            self.hub.download(repo, commit, f, Path(folder, f.name))
        Record(repo, commit, quant, files).write(folder)

    def get_mmproj(self, model_folder: Path) -> Path:
        """The picture reader: MMPROJ, else the one in or beside the model folder; downloaded when it is missing."""
        given = self.settings.mmproj
        if given and given.exists():
            return given
        if not given:
            nearby = mmproj_near(model_folder)
            if nearby:
                return nearby

        repo = self.settings.mmproj_repo or self.settings.model_repo
        if not repo:
            raise SetupError("VISION is on and no mmproj file was found: "
                             "set MMPROJ to it, or MMPROJ_REPO to download one")
        commit = self.hub.resolve_commit(repo, "main")

        candidates = []
        for f in self.hub.list_files(repo, commit):
            is_mmproj = f.name.lower().startswith("mmproj") and f.name.endswith(".gguf")
            if is_mmproj:
                candidates.append(f)
        if not candidates:
            raise SetupError(f"{repo} has no mmproj*.gguf file; "
                             "set MMPROJ_REPO to a repository with one, or VISION=off")
        if len(candidates) > 1:
            names = ", ".join(f.path for f in candidates)
            raise SetupError(f"{repo} has {len(candidates)} mmproj files ({names}): "
                             "download the one you want beside the model folder, or set MMPROJ to it")
        f = candidates[0]

        path = given or Path(model_folder.parent, f.name)     # beside the model folders, as upstream keeps it
        log.info(f"picture reader: {repo}@{commit[:12]} {f.path} -> {path}")
        self.hub.download(repo, commit, f, path)
        return path

    # ---- steps 2 and 3: the pack and the drafter ----------------------------------------------------------------

    def build_pack(self, model: Path) -> None:
        pack = self.paths.pack
        if Path(pack, "native_experts.txt").is_file() and Path(pack, "tokenizer", "vocab.json").is_file():
            return
        log.info(f"building the pack {pack} from {model.name}")
        pack.parent.mkdir(parents=True, exist_ok=True)
        tool("iq_pack.py", "--gguf", str(model), "--out", str(pack), *shlex.split(self.settings.pack_args))

    def build_mtp(self) -> None:
        rt = self.paths.mtp
        if not Path(rt, "experts.bin").is_file():
            root = rt.parent
            experts = self.settings.mtp_experts
            log.info(f"building the MTP draft layer in {root} "
                     "(about 5 GB of tensors from the Qwen3.8-Flash-Next checkpoint)")
            root.mkdir(parents=True, exist_ok=True)
            tool("mtp_fetch.py", "fetch", "--out", str(root), tries=3)
            tool("mtp_fetch.py", "verify", "--out", str(root))
            packed = Path(root, f"mtp-{experts}.gguf")
            tool("mtp_pack.py", "--src", str(root), "--experts", experts, "--out", str(packed))
            tool("mtp_rt.py", "--gguf", str(packed), "--out", str(rt))

        # the draft token subset: cjk when there is none yet, or the one DRAFT_VOCAB names
        target = Path(rt, "draft_vocab.bin")
        choice = self.settings.draft_vocab
        if target.exists() and not choice:
            return
        choice = choice or DraftVocab.CJK
        shutil.copyfile(Path(ROOT, "data", DRAFT_VOCAB_FILES[choice]), target)
        log.info(f"draft layer token subset: {choice}")

    def build_dflash2(self) -> None:
        """The DFlash2 drafter in <models root>/dflash2/, made by upstream's own fetch
        (strata_control/drafting_fetch.py): its pinned files downloaded and checked, packed to Q4_0, the pack checked
        against the exact bytes upstream measured, and the draft vocabulary beside it. The server sets the engine
        flags for it at each start ("drafting")."""
        data_dir = self.paths.data_dir
        step_now = ""
        bar = None

        def progress(step: str, done: int, total: int) -> None:
            """drafting_fetch reports (step, bytes done, bytes total): one progress line per step that has a size."""
            nonlocal step_now, bar
            if step != step_now:
                if bar:
                    bar.finish()
                step_now, bar = step, None
                if total > 1 and done < total:
                    bar = Progress(f"DFlash2: {step.rstrip('…. ')}", total, done=done)
                else:
                    log.info(f"DFlash2: {step}")
            if bar:
                bar.update(done)

        try:
            drafting_fetch.ensure(data_dir, progress=progress)
            if drafting.engine_reads_selector_q4(engine_program("strata")):
                drafting_fetch.convert_selector(data_dir, progress=progress)    # 174 MiB less VRAM, same output
        except (drafting_fetch.DraftingError, acquisition.AcquisitionRefused) as e:
            raise SetupError(f"DFlash2: {e}")
        if bar:
            bar.finish()

        if self.settings.draft_vocab:
            drafting.provide_vocab(data_dir, str(self.settings.draft_vocab))
