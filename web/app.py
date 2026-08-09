from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .database import Database
from .pipeline import Cancelled, StarSampleRuntime, run_pipeline


VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".m2ts", ".webm"}
WEB_ROOT = Path(__file__).parent
database = Database(settings.database_path)


def inside(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise HTTPException(400, "只能使用媒体根目录内的相对路径")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise HTTPException(400, "路径超出允许的目录") from error
    return candidate


def log_tail(job_id: str, lines: int = 120) -> str:
    path = settings.log_root / f"{job_id}.log"
    if not path.is_file():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return "".join(stream.readlines()[-lines:])


class Worker:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.runtime: StarSampleRuntime | None = None
        self.cancel_events: dict[str, threading.Event] = {}
        self.lock = threading.Lock()
        self.current_job_id: str | None = None

    def start(self) -> None:
        database.recover_interrupted()
        self.thread = threading.Thread(target=self.loop, name="upscale-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            for event in self.cancel_events.values():
                event.set()
        if self.thread:
            self.thread.join(timeout=10)

    def cancel(self, job_id: str) -> bool:
        requested = database.request_cancel(job_id)
        with self.lock:
            event = self.cancel_events.get(job_id)
            if event:
                event.set()
        return requested

    def append_log(self, job_id: str, message: str) -> None:
        settings.log_root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with (settings.log_root / f"{job_id}.log").open("a", encoding="utf-8") as stream:
            stream.write(f"[{timestamp}] {message}\n")

    def loop(self) -> None:
        while not self.stop_event.is_set():
            job = database.claim_next()
            if not job:
                self.stop_event.wait(1)
                continue
            job_id = job["id"]
            cancel_event = threading.Event()
            with self.lock:
                self.cancel_events[job_id] = cancel_event
                self.current_job_id = job_id
            try:
                if self.runtime is None:
                    self.runtime = StarSampleRuntime(
                        settings, lambda message: self.append_log(job_id, message)
                    )
                run_pipeline(
                    job,
                    self.runtime,
                    settings,
                    cancel_event,
                    lambda frame, total, fps, eta: database.update_progress(
                        job_id, frame, total, fps, eta
                    ),
                    lambda message: self.append_log(job_id, message),
                )
                database.finish(job_id, "completed")
            except Cancelled:
                self.append_log(job_id, "任务已取消")
                database.finish(job_id, "cancelled")
            except Exception as error:
                self.append_log(job_id, f"失败：{error}")
                database.finish(job_id, "failed", str(error)[:1000])
            finally:
                with self.lock:
                    self.cancel_events.pop(job_id, None)
                    self.current_job_id = None


worker = Worker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.data_root.mkdir(parents=True, exist_ok=True)
    settings.log_root.mkdir(parents=True, exist_ok=True)
    settings.output_root.mkdir(parents=True, exist_ok=True)
    worker.start()
    yield
    worker.stop()


app = FastAPI(title="Anime Upscale", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")


class CreateJobs(BaseModel):
    input_path: str = ""
    output_subdir: str = ""
    recursive: bool = True
    cq: int = Field(default=18, ge=0, le=51)


@app.get("/")
def index():
    return FileResponse(WEB_ROOT / "templates/index.html")


@app.get("/api/status")
def status():
    jobs = database.list()
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    return {
        "model": settings.model_path.name,
        "model_ready": settings.model_path.is_file(),
        "cuda_ready": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "current_job_id": worker.current_job_id,
        "counts": counts,
    }


@app.get("/api/browse")
def browse(path: str = Query(default="")):
    directory = inside(settings.media_root, path)
    if not directory.is_dir():
        raise HTTPException(404, "目录不存在")
    entries = []
    try:
        children = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except PermissionError as error:
        raise HTTPException(403, "没有权限读取该目录") from error
    for child in children[:500]:
        is_directory = child.is_dir()
        if not is_directory and child.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        try:
            relative = child.relative_to(settings.media_root).as_posix()
        except ValueError:
            continue
        entries.append(
            {
                "name": child.name,
                "path": relative,
                "directory": is_directory,
                "size": None if is_directory else child.stat().st_size,
            }
        )
    parent = Path(path).parent.as_posix() if path else None
    if parent == ".":
        parent = ""
    return {"path": path, "parent": parent, "entries": entries}


@app.post("/api/jobs")
def create_jobs(request: CreateJobs):
    selected = inside(settings.media_root, request.input_path)
    output_directory = inside(settings.output_root, request.output_subdir)
    if not selected.exists():
        raise HTTPException(404, "输入路径不存在")
    if selected.is_file():
        files = [selected] if selected.suffix.lower() in VIDEO_EXTENSIONS else []
        base = selected.parent
    else:
        pattern = "**/*" if request.recursive else "*"
        files = sorted(
            path
            for path in selected.glob(pattern)
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        base = selected
    if not files:
        raise HTTPException(400, "没有找到支持的视频文件")
    if len(files) > 500:
        raise HTTPException(400, "一次最多添加 500 个视频")

    planned = []
    for source in files:
        relative_parent = source.parent.relative_to(base) if selected.is_dir() else Path()
        target_dir = output_directory / relative_parent
        target = target_dir / f"{source.stem}.starsample-2x.mkv"
        planned.append((source, target))

    targets = [target for _, target in planned]
    if len(set(targets)) != len(targets):
        raise HTTPException(409, "多个输入会写入同一个输出文件，请调整输入范围")
    for target in targets:
        if target.exists():
            raise HTTPException(409, f"输出文件已存在：{target}")
        if database.output_is_registered(target):
            raise HTTPException(409, f"输出目标已登记：{target}")

    try:
        created = database.create_many(planned, request.cq)
    except sqlite3.IntegrityError as error:
        raise HTTPException(409, "输出目标已被其他任务登记") from error
    return {"created": len(created), "jobs": created}


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": database.list()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = database.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    job["log"] = log_tail(job_id)
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not worker.cancel(job_id):
        raise HTTPException(409, "该任务当前不能取消")
    return {"ok": True}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    if not database.retry(job_id):
        raise HTTPException(409, "只有失败或已取消的任务可以重试")
    return {"ok": True}


@app.get("/healthz")
def health():
    return {"ok": True}
