"use strict";

const $ = (selector) => document.querySelector(selector);
const state = { jobs: [], currentBrowsePath: "", browserParent: null, detailJobId: null };
const statusNames = {
  queued: "等待中", pending: "等待中", running: "运行中", processing: "运行中",
  completed: "已完成", done: "已完成", failed: "失败", cancelled: "已取消", canceled: "已取消"
};
const modelNames = {
  starsample_v2_lite: "StarSample V2 Lite",
  animesr_v2: "AnimeSR v2"
};

async function api(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) }
  });
  const text = await response.text();
  let data = null;
  if (text) {
    try { data = JSON.parse(text); } catch (_) { data = text; }
  }
  if (!response.ok) {
    const message = data?.detail || data?.message || `请求失败（${response.status}）`;
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }
  return data;
}

function normalizeStatus(value) {
  const raw = String(value || "queued").toLowerCase();
  if (raw === "pending") return "queued";
  if (raw === "processing") return "running";
  if (raw === "done") return "completed";
  if (raw === "canceled") return "cancelled";
  return raw;
}

function jobId(job) { return job.id ?? job.job_id; }
function jobPath(job) { return job.input_path || job.input || job.source || "未知输入"; }
function fileName(path) { return String(path).replace(/[\\/]+$/, "").split(/[\\/]/).pop() || path; }
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function progressOf(job) {
  let progress = Number(job.progress ?? job.percent ?? 0);
  if (progress > 0 && progress <= 1) progress *= 100;
  return Math.max(0, Math.min(100, Number.isFinite(progress) ? progress : 0));
}

function formatDuration(value) {
  if (value === null || value === undefined || value === "") return "--";
  if (typeof value === "string" && /[a-z一-龥:]/i.test(value)) return value;
  const seconds = Math.max(0, Number(value));
  if (!Number.isFinite(seconds)) return "--";
  if (seconds < 60) return `${Math.round(seconds)} 秒`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分 ${Math.round(seconds % 60)} 秒`;
  return `${Math.floor(seconds / 3600)} 小时 ${Math.round((seconds % 3600) / 60)} 分`;
}

function formatBytes(value) {
  if (value === null || value === undefined || value === "") return "";
  const bytes = Number(value);
  if (!Number.isFinite(bytes)) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = bytes, unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${size.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function renderJobs() {
  const body = $("#jobsBody");
  $("#emptyState").classList.toggle("visible", state.jobs.length === 0);
  body.innerHTML = state.jobs.map((job) => {
    const id = jobId(job);
    const status = normalizeStatus(job.status);
    const progress = progressOf(job);
    const rate = job.processing_fps || job.speed || job.fps || job.rate;
    const rateText = rate ? (typeof rate === "number" ? `${rate.toFixed(2)} 帧/秒` : String(rate)) : "--";
    const eta = job.eta_seconds ?? job.eta ?? job.remaining_seconds;
    const canCancel = status === "queued" || status === "running";
    const canRetry = status === "failed" || status === "cancelled";
    const actions = canCancel
      ? `<button class="danger" data-action="cancel" data-id="${escapeHtml(id)}">取消</button>`
      : canRetry ? `<button class="secondary" data-action="retry" data-id="${escapeHtml(id)}">重试</button>` : "";
    return `<tr data-id="${escapeHtml(id)}" tabindex="0">
      <td><div class="job-name" title="${escapeHtml(jobPath(job))}">${escapeHtml(fileName(jobPath(job)))}</div><div class="job-path">#${escapeHtml(id)} · ${escapeHtml(modelNames[job.model] || job.model || modelNames.starsample_v2_lite)} · ${escapeHtml(jobPath(job))}</div></td>
      <td><span class="badge ${escapeHtml(status)}">${escapeHtml(statusNames[status] || status)}</span></td>
      <td><div class="progress"><div class="progress-track"><div class="progress-bar" style="width:${progress}%"></div></div><div class="progress-label">${progress.toFixed(1)}%${job.current_frame ? ` · ${escapeHtml(job.current_frame)} 帧` : ""}</div></div></td>
      <td><div class="rate"><div>${escapeHtml(rateText)}</div><div>剩余 ${escapeHtml(formatDuration(eta))}</div></div></td>
      <td><div class="job-actions">${actions}</div></td>
    </tr>`;
  }).join("");
}

function renderCounts(serverStatus = {}) {
  const counts = { running: 0, queued: 0, completed: 0, failed: 0 };
  state.jobs.forEach((job) => {
    const status = normalizeStatus(job.status);
    if (status in counts) counts[status] += 1;
  });
  const serverCounts = serverStatus.counts || {};
  $("#runningCount").textContent = serverCounts.running ?? serverStatus.running ?? serverStatus.running_jobs ?? counts.running;
  $("#queuedCount").textContent = serverCounts.queued ?? serverStatus.queued ?? serverStatus.queued_jobs ?? counts.queued;
  $("#completedCount").textContent = serverCounts.completed ?? serverStatus.completed ?? serverStatus.completed_jobs ?? counts.completed;
  $("#failedCount").textContent = serverCounts.failed ?? serverStatus.failed ?? serverStatus.failed_jobs ?? counts.failed;
  const gpu = serverStatus.gpu || serverStatus.gpu_status;
  $("#gpuStatus").textContent = typeof gpu === "object"
    ? [gpu.name, gpu.memory_used && gpu.memory_total ? `${gpu.memory_used}/${gpu.memory_total}` : null].filter(Boolean).join(" · ") || "可用"
    : gpu || (serverStatus.gpu_available === false ? "不可用" : "就绪");
}

async function refreshAll(showErrors = false) {
  try {
    const [jobsData, statusData] = await Promise.all([api("/api/jobs"), api("/api/status")]);
    state.jobs = Array.isArray(jobsData) ? jobsData : (jobsData?.jobs || jobsData?.items || []);
    renderJobs();
    renderCounts(statusData || {});
    $("#lastUpdated").textContent = `更新于 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
    $("#serviceState").className = "service-state online";
    $("#serviceState").lastElementChild.textContent = "服务正常";
    if (state.detailJobId && $("#detailDialog").open) await loadDetail(state.detailJobId, false);
  } catch (error) {
    $("#serviceState").className = "service-state offline";
    $("#serviceState").lastElementChild.textContent = "连接失败";
    if (showErrors) toast(error.message);
  }
}

async function submitJob(event) {
  event.preventDefault();
  const button = $("#submitJob");
  const message = $("#formMessage");
  button.disabled = true;
  message.textContent = "";
  try {
    await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({
        input_path: $("#inputPath").value.trim(),
        output_subdir: $("#outputSubdir").value.trim(),
        recursive: $("#recursive").checked,
        cq: Number($("#cq").value),
        model: document.querySelector('input[name="model"]:checked').value
      })
    });
    toast("任务已加入队列");
    await refreshAll(true);
  } catch (error) {
    message.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

async function jobAction(action, id, button) {
  button.disabled = true;
  try {
    await api(`/api/jobs/${encodeURIComponent(id)}/${action}`, { method: "POST", body: "{}" });
    toast(action === "cancel" ? "已请求取消任务" : "任务已重新加入队列");
    await refreshAll(true);
  } catch (error) {
    toast(error.message);
    button.disabled = false;
  }
}

function normalizeBrowse(data, requestedPath) {
  const entries = Array.isArray(data) ? data : (data?.entries || data?.items || data?.children || []);
  return {
    path: data?.path || data?.current_path || requestedPath,
    parent: data?.parent || data?.parent_path || null,
    entries: entries.map((entry) => typeof entry === "string"
      ? { name: fileName(entry), path: entry, is_dir: false }
      : { ...entry, is_dir: entry.is_dir ?? entry.directory ?? entry.is_directory ?? (entry.type === "directory") })
  };
}

async function browse(path) {
  const list = $("#browserList");
  list.innerHTML = '<div class="browser-loading">正在读取目录...</div>';
  try {
    const result = normalizeBrowse(await api(`/api/browse?path=${encodeURIComponent(path)}`), path);
    state.currentBrowsePath = result.path;
    state.browserParent = result.parent;
    $("#browserPath").textContent = result.path || "媒体根目录";
    $("#browserUp").disabled = result.parent === null || result.parent === undefined;
    if (!result.entries.length) {
      list.innerHTML = '<div class="browser-loading">目录为空</div>';
      return;
    }
    list.innerHTML = result.entries.map((entry) => {
      const entryPath = entry.path || `${result.path.replace(/\/$/, "")}/${entry.name}`;
      return `<button class="browser-entry" type="button" data-path="${escapeHtml(entryPath)}" data-directory="${entry.is_dir ? "true" : "false"}">
        <span class="entry-kind">${entry.is_dir ? "目录" : "文件"}</span>
        <span class="entry-name">${escapeHtml(entry.name || fileName(entryPath))}</span>
        <span class="entry-size">${escapeHtml(formatBytes(entry.size))}</span>
      </button>`;
    }).join("");
  } catch (error) {
    list.innerHTML = `<div class="browser-loading">${escapeHtml(error.message)}</div>`;
  }
}

function choosePath(path) {
  $("#inputPath").value = path;
  $("#browserDialog").close();
}

async function loadDetail(id, showErrors = true) {
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
    state.detailJobId = id;
    const status = normalizeStatus(job.status);
    $("#detailTitle").textContent = fileName(jobPath(job));
    $("#detailMeta").textContent = `任务 #${id} · ${statusNames[status] || status}`;
    const details = [
      ["输入", jobPath(job)], ["输出", job.output_path || job.output || "--"],
      ["模型", modelNames[job.model] || job.model || modelNames.starsample_v2_lite],
      ["进度", `${progressOf(job).toFixed(1)}%`], ["编码质量", job.cq ?? "--"],
      ["已处理帧", job.current_frame ?? job.processed_frames ?? "--"], ["总帧数", job.total_frames ?? "--"],
      ["开始时间", job.started_at || "--"], ["结束时间", job.finished_at || job.completed_at || "--"]
    ];
    $("#detailGrid").innerHTML = details.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
    const logs = job.logs ?? job.log ?? job.error ?? "暂无日志";
    $("#detailLog").textContent = Array.isArray(logs) ? logs.join("\n") : String(logs);
  } catch (error) {
    if (showErrors) toast(error.message);
  }
}

let toastTimer;
function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("visible"), 2800);
}

$("#jobForm").addEventListener("submit", submitJob);
$("#refreshButton").addEventListener("click", () => refreshAll(true));
$("#browseButton").addEventListener("click", () => {
  const path = $("#inputPath").value.trim() || state.currentBrowsePath;
  $("#browserDialog").showModal();
  browse(path);
});
$("#browserUp").addEventListener("click", () => (state.browserParent !== null && state.browserParent !== undefined) && browse(state.browserParent));
$("#selectFolder").addEventListener("click", () => choosePath(state.currentBrowsePath));
$("#browserList").addEventListener("click", (event) => {
  const entry = event.target.closest(".browser-entry");
  if (!entry) return;
  if (entry.dataset.directory === "true") browse(entry.dataset.path);
  else choosePath(entry.dataset.path);
});
$("#jobsBody").addEventListener("click", async (event) => {
  const actionButton = event.target.closest("button[data-action]");
  if (actionButton) {
    event.stopPropagation();
    await jobAction(actionButton.dataset.action, actionButton.dataset.id, actionButton);
    return;
  }
  const row = event.target.closest("tr[data-id]");
  if (row) {
    await loadDetail(row.dataset.id);
    $("#detailDialog").showModal();
  }
});
$("#jobsBody").addEventListener("keydown", async (event) => {
  if ((event.key === "Enter" || event.key === " ") && event.target.matches("tr[data-id]")) {
    event.preventDefault();
    await loadDetail(event.target.dataset.id);
    $("#detailDialog").showModal();
  }
});
$("#detailRefresh").addEventListener("click", () => state.detailJobId && loadDetail(state.detailJobId));
$("#detailDialog").addEventListener("close", () => { state.detailJobId = null; });

refreshAll();
setInterval(() => refreshAll(false), 2000);
