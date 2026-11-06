/* vidFrame front-end */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const TERMINAL = new Set(["done", "failed", "cancelled", "interrupted", "paused"]);
  const ACTIVE = new Set(["queued", "running", "extracting", "analyzing", "compiling"]);
  const STAGES = ["Upload", "Extract", "Read frames", "Compile"];
  const STATUS_LABEL = {
    queued: "Queued",
    running: "Running",
    extracting: "Extracting",
    analyzing: "Analyzing",
    compiling: "Compiling",
    done: "Ready",
    failed: "Failed",
    cancelled: "Cancelled",
    interrupted: "Interrupted",
    paused: "Paused",
  };
  const STATUS_CLASS = {
    done: "done",
    queued: "queued",
    running: "running",
    extracting: "running",
    analyzing: "running",
    compiling: "running",
    failed: "failed",
    cancelled: "cancelled",
    interrupted: "interrupted",
    paused: "paused",
  };
  const SETTINGS_KEY = "vidframe.settings.v1";

  const DEFAULTS = {
    fps: 1,
    maxFrames: 900,
    batch: 4,
    timeout: 180,
    provider: "local",
    model: "",
    openrouterModel: "",
    compileModel: "",
  };

  const state = {
    config: null,
    settings: { ...DEFAULTS },
    savedSettings: false,
    models: [],
    vlmHealth: null,
    jobs: [],
    job: null,
    jobId: null,
    lastJobSig: "",
    assetSig: "",
    assetsLoading: false,
    eventSource: null,
    pollTimer: null,
    logIds: new Set(),
    logCount: 0,
    frames: [],
    videoMeta: {},
    sampleFps: 0,
    sceneCount: 0,
    timeline: null,
    noteMarkdown: "",
    noteJson: null,
    tab: "report",
    frameLimit: 120,
    liveFrameSig: "",
    liveNotesSig: "",
    liveNotesText: "",
    liveNotesTimer: null,
    modelsByProvider: { local: [], openrouter: [] },
    lbItems: [],
    lbIndex: 0,
    uploading: false,
    searchTimer: null,
    loadedAssetsFor: "",
    explainerTimer: null,
    explainerSig: "",
  };

  const icons = {
    film: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 5v14M17 5v14M3 9h4M3 15h4M17 9h4M17 15h4"/></svg>`,
    clock: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="12" r="8"/><path d="M12 8v4.4l2.6 1.6" stroke-linecap="round"/></svg>`,
    alert: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 8v5M12 16.5v.1" stroke-linecap="round"/><path d="M10.3 4.2 3.1 17a2 2 0 0 0 1.7 3h14.4a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0Z" stroke-linejoin="round"/></svg>`,
    check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m5 12.5 4.4 4.4L19 7.3" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    info: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8v.1" stroke-linecap="round"/></svg>`,
    refresh: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M20 11.5A8 8 0 0 0 5.6 6.8L4 8.5M4 12.5a8 8 0 0 0 14.4 4.7L20 15.5" stroke-linecap="round"/><path d="M4 4v4.5h4.5M20 20v-4.5h-4.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    download: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 4v11m0 0 4-4m-4 4-4-4M5 19h14" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    spark: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="m12 3 1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3Z" stroke-linejoin="round"/><path d="m18.5 15 .8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2Z" stroke-linejoin="round"/></svg>`,
    diagram: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="3" y="3" width="7" height="6" rx="1"/><rect x="14" y="15" width="7" height="6" rx="1"/><path d="M10 6h4a3 3 0 0 1 3 3v6" stroke-linecap="round"/></svg>`,
  };

  /* ------------------------------- helpers ------------------------------- */

  const esc = (value) =>
    String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  function mathHtml(source, ocr = false) {
    const value = String(source ?? "").replace(/\x08/g, "\\b").replace(/\x0c/g, "\\f");
    const render = (expression, displayMode = false) => {
      if (!window.katex) return esc(expression);
      try {
        return window.katex.renderToString(expression, { displayMode, throwOnError: true, trust: false, output: "html" });
      } catch {
        return `<span class="math-error" title="${esc(expression)}">Equation could not be rendered</span>`;
      }
    };
    // OCR often supplies a whole equation without Markdown delimiters.
    if (ocr && !/\$|\\[([]/.test(value) &&
        ((/[=<>]/.test(value) && !/[.!?]$/.test(value)) ||
         /\\(?:frac|dfrac|sqrt|lim|theta|alpha|beta)\b/.test(value))) {
      return render(value);
    }
    return value.split(/(\$\$[\s\S]+?\$\$|\$[^$\n]+\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\])/g)
      .map((part) => {
        if (part.startsWith("$$") && part.endsWith("$$")) return render(part.slice(2, -2), true);
        if (part.startsWith("$") && part.endsWith("$")) return render(part.slice(1, -1));
        if (part.startsWith("\\(") && part.endsWith("\\)")) return render(part.slice(2, -2));
        if (part.startsWith("\\[") && part.endsWith("\\]")) return render(part.slice(2, -2), true);
        return esc(part);
      }).join("");
  }

  function repairDisplayMath(markdown) {
    // JSON decoding can turn the starts of \beta and \frac into control chars.
    const lines = String(markdown).replace(/\x08/g, "\\b").replace(/\x0c/g, "\\f").split("\n");
    return lines.map((line, index) => {
      const content = line.trim();
      // A one-line display equation with a missing closing $$ is distinct
      // from a multiline equation: it ends at a blank line or the document.
      if (content.startsWith("$$") && content.length > 2 &&
          content.indexOf("$$", 2) === -1 &&
          (index === lines.length - 1 || !lines[index + 1].trim())) {
        return `${line}$$`;
      }
      return line;
    }).join("\n");
  }

  function fmtDuration(seconds) {
    if (seconds == null || Number.isNaN(Number(seconds))) return "—";
    let total = Math.max(0, Math.round(Number(seconds)));
    if (total < 60) return `${total}s`;
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
    return `${m}m ${String(s).padStart(2, "0")}s`;
  }

  function fmtClock(seconds) {
    if (seconds == null || Number.isNaN(Number(seconds))) return "—";
    const total = Math.max(0, Math.round(Number(seconds)));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  function timeAgo(iso) {
    if (!iso) return "";
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return "";
    const diff = Math.max(0, Date.now() - then);
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    return new Date(then).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }

  function fmtDateTime(iso) {
    if (!iso) return "—";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso;
    return date.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function parseError(payload, fallback) {
    if (!payload) return fallback;
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) {
      return payload.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
    }
    return fallback;
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    if (!response.ok) {
      let detail = response.statusText || `HTTP ${response.status}`;
      try {
        detail = parseError(await response.json(), detail);
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    if (response.status === 204) return null;
    return response.json();
  }

  async function apiText(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Could not load ${path}`);
    return response.text();
  }

  function toast(message, type = "") {
    const node = document.createElement("div");
    node.className = `toast ${type}`;
    node.textContent = message;
    $("#toasts").appendChild(node);
    setTimeout(() => {
      node.style.opacity = "0";
      node.style.transform = "translateY(6px)";
      node.style.transition = "opacity .2s ease, transform .2s ease";
      setTimeout(() => node.remove(), 220);
    }, 4200);
  }

  function lockScroll(locked) {
    document.body.classList.toggle("locked", locked);
  }

  /* -------------------------------- boot -------------------------------- */

  document.addEventListener("DOMContentLoaded", boot);

  async function boot() {
    bindEvents();
    loadLocalSettings();
    try {
      state.config = await api("/api/config");
    } catch {
      state.config = null;
    }
    if (!state.savedSettings && state.config) {
      state.settings = {
        ...state.settings,
        fps: state.config.default_fps ?? DEFAULTS.fps,
        maxFrames: state.config.default_max_frames ?? DEFAULTS.maxFrames,
        batch: state.config.default_batch_size ?? DEFAULTS.batch,
        timeout: state.config.request_timeout ?? DEFAULTS.timeout,
        provider: state.config.default_provider ?? DEFAULTS.provider,
        model: state.config.vlm_model || "",
        openrouterModel: state.config.openrouter_model || "",
      };
    }
    applySettingsToUI();
    populateModelList();
    route();
    refreshJobs();
    refreshModels();
  }

  function loadLocalSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        state.savedSettings = true;
        state.settings = { ...DEFAULTS, ...parsed };
      }
    } catch {
      /* ignore */
    }
  }

  function applySettingsToUI() {
    const settings = state.settings;
    const fpsSelect = $("#opt-fps");
    const fpsValue = String(settings.fps ?? 1);
    if (!Array.from(fpsSelect.options).some((option) => option.value === fpsValue)) {
      const option = new Option(`${fpsValue} fps`, fpsValue);
      fpsSelect.appendChild(option);
    }
    fpsSelect.value = fpsValue;
    $("#opt-max").value = settings.maxFrames ?? DEFAULTS.maxFrames;
    $("#opt-batch").value = settings.batch ?? DEFAULTS.batch;
    $("#opt-timeout").value = settings.timeout ?? DEFAULTS.timeout;
    $("#opt-provider").value = settings.provider || DEFAULTS.provider;
    $("#opt-model").value = activeModelValue();
    $("#opt-compile-model").value = settings.compileModel || "";
  }

  function providerLabel(provider) {
    return provider === "openrouter" ? "OpenRouter" : "local";
  }

  function currentProvider() {
    return $("#opt-provider")?.value || state.settings.provider || state.config?.default_provider || "local";
  }

  function activeModelValue() {
    return currentProvider() === "openrouter"
      ? state.settings.openrouterModel || state.config?.openrouter_model || ""
      : state.settings.model || state.config?.vlm_model || "";
  }

  function populateModelList() {
    const list = $("#model-list");
    list.innerHTML = "";
    const provider = currentProvider();
    const models = state.modelsByProvider[provider] || state.models;
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model;
      list.appendChild(option);
    }
  }

  async function refreshModels(manual = false, provider = null) {
    const selected = provider || currentProvider();
    try {
      const health = await api(`/api/vlm/models?provider=${encodeURIComponent(selected)}`);
      health.provider = health.provider || selected;
      state.vlmHealth = health;
      state.modelsByProvider[selected] = health.models || [];
      if (selected === currentProvider()) {
        state.models = health.models || [];
        populateModelList();
      }
      updateVlmStatus(health);
      if ($("#settings-modal").classList.contains("hidden") === false) {
        renderConnCard(health);
      }
      if (manual) {
        if (health.ok) toast(`${providerLabel(selected)} connected · ${health.models.length} model(s) available`, "ok");
        else toast(health.error || `Could not reach the ${providerLabel(selected)} vision endpoint`, "error");
      }
    } catch (error) {
      const health = { ok: false, models: [], error: error.message, provider: selected };
      state.vlmHealth = health;
      state.modelsByProvider[selected] = [];
      if (selected === currentProvider()) {
        state.models = [];
        populateModelList();
      }
      updateVlmStatus(health);
      if (!$("#settings-modal").classList.contains("hidden")) renderConnCard(health);
      if (manual) toast(error.message, "error");
    }
  }

  function updateVlmStatus(health) {
    const dot = $("#vlm-status .dot");
    const text = $("#vlm-status .vlm-text");
    if (!dot || !text) return;
    dot.className = "dot";
    const provider = health.provider || currentProvider();
    if (health.ok) {
      dot.classList.add("dot-ok");
      const model = health.model || (provider === "openrouter" ? state.settings.openrouterModel : state.settings.model);
      text.textContent = `${providerLabel(provider)} · ${(model || "model").split("/").pop().slice(0, 24)}`;
      $("#vlm-status").title = `${providerLabel(provider)}\n${model || ""}\n${health.base_url || ""}`;
    } else {
      dot.classList.add("dot-err");
      text.textContent = `${providerLabel(provider)} offline`;
      $("#vlm-status").title = health.error || "Could not reach the endpoint";
    }
  }

  /* -------------------------------- routing -------------------------------- */

  function route() {
    const hash = location.hash.replace(/^#/, "") || "/";
    const match = hash.match(/^\/job\/([a-zA-Z0-9_-]+)/);
    if (match) openJob(match[1]);
    else showHome();
  }

  function showHome() {
    state.jobId = null;
    state.job = null;
    state.lastJobSig = "";
    closeEvents();
    stopPolling();
    $("#view-job").classList.add("hidden");
    $("#view-home").classList.remove("hidden");
    document.title = "vidFrame — visual video analysis";
    refreshJobs();
  }

  async function openJob(id) {
    if (state.jobId !== id) {
      state.jobId = id;
      state.job = null;
      state.lastJobSig = "";
      state.assetSig = "";
      state.loadedAssetsFor = "";
      state.frames = [];
      state.timeline = null;
      state.noteMarkdown = "";
      state.noteJson = null;
      state.logIds = new Set();
      state.logCount = 0;
      state.liveFrameSig = "";
      state.liveNotesSig = "";
      state.liveNotesText = "";
      state.frameLimit = 120;
      state.tab = "report";
      resetJobUI();
    }
    $("#view-home").classList.add("hidden");
    $("#view-job").classList.remove("hidden");
    closeEvents();
    stopPolling();
    try {
      const job = await api(`/api/jobs/${id}`);
      applyJob(job);
      if (location.hash.endsWith("/episodes")) switchTab("episodes");
      else if (!job.note_available) switchTab("costs");
    } catch (error) {
      toast(error.message, "error");
      location.hash = "#/";
      return;
    }
    startPolling();
    connectEvents();
    await refreshAssets(true);
  }

  function resetJobUI() {
    clearTimeout(state.explainerTimer);
    state.explainerSig = "";
    $("#job-title").textContent = "Loading…";
    $("#job-filename").textContent = "";
    $("#job-chips").innerHTML = "";
    $("#job-banner").innerHTML = "";
    $("#job-content").classList.add("hidden");
    $("#progress-card").classList.remove("hidden");
    $("#note").innerHTML = "";
    $("#toc").innerHTML = "";
    $("#log").innerHTML = "";
    $("#log-count").textContent = "0";
    $("#live-frames").classList.add("hidden");
    $("#live-notes").classList.add("hidden");
    $("#live-notes-body").innerHTML = "";
    $("#live-notes-count").textContent = "0";
    $("#copy-note-btn").hidden = true;
    $("#download-pdf").hidden = true;
    $("#menu-pop").innerHTML = "";
    $("#progress-bar").style.width = "0%";
    $("#progress-pct").textContent = "0%";
    $("#frames-grid").innerHTML = "";
    $("#timeline").innerHTML = "";
    switchTab("report");
  }

  /* -------------------------------- job state -------------------------------- */

  function applyJob(job) {
    if (!job) return;
    state.job = job;
    state.jobId = job.id;
    const signature = JSON.stringify(job);
    if (signature === state.lastJobSig) return;
    state.lastJobSig = signature;

    document.title = `${job.display_title || job.filename} — vidFrame`;
    $("#crumb-title").textContent = job.display_title || job.filename;
    $("#job-title").textContent = job.display_title || job.filename;
    $("#job-filename").textContent = job.filename;

    const statusClass = STATUS_CLASS[job.status] || "";
    $("#job-eyebrow").textContent =
      job.status === "done"
        ? "Visual report"
        : job.status === "failed"
          ? "Analysis failed"
          : TERMINAL.has(job.status)
            ? "Analysis stopped"
            : "Analysis in progress";

    const chips = [];
    if (job.duration) chips.push(`<span class="chip">${icons.clock}${esc(fmtDuration(job.duration))}</span>`);
    if (job.sample_fps) chips.push(`<span class="chip">~<b>${Number(job.sample_fps).toFixed(2)}</b> fps</span>`);
    if (job.frame_count) chips.push(`<span class="chip"><b>${job.frame_count}</b> frames</span>`);
    if (job.scene_count) chips.push(`<span class="chip"><b>${job.scene_count}</b> scenes</span>`);
    if (job.diagram_count) chips.push(`<span class="chip">${icons.diagram}<b>${job.diagram_count}</b> diagram${job.diagram_count === 1 ? "" : "s"}</span>`);
    if (job.model) chips.push(`<span class="chip">${esc(job.model.split("/").pop())}</span>`);
    $("#job-chips").innerHTML = chips.join("");

    renderActions(job);
    renderBanner(job);
    renderProgress(job);

    const hasNote = Boolean(job.note_available);
    $("#job-content").classList.remove("hidden");
    $("#progress-card").classList.toggle("hidden", job.status === "done" && hasNote);

    if (job.frames_available && (!hasNote || ACTIVE.has(job.status))) {
      loadLiveFrames(job);
    }
    if ((job.frames_available || job.observations_available) && (!hasNote || ACTIVE.has(job.status) || job.live_notes_available)) {
      loadLiveNotes(job);
    }

    if (TERMINAL.has(job.status)) {
      closeEvents();
      stopPolling();
      if (job.frames_available || hasNote) refreshAssets(false);
    }
  }

  function renderActions(job) {
    const hasNote = Boolean(job.note_available);
    const busy = Boolean(job.active);
    $("#copy-note-btn").hidden = !hasNote;
    $("#download-pdf").hidden = !job.pdf_available;
    if (job.pdf_available) $("#download-pdf").href = `/api/jobs/${job.id}/download?kind=study-pdf`;

    $("#job-actions").querySelector(".job-controls")?.remove();
    if (busy) {
      $("#job-actions").insertAdjacentHTML(
        "afterbegin",
        `<div class="job-controls"><button class="button" data-action="pause">Pause</button><button class="button danger-outline" data-action="cancel">Cancel</button></div>`
      );
    }

    const dis = (condition) => (condition ? "disabled" : "");
    $("#menu-pop").innerHTML = `
      <div class="menu-label">Export</div>
      ${
        hasNote
          ? `<a href="/api/jobs/${job.id}/download?kind=note-md">${icons.download}Markdown report <span class="kind">.md</span></a>
             <a href="/api/jobs/${job.id}/download?kind=note-json">${icons.spark}Structured note <span class="kind">.json</span></a>`
          : `<span class="menu-empty">Report not ready</span>`
      }
      <a href="/api/jobs/${job.id}/download?kind=observations" class="${job.observations_available ? "" : "is-disabled"}">${icons.film}Frame observations <span class="kind">.jsonl</span></a>
      <a href="/api/jobs/${job.id}/download?kind=frames" class="${job.frames_available ? "" : "is-disabled"}">${icons.film}Frame index <span class="kind">.json</span></a>
      <a href="/api/jobs/${job.id}/download?kind=study-pdf" class="${job.pdf_available ? "" : "is-disabled"}">${icons.download}Study notes <span class="kind">.pdf</span></a>
      <a href="/api/jobs/${job.id}/download?kind=transcript" class="${job.transcript_available ? "" : "is-disabled"}">${icons.download}Transcript <span class="kind">.json</span></a>
      <a href="/api/jobs/${job.id}/download?kind=diagrams" class="${job.diagram_count ? "" : "is-disabled"}">${icons.diagram}Diagram evidence <span class="kind">.zip</span></a>
      <hr />
      <div class="menu-label">Re-extract</div>
      <button type="button" data-action="recompile" ${dis(!job.observations_available || busy)}>${icons.refresh}Recompile note</button>
      <button type="button" data-action="reanalyze" ${dis(!job.frames_available || busy)}>${icons.spark}Re-run vision analysis</button>
      <button type="button" data-action="reprocess" ${dis(!job.source_available || busy)}>${icons.film}Reprocess video…</button>
      <hr />
      <button type="button" class="danger" data-action="delete" ${dis(busy)}>Delete analysis</button>
    `;
  }

  function renderBanner(job) {
    const node = $("#job-banner");
    if (job.status === "failed") {
      node.innerHTML = `
        <div class="banner error">
          <span class="b-icon">${icons.alert}</span>
          <div>
            <strong>Analysis failed</strong>
            <p>${esc(job.error || job.message || "The job stopped unexpectedly.")}</p>
            <div class="banner-actions">
              ${job.source_available ? `<button class="button" data-action="retry">${icons.refresh}Retry</button>` : ""}
              ${job.observations_available ? `<button class="button" data-action="recompile">${icons.refresh}Compile saved frames</button>` : ""}
              <a class="button" href="#/">New analysis</a>
            </div>
          </div>
        </div>`;
      return;
    }
    if (job.status === "paused" || job.status === "cancelled" || job.status === "interrupted") {
      node.innerHTML = `
        <div class="banner info">
          <span class="b-icon">${icons.info}</span>
          <div>
            <strong>${job.status === "paused" ? "Job paused" : job.status === "cancelled" ? "Job cancelled" : "Server restarted"}</strong>
            <p>${esc(job.message || "This analysis did not finish.")} Stored frames and observations are kept, so you can resume without uploading again.</p>
            <div class="banner-actions">
              ${job.status === "paused" || job.status === "interrupted" ? `<button class="button primary" data-action="resume">${icons.refresh}Resume</button>` : ""}
              ${job.frames_available ? `<button class="button" data-action="reanalyze">${icons.spark}Resume analysis</button>` : ""}
              ${job.observations_available ? `<button class="button" data-action="recompile">${icons.refresh}Compile saved frames</button>` : ""}
            </div>
          </div>
        </div>`;
      return;
    }
    if (job.status === "done" && job.failed_count > 0) {
      node.innerHTML = `
        <div class="banner warn">
          <span class="b-icon">${icons.alert}</span>
          <div>
            <strong>${job.failed_count} frame${job.failed_count === 1 ? "" : "s"} could not be analyzed</strong>
            <p>The rest of the report is complete. Failed moments are noted in the report caveats.</p>
          </div>
        </div>`;
      return;
    }
    node.innerHTML = "";
  }

  function stageIndex(job) {
    if (job.status === "done") return 4;
    if (job.stage === "compiling" || job.progress >= 90) return 3;
    if (job.stage === "analyzing" || job.progress >= 40) return 2;
    if (job.stage === "extracting" || job.progress >= 3) return 1;
    return 0;
  }

  function renderProgress(job) {
    const status = job.status;
    const label = STATUS_LABEL[status] || status;
    const current = stageIndex(job);
    const pill = $("#stage-pill");
    pill.className = `stage-pill ${STATUS_CLASS[status] || ""}`;
    pill.textContent = label;
    $("#progress-message").textContent = job.message || "Working…";
    const pct = Math.max(0, Math.min(100, Number(job.progress) || 0));
    $("#progress-pct").textContent = `${Math.round(pct)}%`;
    $("#progress-bar").style.width = `${pct}%`;

    $("#stepper").innerHTML = STAGES.map((name, index) => {
      let cls = "";
      if (status === "done" || index < current) cls = "done";
      else if (index === current) cls = "current";
      return `<div class="step-item ${cls}"><span class="bullet"></span>${esc(name)}</div>`;
    }).join("");

    const meta = [];
    if (job.frame_count) meta.push(`${job.frame_count} frames`);
    if (job.analyzed_count) meta.push(`${job.analyzed_count} analyzed`);
    if (job.failed_count) meta.push(`${job.failed_count} failed`);
    if (job.scene_count) meta.push(`${job.scene_count} scenes`);
    if (job.diagram_count) meta.push(`${job.diagram_count} diagram${job.diagram_count === 1 ? "" : "s"}`);
    $("#progress-counts").textContent = meta.join(" · ");
    $("#progress-eta").textContent = job.started_at ? `Started ${timeAgo(job.started_at)}` : "";
    $("#progress-elapsed").textContent = job.elapsed ? `${job.elapsed.toFixed(1)}s elapsed` : "";
  }

  async function loadLiveFrames(job) {
    const signature = `${job.frame_count}:${job.analyzed_count}`;
    if (state.liveFrameSig === signature) return;
    state.liveFrameSig = signature;
    try {
      const data = await api(`/api/jobs/${job.id}/frames?limit=12&duplicates=false`);
      if (!data.frames.length) return;
      const strip = data.frames
        .map(
          (frame) =>
            `<img loading="lazy" src="/api/jobs/${job.id}/frames/${frame.index}/image" alt="Frame at ${esc(
              fmtClock(frame.time)
            )}" title="${esc(fmtClock(frame.time))}" />`
        )
        .join("");
      $("#live-frames").innerHTML = `<div class="live-label">Frames arriving · latest ${data.frames.length}</div><div class="live-strip">${strip}</div>`;
      $("#live-frames").classList.remove("hidden");
    } catch {
      /* ignore */
    }
  }

  function parseLiveNotes(markdown) {
    const entries = [];
    let summary = "";
    let current = null;
    for (const raw of String(markdown || "").split("\n")) {
      const line = raw.trim();
      if (!line) continue;
      if (line.startsWith("# ")) continue;
      if (/^_.*_$/.test(line)) {
        summary = line.slice(1, -1);
        continue;
      }
      const heading = line.match(/^##\s+(.+)$/);
      if (heading) {
        const parts = heading[1].split(" · ");
        current = {
          time: (parts.shift() || "").trim(),
          title: parts.join(" · ").trim(),
          description: [],
          onScreen: [],
          diagrams: [],
        };
        entries.push(current);
        continue;
      }
      if (!current) continue;
      if (line.startsWith("**On screen:**")) {
        current.onScreen = Array.from(line.slice(13).matchAll(/`([^`]+)`/g)).map((match) => match[1]);
        continue;
      }
      const diagramHead = line.match(/^\*\*Diagram · (.+?) · (.+?)\*\*$/);
      if (diagramHead) {
        current.diagrams.push({ type: diagramHead[1], title: diagramHead[2], description: "", values: [] });
        continue;
      }
      if (line.startsWith("**Values:**")) {
        const diagram = current.diagrams[current.diagrams.length - 1];
        if (diagram) {
          diagram.values = line
            .slice(11)
            .split(";")
            .map((value) => value.trim())
            .filter(Boolean);
        }
        continue;
      }
      const diagram = current.diagrams[current.diagrams.length - 1];
      if (diagram && !diagram.description) {
        diagram.description = line;
        continue;
      }
      current.description.push(line);
    }
    return { entries, summary };
  }

  function renderLiveNotes(markdown) {
    const { entries, summary } = parseLiveNotes(markdown);
    $("#live-notes-count").textContent = String(entries.length);
    const body = $("#live-notes-body");
    const stick = body.scrollTop + body.clientHeight >= body.scrollHeight - 24;
    if (!entries.length) {
      body.innerHTML = `<div class="live-notes-empty">${
        summary ? mathHtml(summary) : "Waiting for the first frame to be analyzed…"
      }</div>`;
      $("#live-notes").classList.remove("hidden");
      return;
    }
    body.innerHTML = entries
      .map((entry) => {
        const description = entry.description.map((line) => `<p class="live-note-desc">${mathHtml(line)}</p>`).join("");
        const texts = entry.onScreen.length
          ? `<div class="live-note-text">${entry.onScreen
              .slice(0, 14)
              .map((text) => `<span>${mathHtml(text, true)}</span>`)
              .join("")}</div>`
          : "";
        const diagrams = entry.diagrams
          .map(
            (diagram) => `
            <div class="live-note-diagram">
              <span class="diagram-badge">${icons.diagram}${esc(diagram.type)}</span>
              <strong>${mathHtml(diagram.title)}</strong>
              ${diagram.description ? `<p class="live-note-desc">${mathHtml(diagram.description)}</p>` : ""}
              ${
                diagram.values.length
                  ? `<div class="live-note-text">${diagram.values
                      .slice(0, 10)
                      .map((value) => `<span>${mathHtml(value, true)}</span>`)
                      .join("")}</div>`
                  : ""
              }
            </div>`
          )
          .join("");
        return `
          <div class="live-note">
            <div class="live-note-top">
              <span class="live-note-time">${esc(entry.time)}</span>
              <span class="live-note-title">${mathHtml(entry.title || "Frame analyzed")}</span>
            </div>
            ${description}
            ${texts}
            ${diagrams}
          </div>`;
      })
      .join("");
    $("#live-notes").classList.remove("hidden");
    if (stick) body.scrollTop = body.scrollHeight;
  }

  async function loadLiveNotes(job, force = false) {
    if (!job) return;
    const signature = `${job.id}:${job.updated_at || ""}:${job.analyzed_count || 0}:${job.diagram_count || 0}:${job.status}`;
    if (!force && state.liveNotesSig === signature) return;
    state.liveNotesSig = signature;
    const jobId = job.id;
    try {
      const text = await apiText(`/api/jobs/${jobId}/live-notes`);
      if (state.jobId !== jobId) return;
      state.liveNotesText = text;
      renderLiveNotes(text);
    } catch {
      if (state.jobId !== jobId) return;
      if (state.liveNotesText) {
        renderLiveNotes(state.liveNotesText);
      } else if (ACTIVE.has(job.status)) {
        renderLiveNotes("");
      } else {
        $("#live-notes").classList.add("hidden");
      }
    }
  }

  function scheduleLiveNotes() {
    clearTimeout(state.liveNotesTimer);
    state.liveNotesTimer = setTimeout(() => {
      if (state.job) loadLiveNotes(state.job);
    }, 900);
  }

  function connectEvents() {
    if (!state.jobId) return;
    closeEvents();
    const source = new EventSource(`/api/jobs/${state.jobId}/events?after=0`);
    state.eventSource = source;

    source.addEventListener("status", (event) => {
      let job;
      try {
        job = JSON.parse(event.data);
      } catch {
        return;
      }
      applyJob(job);
      if (TERMINAL.has(job.status)) {
        closeEvents();
        stopPolling();
        refreshJobs();
      } else {
        startPolling();
      }
    });

    source.addEventListener("log", (event) => {
      try {
        const payload = JSON.parse(event.data);
        appendLog(payload);
        if (["frame", "diagrams", "analyzing"].includes(payload.stage)) scheduleLiveNotes();
      } catch {
        /* ignore */
      }
    });

    source.onerror = () => {
      if (state.eventSource !== source) return;
      if (source.readyState === EventSource.CLOSED) {
        state.eventSource = null;
        startPolling();
      }
    };
  }

  function closeEvents() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
  }

  function startPolling() {
    if (state.pollTimer) return;
    state.pollTimer = setInterval(async () => {
      if (!state.jobId || document.hidden) return;
      try {
        const job = await api(`/api/jobs/${state.jobId}`);
        applyJob(job);
      } catch {
        /* keep trying */
      }
    }, 2500);
  }

  function stopPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function appendLog(event) {
    if (!event || state.logIds.has(event.id)) return;
    state.logIds.add(event.id);
    state.logCount += 1;
    $("#log-count").textContent = String(state.logCount);
    const log = $("#log");
    const row = document.createElement("div");
    row.className = `log-row ${event.level || ""}`;
    const time = new Date(event.ts);
    const timeText = Number.isNaN(time.getTime()) ? "" : time.toLocaleTimeString(undefined, { hour12: false });
    row.innerHTML = `<span class="lt">${esc(timeText)}</span><span class="ls">${esc(event.stage || "job")}</span><span class="lm">${esc(
      event.message
    )}</span>`;
    log.appendChild(row);
    while (log.children.length > 400) log.firstElementChild.remove();
    log.scrollTop = log.scrollHeight;
  }

  /* -------------------------------- assets -------------------------------- */

  async function refreshAssets(force = false) {
    const job = state.job;
    if (!job || state.assetsLoading) return;
    const signature = `${job.id}:${job.updated_at}:${job.note_available}:${job.frame_count}`;
    if (!force && state.assetSig === signature) return;
    state.assetSig = signature;
    state.assetsLoading = true;
    try {
      if (job.frames_available) {
        const data = await api(`/api/jobs/${job.id}/frames?limit=2000&duplicates=true`);
        state.frames = data.frames || [];
        state.videoMeta = data.video || {};
        state.sampleFps = data.sample_fps || 0;
        state.sceneCount = data.scene_count || 0;
        populateSceneFilter();
        renderFrames(true);
        $("#tab-frame-count").textContent = String(state.frames.length);
        state.timeline = await api(`/api/jobs/${job.id}/timeline`).catch(() => null);
        renderTimeline();
      }
      if (job.note_available) {
        const markdown = await apiText(`/api/jobs/${job.id}/note?format=md`);
        if (markdown !== state.noteMarkdown) {
          state.noteMarkdown = markdown;
          renderNote(markdown);
        }
        state.noteJson = await api(`/api/jobs/${job.id}/note?format=json`).catch(() => null);
        renderData();
        $("#job-content").classList.remove("hidden");
      }
    } catch (error) {
      toast(`Could not load job assets: ${error.message}`, "error");
    } finally {
      state.assetsLoading = false;
      state.loadedAssetsFor = signature;
    }
  }

  /* -------------------------------- report -------------------------------- */

  // Capture math before Markdown interprets backslashes and underscores. Code
  // spans and fenced blocks are handled by marked before these inline tokens.
  if (window.marked && window.katex) {
    const mathPlaceholder = (token) =>
      `<span class="math-source" data-math="${encodeURIComponent(token.text)}" data-display="${token.display}"></span>`;
    window.marked.use({
      extensions: [
        {
          name: "displayMath",
          level: "block",
          start: (src) => {
            const match = /(?:^|\n)\$\$/.exec(src);
            return match ? match.index + (src[match.index] === "\n" ? 1 : 0) : -1;
          },
          tokenizer(src) {
            const match = /^\$\$([^\n]+?)\$\$(?:\n|$)/.exec(src) ||
              /^\$\$[ \t]*\n([\s\S]+?)\n[ \t]*\$\$(?:\n|$)/.exec(src);
            if (match) return { type: "displayMath", raw: match[0], text: match[1], display: "true" };
          },
          renderer: mathPlaceholder,
        },
        {
          name: "inlineMath",
          level: "inline",
          start: (src) => src.search(/\$|\\[([]/),
          tokenizer(src) {
            const match = /^(?:\$\$([^\n]+?)\$\$|\$([^\s$](?:[^$\n]*?[^\s$])?)\$|\\\(([^\n]+?)\\\)|\\\[([\s\S]+?)\\\])/.exec(src);
            if (match) return { type: "inlineMath", raw: match[0], text: match[1] ?? match[2] ?? match[3] ?? match[4], display: String(match[1] !== undefined || match[4] !== undefined) };
          },
          renderer: mathPlaceholder,
        },
      ],
    });
  }

  function renderNote(markdown) {
    const note = $("#note");
    if (!markdown.trim()) {
      note.innerHTML = `<div class="empty">The report is empty.</div>`;
      return;
    }
    let html;
    if (window.marked) {
      try {
        html = window.marked.parse(repairDisplayMath(markdown), { gfm: true, breaks: false, mangle: false, headerIds: false });
      } catch {
        html = `<pre>${esc(markdown)}</pre>`;
      }
    } else {
      html = `<pre>${esc(markdown)}</pre>`;
    }
    if (window.DOMPurify) {
      html = window.DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
    }
    note.innerHTML = html;
    note.querySelectorAll(".math-source").forEach((element) => {
      const expression = decodeURIComponent(element.dataset.math);
      try {
        window.katex.render(expression, element, {
          displayMode: element.dataset.display === "true",
          throwOnError: true,
          trust: false,
        });
      } catch {
        element.classList.add("math-error");
        element.title = expression;
        element.textContent = "Equation could not be rendered";
      }
    });
    note.querySelectorAll("code").forEach((code) => {
      if (code.closest("pre") || !/\\(?:frac|dfrac|sqrt|lim|sum|int|alpha|beta|theta)\b/.test(code.textContent)) return;
      const math = document.createElement("span");
      try {
        window.katex.render(code.textContent, math, { throwOnError: true, trust: false });
        code.replaceWith(math);
      } catch {
        // Keep uncertain OCR as source text instead of inventing a formula.
      }
    });
    // The job header already shows the report title; avoid rendering it twice.
    const duplicateTitle = note.querySelector("h1");
    if (duplicateTitle) duplicateTitle.remove();
    note.querySelectorAll("a").forEach((anchor) => {
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
    });
    buildToc(note);
    $("#copy-note-btn").hidden = false;
  }

  function buildToc(note) {
    const toc = $("#toc");
    const headings = Array.from(note.querySelectorAll("h2, h3"));
    toc.innerHTML = "";
    headings.forEach((heading, index) => {
      const id = `sec-${index}`;
      heading.id = id;
      const link = document.createElement("a");
      link.href = `#${id}`;
      // Keep KaTeX and other inline formatting from the rendered heading.
      // textContent exposes KaTeX's underlying LaTeX source in the contents.
      for (const child of heading.childNodes) link.appendChild(child.cloneNode(true));
      link.querySelectorAll("a").forEach((nestedLink) => {
        const span = document.createElement("span");
        span.replaceChildren(...nestedLink.childNodes);
        nestedLink.replaceWith(span);
      });
      link.dataset.target = id;
      if (heading.tagName === "H3") link.classList.add("sub");
      link.addEventListener("click", (event) => {
        event.preventDefault();
        heading.scrollIntoView({ behavior: "smooth", block: "start" });
      });
      toc.appendChild(link);
    });
    if (!headings.length) toc.innerHTML = `<span class="toc-empty">No sections</span>`;
  }

  /* -------------------------------- timeline -------------------------------- */

  function renderTimeline() {
    const data = state.timeline;
    const root = $("#timeline");
    if (!data || !data.scenes || !data.scenes.length) {
      root.innerHTML = `<div class="empty">No timeline data yet.</div>`;
      return;
    }
    const header = `<div class="timeline-head"><h2>Visual timeline</h2><span>${data.scenes.length} scenes · ${
      state.frames.length
    } frames</span></div>`;
    const cards = data.scenes
      .map((scene) => {
        const preview = scene.frames.slice(0, 6);
        const thumbs = preview
          .map(
            (frame) => `
            <button class="scene-thumb ${frame.duplicate ? "dupe" : ""}" type="button"
              data-scene="${scene.scene}" data-frame="${frame.index}" title="${esc(
                frame.time_label
              )} · ${esc(frame.description || "frame")}">
              <img loading="lazy" src="/api/jobs/${state.jobId}/frames/${frame.index}/image" alt="Frame at ${esc(
                frame.time_label
              )}" />
              <span class="t">${esc(frame.time_label)}</span>
            </button>`
          )
          .join("");
        const extra =
          scene.frames.length > preview.length
            ? `<div class="scene-thumb more">+${scene.frames.length - preview.length}</div>`
            : "";
        const texts = (scene.screen_text || [])
          .slice(0, 6)
          .map((text) => `<span class="text-chip">${mathHtml(text, true)}</span>`)
          .join("");
        const description =
          (scene.frames.find((frame) => frame.description) || {}).description || "No description available for this scene.";
        const diagramCount = scene.frames.filter((frame) => frame.has_diagram || frame.diagram).length;
        return `
          <div class="scene-card">
            <div class="scene-head">
              <span class="scene-num">Scene ${String(scene.scene + 1).padStart(2, "0")}</span>
              <span class="scene-title">${mathHtml(scene.title || "Visual segment")}</span>
              ${diagramCount ? `<span class="scene-diagram">${icons.diagram}${diagramCount}</span>` : ""}
              <span class="scene-time">${esc(scene.start_label)} – ${esc(scene.end_label)}</span>
            </div>
            <div class="scene-body">
              <div>
                <p class="scene-desc">${mathHtml(description)}</p>
                ${texts ? `<div class="scene-texts">${texts}</div>` : ""}
              </div>
              <div class="scene-strip">${thumbs}${extra}</div>
            </div>
          </div>`;
      })
      .join("");
    root.innerHTML = header + cards;
  }

  /* -------------------------------- frames -------------------------------- */

  function populateSceneFilter() {
    const select = $("#scene-filter");
    const previous = select.value;
    const scenes = new Map();
    for (const frame of state.frames) {
      if (!scenes.has(frame.scene)) scenes.set(frame.scene, frame.time);
    }
    select.innerHTML = `<option value="">All scenes</option>`;
    Array.from(scenes.entries())
      .sort((a, b) => a[0] - b[0])
      .forEach(([scene, time]) => {
        const option = new Option(`Scene ${scene + 1} · ${fmtClock(time)}`, String(scene));
        select.appendChild(option);
      });
    if (Array.from(select.options).some((option) => option.value === previous)) select.value = previous;
  }

  function filteredFrames() {
    const query = $("#frame-search").value.trim().toLowerCase();
    const scene = $("#scene-filter").value;
    const showDupes = $("#show-dupes").checked;
    return state.frames.filter((frame) => {
      if (!showDupes && frame.duplicate) return false;
      if (scene !== "" && String(frame.scene) !== scene) return false;
      if (!query) return true;
      const obs = frame.observation || {};
      const haystack = [
        obs.scene,
        obs.summary,
        obs.description,
        obs.setting,
        obs.visual_notes,
        ...(obs.screen_text || []),
        ...(obs.objects || []),
        ...(obs.people || []),
        ...(obs.diagram ? [obs.diagram.title, obs.diagram.description, ...(obs.diagram.values || []), ...(obs.diagram.labels || [])] : []),
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query) || `frame ${frame.index}`.includes(query);
    });
  }

  function renderFrames(reset = false) {
    if (reset) state.frameLimit = 120;
    const frames = filteredFrames();
    const page = frames.slice(0, state.frameLimit);
    const grid = $("#frames-grid");
    grid.innerHTML = page
      .map((frame) => {
        const obs = frame.observation || {};
        const title = obs.scene || obs.summary || `Frame ${frame.index}`;
        const desc =
          obs.description ||
          (frame.duplicate ? "Near-duplicate of an earlier frame." : obs.error || "No description yet.");
        return `
          <button class="frame-card ${frame.duplicate ? "duplicate" : ""}" type="button" data-index="${frame.index}">
            <div class="frame-media">
              <img loading="lazy" src="/api/jobs/${state.jobId}/frames/${frame.index}/image" alt="Frame at ${esc(
                fmtClock(frame.time)
              )}" />
              <span class="frame-time">${esc(fmtClock(frame.time))}</span>
              <span class="frame-scene">S${frame.scene + 1}</span>
              ${obs.has_diagram || obs.diagram ? `<span class="frame-diagram">${icons.diagram}diagram</span>` : ""}
              ${frame.duplicate ? `<span class="frame-dupe">duplicate</span>` : ""}
            </div>
            <div class="frame-body">
              <div class="frame-scene-title">${mathHtml(title)}</div>
              <div class="frame-desc">${mathHtml(desc)}</div>
            </div>
          </button>`;
      })
      .join("");
    $("#frames-count").textContent = `${frames.length} of ${state.frames.length} frames`;
    $("#frames-empty").classList.toggle("hidden", frames.length !== 0);
    $("#load-more").classList.toggle("hidden", frames.length <= state.frameLimit);
  }

  /* -------------------------------- lightbox -------------------------------- */

  function toLbItem(frame, sceneTitle = "") {
    const obs = frame.observation || {};
    return {
      index: frame.index,
      time: frame.time,
      time_label: frame.time_label || fmtClock(frame.time),
      scene: frame.scene,
      sceneTitle: sceneTitle || obs.scene || obs.summary || "",
      description: obs.description || frame.description || obs.summary || "",
      observation: obs,
      screenText: (obs.screen_text || frame.screen_text || []).filter(Boolean),
    };
  }

  function openLightbox(items, index) {
    if (!items.length) return;
    state.lbItems = items;
    state.lbIndex = Math.max(0, Math.min(index, items.length - 1));
    $("#lightbox").classList.remove("hidden");
    lockScroll(true);
    renderLightbox();
  }

  function renderLightbox() {
    const item = state.lbItems[state.lbIndex];
    if (!item) return;
    const obs = item.observation || {};
    const image = $("#lb-image");
    image.src = `/api/jobs/${state.jobId}/frames/${item.index}/image`;
    image.alt = `Frame at ${item.time_label}`;
    $("#lb-time").textContent = item.time_label;
    $("#lb-scene").textContent = `Scene ${item.scene + 1}`;
    const title = item.sceneTitle || `Frame ${item.index}`;
    const sections = [];
    if (obs.setting) sections.push(`<h4>Setting</h4><p>${mathHtml(obs.setting)}</p>`);
    const listSection = (label, values) =>
      values && values.length
        ? `<h4>${esc(label)}</h4><ul>${values.slice(0, 20).map((value) => `<li>${mathHtml(value, label === "On-screen text" || label === "Diagram details")}</li>`).join("")}</ul>`
        : "";
    sections.push(listSection("People", obs.people));
    sections.push(listSection("Objects", obs.objects));
    sections.push(listSection("Actions", obs.actions));
    sections.push(listSection("On-screen text", item.screenText));
    const diagram = obs.diagram;
    if (diagram) {
      const values = [];
      if (diagram.type) values.push(`Type: ${diagram.type}`);
      if (diagram.values?.length) values.push(...diagram.values);
      if (diagram.labels?.length) values.push(...diagram.labels.map((label) => `Label: ${label}`));
      if (diagram.edges?.length) values.push(...diagram.edges.map((edge) => `Edge: ${edge}`));
      const crop = diagram.image
        ? `<img class="lb-diagram" loading="lazy" src="/api/jobs/${state.jobId}/${String(diagram.image).replace(/^\/+/, "")}" alt="${esc(
            diagram.title || "Diagram crop"
          )}" />`
        : "";
      sections.push(
        `<h4>Diagram</h4>${diagram.description ? `<p>${mathHtml(diagram.description)}</p>` : ""}${crop}${listSection(
          diagram.title || "Diagram details",
          values
        )}`
      );
    }
    if (obs.visual_notes) sections.push(`<h4>Visual notes</h4><p>${mathHtml(obs.visual_notes)}</p>`);
    if (obs.error) sections.push(`<h4>Error</h4><p>${esc(obs.error)}</p>`);
    $("#lb-details").innerHTML = `
      <h3>${mathHtml(title)}</h3>
      <div class="lb-sub">Frame ${item.index} · ${esc(item.time_label)} · ${esc(obs.status || "pending")}</div>
      <p>${mathHtml(item.description || "No description available.")}</p>
      ${sections.join("")}`;
    $("#lb-prev").disabled = state.lbIndex === 0;
    $("#lb-next").disabled = state.lbIndex >= state.lbItems.length - 1;
  }

  function closeLightbox() {
    $("#lightbox").classList.add("hidden");
    $("#lb-image").removeAttribute("src");
    lockScroll(false);
  }

  function stepLightbox(delta) {
    const next = state.lbIndex + delta;
    if (next < 0 || next >= state.lbItems.length) return;
    state.lbIndex = next;
    renderLightbox();
  }

  function openFrameByIndex(index) {
    const frames = filteredFrames();
    const position = frames.findIndex((frame) => frame.index === index);
    openLightbox(frames.map((frame) => toLbItem(frame)), position >= 0 ? position : 0);
  }

  function openSceneFrame(sceneNumber, frameIndex) {
    const scene = (state.timeline?.scenes || []).find((item) => item.scene === sceneNumber);
    if (!scene) return;
    const items = scene.frames.map((frame) => ({
      index: frame.index,
      time: frame.time,
      time_label: frame.time_label,
      scene: scene.scene,
      sceneTitle: scene.title || frame.description,
      description: frame.description,
      observation: {
        scene: scene.title,
        description: frame.description,
        screen_text: frame.screen_text,
        status: frame.status,
        people: [],
        objects: [],
        actions: [],
      },
      screenText: frame.screen_text || [],
    }));
    const position = items.findIndex((item) => item.index === frameIndex);
    openLightbox(items, position >= 0 ? position : 0);
  }

  /* -------------------------------- tabs -------------------------------- */

  function switchTab(tab) {
    if (tab !== "episodes") $("#tab-episodes video")?.pause();
    if (tab !== "explainers") {
      $("#tab-explainers video")?.pause();
      clearTimeout(state.explainerTimer);
    }
    state.tab = tab;
    $$(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
    $$(".tabpane").forEach((pane) => pane.classList.toggle("hidden", pane.id !== `tab-${tab}`));
    if (tab === "episodes") renderEpisodes();
    if (tab === "explainers") renderExplainers();
    if (tab === "data") renderData();
    if (tab === "costs") renderCosts();
  }

  async function renderExplainers() {
    const job = state.job;
    if (!job) return;
    clearTimeout(state.explainerTimer);
    const status = $("#explainer-status");
    const history = $("#explainer-history");
    const form = $("#explainer-form");
    try {
      const data = await api(`/api/jobs/${job.id}/explainers`);
      if (state.job?.id !== job.id || state.tab !== "explainers") return;
      form.hidden = !data.available;
      status.textContent = data.available ? "" : "This lecture needs completed notes, a transcript, and teacher speech before an explainer can be generated.";
      const items = data.items || [];
      const sig = JSON.stringify(items);
      if (sig !== state.explainerSig) {
        state.explainerSig = sig;
        history.innerHTML = items.length ? items.map((item) => {
          const base = `<div class="explainer-card-head"><h3>${mathHtml(item.topic)}</h3><span>${esc(item.status)} · ${Math.round(item.progress || 0)}%</span></div>`;
          const detail = item.status === "failed" ? `<p class="explainer-error">${esc(item.error || "Generation failed")}</p>` :
            `<p>${mathHtml(item.message || "")}</p>`;
          const media = item.status === "done" ? `<video controls preload="metadata" src="${item.playback_url}" aria-label="Explainer: ${esc(item.topic)}"></video><a class="button" href="${item.download_url}">Download MP4</a>` : "";
          return `<article class="explainer-card">${base}${detail}${media}</article>`;
        }).join("") : `<p class="explainer-empty">No explainers yet. Enter a topic from the lecture notes to make one.</p>`;
      }
      if (items.some((item) => item.status === "queued" || item.status === "running")) {
        state.explainerTimer = setTimeout(renderExplainers, 3000);
      }
    } catch (error) {
      if (state.job?.id === job.id) status.textContent = error.message;
    }
  }

  function episodeEvidence(group) {
    const speech = [group?.speech || []].flat();
    const items = speech.map(s => ({time: s.start, text: s.text, label: "Transcript"}));
    for (const f of group?.visuals || []) {
      items.push({time: f.time, text: [...(f.actions || []), f.summary].filter(Boolean).join(" "), label: "Frame " + f.index});
    }
    return items.sort((a, b) => a.time - b.time);
  }

  async function renderEpisodes() {
    const job = state.job;
    if (!job) return;
    const root = $("#tab-episodes");
    root.textContent = "Loading teaching episodes…";
    try {
      const {available, episodes, running, error} = await api(`/api/jobs/${job.id}/teaching-episodes`);
      if (state.job?.id !== job.id || state.tab !== "episodes") return;
      if (!episodes.length) {
        if (running) {
          root.textContent = "Extracting teaching episodes from the saved transcript and frame observations…";
          setTimeout(() => {
            if (state.job?.id === job.id && state.tab === "episodes") renderEpisodes();
          }, 4000);
        } else if (available) {
          root.textContent = "The extractor found no candidate episodes.";
        } else if (job.transcript_available && job.observations_available) {
          root.innerHTML = `<p>${esc(error || "Generate teaching episodes from this lecture’s saved transcript and visual observations.")}</p>
            <button class="button" type="button">Generate teaching episodes</button>`;
          root.querySelector("button").onclick = async event => {
            event.currentTarget.disabled = true;
            try {
              await api(`/api/jobs/${job.id}/teaching-episodes`, {method: "POST"});
              if (state.job?.id === job.id && state.tab === "episodes") renderEpisodes();
            } catch (err) {
              toast(err.message, "error");
              if (state.job?.id === job.id && state.tab === "episodes") renderEpisodes();
            }
          };
        } else {
          root.textContent = "Teaching episodes require a saved transcript and frame observations.";
        }
        return;
      }
      const player = job.source_available ? '<video controls preload="none" style="width:100%;max-height:40vh" aria-label="Lecture video"></video>' : "";
      root.innerHTML = player + `<h3>${episodes.length} candidate teaching episodes</h3>` + episodes.map(row => {
        const target = episodeEvidence(row.target);
        const time = target[0]?.time;
        const jump = job.source_available && Number.isFinite(time)
          ? `<button class="button" type="button" data-seek="${time}">Play at ${fmtClock(time)}</button>` : "";
        const evidence = [["Lead-up", row.context], ["Teaching move evidence", row.target], ["Follow-up", row.followup]].map(([label, group]) => {
          const items = episodeEvidence(group);
          if (!items.length) return "";
          return `<h4>${label}</h4>${items.map(item => `<p>${fmtClock(item.time)} · ${esc(item.label)} — ${mathHtml(item.text)}</p>`).join("")}`;
        }).join("");
        return `<article class="data-card"><h3>${esc(row.target.moves.map(m => m.move.replaceAll("_", " ")).join(", "))}</h3>
          <p>${mathHtml(row.target.moves.map(m => m.method || m.cue).join(" "))}</p>${jump}
          <details><summary>Source evidence</summary>${evidence}</details></article>`;
      }).join("");
      const video = root.querySelector("video");
      if (video) video.src = `/api/jobs/${job.id}/source-video`;
      root.onclick = event => {
        const button = event.target.closest("[data-seek]");
        if (!button || !video) return;
        video.currentTime = Number(button.dataset.seek);
        video.scrollIntoView({block: "center"});
        video.play().catch(() => toast("Could not play the source video.", "error"));
      };
    } catch (error) {
      if (state.job?.id === job.id) root.textContent = error.message;
    }
  }

  async function renderCosts() {
    const job = state.job;
    if (!job) return;
    const root = $("#costs-view");
    root.innerHTML = '<div class="data-card"><p>Loading costs…</p></div>';
    try {
      const { items = [] } = await api(`/api/jobs/${job.id}/costs`);
      if (state.job?.id !== job.id || state.tab !== "costs") return;
      const groups = new Map();
      for (const item of items) {
        const key = `${item.stage}|${item.provider}|${item.model}`;
        if (!groups.has(key)) groups.set(key, { ...item, calls: 0, prompt: 0, completion: 0, seconds: 0, cost: 0, priced: 0 });
        const group = groups.get(key);
        group.calls++;
        group.prompt += item.prompt_tokens || 0;
        group.completion += item.completion_tokens || 0;
        group.seconds += item.audio_seconds || 0;
        if (item.cost_usd !== null) { group.cost += item.cost_usd; group.priced++; }
      }
      const total = [...groups.values()].reduce((sum, group) => sum + group.cost, 0);
      const priced = items.filter((item) => item.cost_usd !== null).length;
      const transcription = groups.size && [...groups.values()].some((group) => group.stage === "Transcription");
      const money = (value) => `$${value.toFixed(value < 0.01 ? 6 : 4)}`;
      root.innerHTML = `<div class="data-grid"><div class="data-card"><h3>Provider charges</h3><dl class="kv">
        <dt>Reported total</dt><dd>${priced ? money(total) : "—"}</dd>
        <dt>Model requests</dt><dd>${items.length}</dd>
        <dt>Priced requests</dt><dd>${priced} of ${items.length}</dd>
        <dt>Input tokens</dt><dd>${items.reduce((sum, item) => sum + (item.prompt_tokens || 0), 0).toLocaleString()}</dd>
        <dt>Output tokens</dt><dd>${items.reduce((sum, item) => sum + (item.completion_tokens || 0), 0).toLocaleString()}</dd>
      </dl><p>Charges come from provider response usage. Unpriced requests are excluded from the reported total.${!transcription && job.transcript_available ? " No transcription charge was recorded for this run." : ""}</p></div>
      <div class="data-card"><h3>Breakdown</h3>${groups.size ? [...groups.values()].map((group) => `<div class="cost-row">
        <strong>${esc(group.stage)}</strong><span>${esc(group.provider)} / ${esc(group.model)}</span>
        <span>${group.calls} request${group.calls === 1 ? "" : "s"}${group.seconds ? ` · ${group.seconds.toFixed(1)} audio seconds` : ""}${group.prompt || group.completion ? ` · ${group.prompt.toLocaleString()} input · ${group.completion.toLocaleString()} output tokens` : ""}</span>
        <b>${group.priced ? money(group.cost) : "Charge unavailable"}${group.priced < group.calls ? ` · ${group.calls - group.priced} unpriced` : ""}</b>
      </div>`).join("") : "<p>No model requests logged for this extraction yet.</p>"}</div></div>`;
    } catch (error) {
      root.innerHTML = `<div class="data-card"><p>Could not load costs: ${esc(error.message)}</p></div>`;
    }
  }

  /* -------------------------------- data tab -------------------------------- */

  function renderData() {
    const job = state.job;
    if (!job) return;
    const note = state.noteJson;
    const root = $("#data-view");
    const config = job.config || {};
    root.innerHTML = `
      <div class="data-grid">
        <div class="data-card">
          <h3>Analysis</h3>
          <dl class="kv">
            <dt>Job ID</dt><dd>${esc(job.id)}</dd>
            <dt>Source</dt><dd>${esc(job.filename)}</dd>
            <dt>Status</dt><dd>${esc(STATUS_LABEL[job.status] || job.status)}</dd>
            <dt>Duration</dt><dd>${esc(fmtDuration(job.duration))}</dd>
            <dt>Sample rate</dt><dd>${job.sample_fps ? `~${Number(job.sample_fps).toFixed(2)} fps` : "—"}</dd>
            <dt>Frames</dt><dd>${job.frame_count || 0} sampled · ${job.analyzed_count || 0} analyzed</dd>
            <dt>Scenes</dt><dd>${job.scene_count || 0}</dd>
            <dt>Diagrams</dt><dd>${job.diagram_count || 0}</dd>
            <dt>Provider</dt><dd>${esc(config.provider || "local")}</dd>
            <dt>Vision model</dt><dd>${esc(config.model || job.model || "—")}</dd>
            <dt>Diagram analysis model</dt><dd>${esc(config.diagram_model || config.model || job.model || "—")}</dd>
            <dt>Compile model</dt><dd>${esc(config.compile_model || "—")}</dd>
            <dt>Created</dt><dd>${esc(fmtDateTime(job.created_at))}</dd>
            <dt>Finished</dt><dd>${esc(fmtDateTime(job.finished_at))}</dd>
          </dl>
        </div>

        <div class="data-card">
          <h3>Exports</h3>
          <p>Every layer is stored separately. Re-extract a note from observations, or run the vision model again on the same frames without re-uploading.</p>
          <div class="data-links">
            <a href="/api/jobs/${job.id}/download?kind=note-md" class="${job.note_available ? "" : "is-disabled"}">Markdown report <span class="kind">.md</span></a>
            <a href="/api/jobs/${job.id}/download?kind=note-json" class="${job.note_available ? "" : "is-disabled"}">Structured note <span class="kind">.json</span></a>
            <a href="/api/jobs/${job.id}/download?kind=observations" class="${job.observations_available ? "" : "is-disabled"}">Frame observations <span class="kind">.jsonl</span></a>
            <a href="/api/jobs/${job.id}/download?kind=frames" class="${job.frames_available ? "" : "is-disabled"}">Frame index <span class="kind">.json</span></a>
            <a href="/api/jobs/${job.id}/download?kind=diagrams" class="${job.diagram_count ? "" : "is-disabled"}">Diagram images <span class="kind">.zip</span></a>
          </div>
        </div>

        <div class="data-card">
          <h3>Recompile</h3>
          <p>Rebuild the report from stored observations with the current compile model. No frames are re-read and no vision calls are needed.</p>
          <div class="action-row">
            <button class="button" data-action="recompile" ${!job.observations_available || job.active ? "disabled" : ""}>Recompile note</button>
          </div>
        </div>

        <div class="data-card">
          <h3>Re-extract</h3>
          <p>Run vision again on the stored frames, or reprocess the source video with a new sample rate and frame cap.</p>
          <div class="action-row">
            <button class="button" data-action="reanalyze" ${!job.frames_available || job.active ? "disabled" : ""}>Re-run vision</button>
            <button class="button" data-action="reprocess" ${!job.source_available || job.active ? "disabled" : ""}>Reprocess video…</button>
          </div>
        </div>
      </div>

      <details class="json-box">
        <summary>Structured note JSON ${note ? `· ${Object.keys(note).length} keys` : ""}</summary>
        <pre>${esc(note ? JSON.stringify(note, null, 2) : "No structured note available yet.")}</pre>
      </details>

      <details class="json-box">
        <summary>Job configuration</summary>
        <pre>${esc(JSON.stringify(job.config || {}, null, 2))}</pre>
      </details>
    `;
  }

  /* -------------------------------- actions -------------------------------- */

  async function postAction(action, payload = {}) {
    const job = state.job;
    if (!job) return;
    const paths = {
      recompile: `/api/jobs/${job.id}/recompile`,
      reanalyze: `/api/jobs/${job.id}/reanalyze`,
      reprocess: `/api/jobs/${job.id}/reprocess`,
      resume: `/api/jobs/${job.id}/resume`,
    };
    try {
      await api(paths[action], { method: "POST", body: JSON.stringify(payload) });
      state.lastJobSig = "";
      state.assetSig = "";
      state.liveFrameSig = "";
      toast(
        action === "recompile" ? "Recompiling from stored observations" : action === "reanalyze" ? "Re-running vision analysis" : action === "resume" ? "Resuming saved progress" : "Reprocessing video",
        "ok"
      );
      setTimeout(async () => {
        try {
          applyJob(await api(`/api/jobs/${job.id}`));
        } catch {
          /* ignore */
        }
      }, 350);
      connectEvents();
      startPolling();
      refreshJobs();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async function handleAction(action) {
    const job = state.job;
    if (!job) return;
    $("#job-menu").removeAttribute("open");
    if (action === "recompile") {
      await postAction("recompile", {});
    } else if (action === "reanalyze") {
      if (!confirm("Re-run the vision model on every stored frame? This replaces the current observations and report.")) return;
      await postAction("reanalyze", {});
    } else if (action === "reprocess" || action === "retry") {
      const current = job.config?.sample_fps || 1;
      const entered = action === "retry" ? current : prompt("Sample rate in frames per second (e.g. 1 = one frame per second):", String(current));
      if (entered === null) return;
      const fps = Number(entered);
      if (!Number.isFinite(fps) || fps < 0.05 || fps > 10) {
        toast("Sample rate must be between 0.05 and 10 fps", "error");
        return;
      }
      await postAction("reprocess", { sample_fps: fps });
    } else if (action === "resume") {
      await postAction("resume", {});
    } else if (action === "delete") {
      if (!confirm(`Delete “${job.display_title || job.filename}” and all extracted frames? This cannot be undone.`)) return;
      try {
        await api(`/api/jobs/${job.id}`, { method: "DELETE" });
        toast("Analysis deleted", "ok");
        location.hash = "#/";
        refreshJobs();
      } catch (error) {
        toast(error.message, "error");
      }
    } else if (action === "pause" || action === "cancel") {
      try {
        await api(`/api/jobs/${job.id}/${action}`, { method: "POST" });
        toast(action === "pause" ? "Pause requested…" : "Cancelling…");
      } catch (error) {
        toast(error.message, "error");
      }
    }
  }

  /* -------------------------------- upload -------------------------------- */

  async function uploadFile(file) {
    if (!file) return;
    if (state.uploading) return;
    const allowed = /\.(mp4|mov|mkv|webm|avi|m4v|mpeg|mpg|ts)$/i;
    if (!allowed.test(file.name)) {
      toast("Unsupported file type. Choose an MP4, MOV, MKV, WebM, AVI, or TS video.", "error");
      return;
    }
    if (
      $("#opt-provider").value === "openrouter" &&
      state.config &&
      state.config.openrouter_configured === false &&
      state.config.vlm_mode !== "mock"
    ) {
      toast("OpenRouter is selected but no server API key is configured. Add VIDFRAME_OPENROUTER_API_KEY or choose Local.", "error");
      return;
    }
    const maxMb = state.config?.max_upload_mb || 4096;
    if (file.size > maxMb * 1024 * 1024) {
      toast(`File is larger than the ${maxMb} MB upload limit.`, "error");
      return;
    }

    state.uploading = true;
    const dropzone = $("#dropzone");
    dropzone.classList.add("uploading");
    $("#uploading-state").classList.remove("hidden");
    $("#upload-name").textContent = file.name;
    $("#upload-pct").textContent = "0%";
    $("#upload-bar").style.width = "0%";

    const form = new FormData();
    form.append("file", file);
    form.append("sample_fps", $("#opt-fps").value);
    form.append("max_frames", $("#opt-max").value);
    form.append("batch_size", $("#opt-batch").value);
    form.append("request_timeout", $("#opt-timeout").value);
    form.append("provider", $("#opt-provider").value);
    if ($("#opt-model").value.trim()) form.append("model", $("#opt-model").value.trim());
    if ($("#opt-compile-model").value.trim()) form.append("compile_model", $("#opt-compile-model").value.trim());
    if ($("#opt-stt-language")?.value.trim()) form.append("stt_language", $("#opt-stt-language").value.trim());
    if ($("#opt-transcript-appendix")?.checked) form.append("include_transcript_appendix", "true");

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.round((event.loaded / event.total) * 100);
      $("#upload-pct").textContent = `${percent}%`;
      $("#upload-bar").style.width = `${percent}%`;
    };
    xhr.onload = () => {
      let payload = null;
      try {
        payload = JSON.parse(xhr.responseText);
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300 && payload?.id) {
        toast("Upload complete · analysis started", "ok");
        resetUploadUI();
        refreshJobs();
        location.hash = `#/job/${payload.id}`;
      } else {
        toast(parseError(payload, `Upload failed (HTTP ${xhr.status})`), "error");
        resetUploadUI();
      }
    };
    xhr.onerror = () => {
      toast("Upload failed. Is the server running?", "error");
      resetUploadUI();
    };
    xhr.send(form);
  }

  function resetUploadUI() {
    state.uploading = false;
    $("#dropzone").classList.remove("uploading");
    $("#uploading-state").classList.add("hidden");
    $("#file-input").value = "";
  }

  /* -------------------------------- jobs list -------------------------------- */

  async function refreshJobs() {
    try {
      const data = await api("/api/jobs?limit=100");
      state.jobs = data.jobs || [];
      renderSidebar();
      renderRecent();
    } catch {
      /* ignore while server is unavailable */
    }
  }

  function jobStatusDot(status) {
    if (status === "done") return "dot-ok";
    if (status === "failed") return "dot-err";
    if (ACTIVE.has(status)) return "dot-run";
    return "dot-idle";
  }

  function renderSidebar() {
    const root = $("#job-list");
    if (!state.jobs.length) {
      root.innerHTML = `<div class="sidebar-empty">No analyses yet</div>`;
      return;
    }
    root.innerHTML = state.jobs
      .slice(0, 30)
      .map(
        (job) => `
        <a class="job-item ${job.id === state.jobId ? "active" : ""}" href="#/job/${job.id}">
          <span class="dot ${jobStatusDot(job.status)}"></span>
          <span class="job-item-body">
            <span class="job-item-title">${esc(job.display_title || job.filename)}</span>
            <span class="job-item-meta">${esc(STATUS_LABEL[job.status] || job.status)} · ${esc(
              job.duration ? fmtDuration(job.duration) : timeAgo(job.created_at)
            )}</span>
          </span>
        </a>`
      )
      .join("");
  }

  function renderRecent() {
    const root = $("#recent-grid");
    if (!state.jobs.length) {
      root.innerHTML = `<div class="recent-empty">Nothing analyzed yet. Drop a video above to build your first visual report.</div>`;
      return;
    }
    root.innerHTML = state.jobs
      .slice(0, 12)
      .map((job) => {
        const thumb = job.frames_available
          ? `<img loading="lazy" src="/api/jobs/${job.id}/frames/0/image" alt="" onerror="this.remove()" />`
          : icons.film;
        return `
          <a class="recent-card" href="#/job/${job.id}">
            <span class="recent-thumb">${thumb}</span>
            <span class="recent-body">
              <span class="recent-title">${esc(job.display_title || job.filename)}</span>
              <span class="recent-meta">${esc(
                [job.duration ? fmtDuration(job.duration) : "", job.frame_count ? `${job.frame_count} frames` : "", timeAgo(job.created_at)]
                  .filter(Boolean)
                  .join(" · ")
              )}</span>
            </span>
            <span class="recent-status"><span class="status-badge ${STATUS_CLASS[job.status] || ""}">${esc(
              STATUS_LABEL[job.status] || job.status
            )}</span></span>
          </a>`;
      })
      .join("");
  }

  /* -------------------------------- settings -------------------------------- */

  function openSettings() {
    const settings = state.settings;
    $("#set-provider").value = settings.provider || DEFAULTS.provider;
    $("#set-model").value = settings.model || "";
    $("#set-openrouter-model").value = settings.openrouterModel || state.config?.openrouter_model || "";
    updateSettingsEndpoint();
    $("#set-fps").value = String(settings.fps ?? 1);
    if (!Array.from($("#set-fps").options).some((option) => option.value === $("#set-fps").value)) {
      $("#set-fps").appendChild(new Option(`${settings.fps} fps`, String(settings.fps)));
      $("#set-fps").value = String(settings.fps);
    }
    $("#set-max").value = settings.maxFrames ?? DEFAULTS.maxFrames;
    $("#set-batch").value = settings.batch ?? DEFAULTS.batch;
    $("#set-compile-model").value = settings.compileModel || "";
    if (state.vlmHealth) renderConnCard(state.vlmHealth);
    $("#settings-modal").classList.remove("hidden");
    lockScroll(true);
    refreshModels(false, $("#set-provider").value);
  }

  function selectedProviderBase(provider) {
    if (provider === "openrouter") return state.config?.openrouter_base_url || "https://openrouter.ai/api/v1";
    return state.config?.vlm_base_url || "—";
  }

  function updateSettingsEndpoint() {
    const provider = $("#set-provider").value || DEFAULTS.provider;
    const base = selectedProviderBase(provider);
    const suffix = provider === "openrouter"
      ? state.config?.openrouter_configured
        ? " · key configured"
        : " · key missing"
      : "";
    $("#set-base").value = base + suffix;
  }

  function renderConnCard(health) {
    const node = $("#settings-status");
    const dot = health.ok ? "dot-ok" : "dot-err";
    const provider = health.provider || $("#set-provider").value || currentProvider();
    const title = health.ok
      ? `${providerLabel(provider)} vision endpoint connected`
      : `${providerLabel(provider)} vision endpoint unavailable`;
    const detail = health.ok
      ? `${health.models.length} model(s) · ${health.latency_ms ?? 0} ms · ${health.base_url || selectedProviderBase(provider)}`
      : health.error || "Could not reach the endpoint";
    node.innerHTML = `<span class="dot ${dot}"></span><div><strong>${esc(title)}</strong><p>${esc(detail)}</p></div>`;
  }

  function saveSettings() {
    const fps = Number($("#set-fps").value);
    const settings = {
      fps: Number.isFinite(fps) ? fps : DEFAULTS.fps,
      maxFrames: Number($("#set-max").value) || DEFAULTS.maxFrames,
      batch: Number($("#set-batch").value) || DEFAULTS.batch,
      timeout: state.settings.timeout || DEFAULTS.timeout,
      provider: $("#set-provider").value || DEFAULTS.provider,
      model: $("#set-model").value.trim(),
      openrouterModel: $("#set-openrouter-model").value.trim() || state.config?.openrouter_model || "",
      compileModel: $("#set-compile-model").value.trim(),
    };
    state.settings = settings;
    state.savedSettings = true;
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch {
      /* ignore */
    }
    applySettingsToUI();
    populateModelList();
    refreshModels(false, settings.provider);
    closeModal("settings-modal");
    toast("Defaults saved", "ok");
  }

  function closeModal(id) {
    $(`#${id}`).classList.add("hidden");
    if ($("#settings-modal").classList.contains("hidden") && $("#lightbox").classList.contains("hidden")) {
      lockScroll(false);
    }
  }

  /* -------------------------------- bindings -------------------------------- */

  function bindEvents() {
    window.addEventListener("hashchange", route);

    // upload
    const dropzone = $("#dropzone");
    const fileInput = $("#file-input");
    dropzone.addEventListener("click", (event) => {
      if (state.uploading) return;
      if (event.target === fileInput) return;
      fileInput.click();
    });
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        if (!state.uploading) fileInput.click();
      }
    });
    fileInput.addEventListener("change", () => uploadFile(fileInput.files?.[0]));
    for (const name of ["dragenter", "dragover"]) {
      dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        if (!state.uploading) dropzone.classList.add("dragover");
      });
    }
    for (const name of ["dragleave", "dragend"]) {
      dropzone.addEventListener(name, () => dropzone.classList.remove("dragover"));
    }
    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
      uploadFile(event.dataTransfer?.files?.[0]);
    });

    $("#advanced-toggle").addEventListener("click", () => {
      const panel = $("#advanced");
      const open = panel.classList.toggle("hidden") === false;
      $("#advanced-toggle").setAttribute("aria-expanded", String(open));
    });

    $("#refresh-jobs").addEventListener("click", refreshJobs);
    $("#settings-btn").addEventListener("click", openSettings);
    $("#save-settings").addEventListener("click", saveSettings);
    $("#test-vlm").addEventListener("click", () => refreshModels(true, $("#set-provider").value));
    $("#set-provider").addEventListener("change", () => {
      updateSettingsEndpoint();
      refreshModels(false, $("#set-provider").value);
    });
    $("#opt-provider").addEventListener("change", () => {
      state.settings.provider = $("#opt-provider").value;
      $("#opt-model").value = activeModelValue();
      populateModelList();
      refreshModels(false, $("#opt-provider").value);
    });
    $("#vlm-status").addEventListener("click", openSettings);

    // modals
    $$("[data-close]").forEach((button) =>
      button.addEventListener("click", () => closeModal(button.dataset.close))
    );
    $$(".modal").forEach((modal) =>
      modal.addEventListener("click", (event) => {
        if (event.target === modal) closeModal(modal.id);
      })
    );

    // job actions
    $("#job-actions").addEventListener("click", (event) => {
      const button = event.target.closest("[data-action]");
      if (button) {
        event.preventDefault();
        handleAction(button.dataset.action);
      }
      if (!event.target.closest(".menu")) $("#job-menu").removeAttribute("open");
    });
    $("#job-menu").addEventListener("click", (event) => {
      const link = event.target.closest("a.is-disabled");
      if (link) event.preventDefault();
    });
    document.addEventListener("click", (event) => {
      if (!event.target.closest(".menu")) $("#job-menu").removeAttribute("open");
      const action = event.target.closest("[data-action]");
      if (action && !event.target.closest("#job-actions")) handleAction(action.dataset.action);
    });

    $("#copy-note-btn").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(state.noteMarkdown);
        toast("Report copied to clipboard", "ok");
      } catch {
        toast("Could not copy to clipboard", "error");
      }
    });

    // tabs
    $$(".tab").forEach((button) =>
      button.addEventListener("click", () => switchTab(button.dataset.tab))
    );
    $("#explainer-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const job = state.job;
      const topic = $("#explainer-topic").value.trim();
      if (!job || !topic) return;
      const button = $("#explainer-form button[type=submit]");
      button.disabled = true;
      $("#explainer-status").textContent = "Starting explainer…";
      try {
        await api(`/api/jobs/${job.id}/explainers`, {
          method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({topic}),
        });
        $("#explainer-topic").value = "";
        state.explainerSig = "";
        await renderExplainers();
      } catch (error) {
        $("#explainer-status").textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });

    // frames
    $("#frame-search").addEventListener("input", () => {
      clearTimeout(state.searchTimer);
      state.searchTimer = setTimeout(() => renderFrames(true), 160);
    });
    $("#scene-filter").addEventListener("change", () => renderFrames(true));
    $("#show-dupes").addEventListener("change", () => renderFrames(true));
    $("#load-more").addEventListener("click", () => {
      state.frameLimit += 120;
      renderFrames(false);
    });
    $("#frames-grid").addEventListener("click", (event) => {
      const card = event.target.closest(".frame-card");
      if (card) openFrameByIndex(Number(card.dataset.index));
    });
    $("#timeline").addEventListener("click", (event) => {
      const thumb = event.target.closest(".scene-thumb[data-frame]");
      if (thumb) openSceneFrame(Number(thumb.dataset.scene), Number(thumb.dataset.frame));
    });

    // lightbox
    $("#lb-close").addEventListener("click", closeLightbox);
    $("#lb-prev").addEventListener("click", () => stepLightbox(-1));
    $("#lb-next").addEventListener("click", () => stepLightbox(1));
    $("#lightbox").addEventListener("click", (event) => {
      if (event.target === $("#lightbox")) closeLightbox();
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        if (!$("#lightbox").classList.contains("hidden")) closeLightbox();
        else if (!$("#settings-modal").classList.contains("hidden")) closeModal("settings-modal");
      }
      if ($("#lightbox").classList.contains("hidden")) return;
      if (event.key === "ArrowLeft") stepLightbox(-1);
      if (event.key === "ArrowRight") stepLightbox(1);
    });
  }
})();
