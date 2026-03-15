(() => {
  const $ = (sel) => document.querySelector(sel);

  const fileInput = $("#file-input");
  const dropZone = $("#drop-zone");
  const fileNameEl = $("#file-name");
  const startBtn = $("#start-btn");
  const sourceLang = $("#source-lang");
  const targetLang = $("#target-lang");

  const uploadSection = $("#upload-section");
  const processingSection = $("#processing-section");
  const resultSection = $("#result-section");
  const errorSection = $("#error-section");

  const statusText = $("#status-text");
  const transcriptArea = $("#transcript-area");
  const transcriptList = $("#transcript-list");

  const videoPlayer = $("#video-player");
  const playBtn = $("#play-btn");
  const downloadLink = $("#download-link");
  const errorText = $("#error-text");

  let selectedFile = null;

  const STEP_ORDER = ["extracting", "transcribing", "translating", "synthesizing", "building", "done"];

  // --- Upload handling ---

  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    if (!file.type.startsWith("video/")) {
      alert("Please select a video file.");
      return;
    }
    selectedFile = file;
    fileNameEl.textContent = file.name;
    startBtn.disabled = false;
  }

  // --- Start dubbing ---

  startBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    startBtn.disabled = true;
    startBtn.textContent = "Uploading...";

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const resp = await fetch("/video/upload", { method: "POST", body: formData });
      if (!resp.ok) throw new Error("Upload failed");

      const { job_id } = await resp.json();

      showSection("processing");
      connectWebSocket(job_id);
    } catch (err) {
      showError("Upload error: " + err.message);
    }
  });

  // --- Section visibility ---

  function showSection(name) {
    uploadSection.hidden = name !== "upload";
    processingSection.hidden = name !== "processing";
    resultSection.hidden = name !== "result";
    errorSection.hidden = name !== "error";
  }

  function showError(msg) {
    errorText.textContent = msg;
    showSection("error");
  }

  // --- Step indicator ---

  function activateStep(stepName) {
    const stepIdx = STEP_ORDER.indexOf(stepName);
    if (stepIdx === -1) return;

    for (let i = 0; i < STEP_ORDER.length; i++) {
      const el = $(`#step-${STEP_ORDER[i]}`);
      if (!el) continue;

      el.classList.remove("active", "done");
      if (i < stepIdx) {
        el.classList.add("done");
      } else if (i === stepIdx) {
        el.classList.add("active");
      }
    }
  }

  // --- WebSocket ---

  function connectWebSocket(jobId) {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${location.host}/video/dub/${jobId}`);

    ws.onopen = () => {
      statusText.textContent = "Connected. Starting pipeline...";
      ws.send(
        JSON.stringify({
          source_lang: sourceLang.value,
          target_lang: targetLang.value,
        })
      );
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    };

    ws.onerror = () => showError("WebSocket connection error.");
    ws.onclose = () => {};
  }

  function handleMessage(msg) {
    switch (msg.type) {
      case "step":
        activateStep(msg.step);
        statusText.textContent = msg.message || "";

        if (msg.segments) {
          showTranscript(msg.segments);
        }
        break;

      case "complete":
        activateStep("done");
        statusText.textContent = msg.message;
        showResult(msg.download_url);
        break;

      case "error":
        showError(msg.message);
        break;
    }
  }

  // --- Transcript display ---

  function showTranscript(segments) {
    transcriptArea.hidden = false;
    transcriptList.innerHTML = "";

    for (const seg of segments) {
      const item = document.createElement("div");
      item.className = "transcript-item";
      item.innerHTML = `
        <div class="time">${fmtTime(seg.start)} - ${fmtTime(seg.end)}</div>
        <div class="original">${seg.original}</div>
        <div class="translated">${seg.translated}</div>
      `;
      transcriptList.appendChild(item);
    }
  }

  function fmtTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  // --- Result ---

  function showResult(downloadUrl) {
    showSection("result");
    videoPlayer.src = downloadUrl;
    downloadLink.href = downloadUrl;

    playBtn.addEventListener("click", () => {
      videoPlayer.play();
    });
  }
})();
