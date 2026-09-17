// app.js -- loads the precomputed STA data (sta_data.json, one entry per
// RGC) and renders an interactive per-cell explorer: a clickable grid of
// all 27 cells, a scrubbable spatial receptive-field map, a time-course
// plot at the peak pixel, and the full (lags x pixels) STA heatmap.
//
// Everything here is PRECOMPUTED data (see 02_compute_all_and_export.py) --
// there's no live model inference in this app, unlike the pixel project's
// characterizer. That's the right design for this data: an STA is a fixed
// property of a recorded cell, not something with free parameters to explore.

let DATA = null;
let selectedCell = 0;
let playTimer = null;

const RDBU = (t) => {
  // Diverging red-blue colormap, t in [-1, 1]. Matches matplotlib's
  // RdBu_r closely enough for this purpose: blue = negative (OFF),
  // red = positive (ON), white = zero.
  t = Math.max(-1, Math.min(1, t));
  if (t >= 0) {
    const v = Math.round(255 * (1 - t));
    return `rgb(255,${v},${v})`;
  } else {
    const v = Math.round(255 * (1 + t));
    return `rgb(${v},${v},255)`;
  }
};

async function main() {
  DATA = await fetch("sta_data.json").then((r) => r.json());

  document.getElementById("n-cells").textContent = DATA.n_cells;
  document.getElementById("n-frames").textContent = DATA.n_frames.toLocaleString();
  const totalSpikes = DATA.cells.reduce((s, c) => s + c.n_spikes, 0);
  document.getElementById("n-spikes-total").textContent = totalSpikes.toLocaleString();

  buildCellList();
  selectCell(0);

  document.getElementById("lag-slider").addEventListener("input", (e) => {
    renderSpatial(parseInt(e.target.value, 10));
  });
  document.getElementById("play-btn").addEventListener("click", togglePlay);
}

function buildCellList() {
  const root = document.getElementById("cell-list");
  root.innerHTML = "";
  DATA.cells.forEach((c) => {
    const div = document.createElement("div");
    div.className = "cell-thumb " + (c.polarity.startsWith("OFF") ? "off" : "on");
    div.id = `thumb-${c.cell_index}`;

    const canvas = document.createElement("canvas");
    canvas.width = DATA.nx;
    canvas.height = DATA.ny;
    drawSpatialToCanvas(canvas, c.sta[c.peak_lag], DATA.nx, DATA.ny);

    const label = document.createElement("div");
    label.className = "label";
    label.textContent = `#${c.matlab_cellnum} ${c.polarity.startsWith("OFF") ? "OFF" : "ON"}`;

    div.appendChild(canvas);
    div.appendChild(label);
    div.addEventListener("click", () => selectCell(c.cell_index));
    root.appendChild(div);
  });
}

function drawSpatialToCanvas(canvas, flatValues, nx, ny) {
  const ctx = canvas.getContext("2d");
  const vmax = Math.max(1e-9, Math.max(...flatValues.map(Math.abs)));
  const img = ctx.createImageData(nx, ny);
  for (let i = 0; i < flatValues.length; i++) {
    const t = flatValues[i] / vmax;
    const color = RDBU(t);
    const [r, g, b] = color.match(/\d+/g).map(Number);
    img.data[i * 4 + 0] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function selectCell(idx) {
  selectedCell = idx;
  document.querySelectorAll(".cell-thumb").forEach((el) => el.classList.remove("selected"));
  document.getElementById(`thumb-${idx}`)?.classList.add("selected");

  const c = DATA.cells[idx];
  document.getElementById("detail-title").textContent =
    `Cell #${c.matlab_cellnum} — ${c.polarity}`;
  document.getElementById("detail-sub").textContent =
    `${c.n_spikes.toLocaleString()} spikes over ${DATA.n_frames.toLocaleString()} frames · ` +
    `peak response ${c.peak_value.toFixed(4)} at lag ${c.peak_lag}, pixel ${c.peak_pixel}`;

  document.getElementById("lag-slider").value = c.peak_lag;
  renderSpatial(c.peak_lag);
  renderTimeCourse();
  renderFullSTA();
}

function renderSpatial(lag) {
  const c = DATA.cells[selectedCell];
  document.getElementById("lag-readout").textContent =
    `lag ${lag} ${lag === c.peak_lag ? "(peak)" : ""}`;

  const canvas = document.getElementById("spatial-canvas");
  const ctx = canvas.getContext("2d");
  const small = document.createElement("canvas");
  small.width = DATA.nx;
  small.height = DATA.ny;
  drawSpatialToCanvas(small, c.sta[lag], DATA.nx, DATA.ny);

  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
}

function renderTimeCourse() {
  const c = DATA.cells[selectedCell];
  const values = c.sta.map((row) => row[c.peak_pixel]);
  const canvas = document.getElementById("time-canvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  const padL = 36, padR = 10, padT = 10, padB = 20;
  const plotW = w - padL - padR, plotH = h - padT - padB;

  ctx.clearRect(0, 0, w, h);
  const vmax = Math.max(...values.map(Math.abs)) * 1.15;
  const px = (i) => padL + (i / (values.length - 1)) * plotW;
  const py = (v) => padT + plotH / 2 - (v / vmax) * (plotH / 2);

  // zero line + grid
  ctx.strokeStyle = "#262d38";
  ctx.beginPath(); ctx.moveTo(padL, py(0)); ctx.lineTo(w - padR, py(0)); ctx.stroke();
  ctx.fillStyle = "#7d8896";
  ctx.font = "9px monospace";
  ctx.fillText(vmax.toFixed(2), 2, padT + 4);
  ctx.fillText((-vmax).toFixed(2), 2, h - padB);
  ctx.fillText("0", padL - 10, h - 4);
  ctx.fillText(String(values.length - 1), w - padR - 10, h - 4);

  // trace
  const isOff = c.polarity.startsWith("OFF");
  ctx.strokeStyle = isOff ? getCSSVar("--off") : getCSSVar("--on");
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = px(i), y = py(v);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // peak marker
  ctx.fillStyle = ctx.strokeStyle;
  const peakX = px(c.peak_lag), peakY = py(values[c.peak_lag]);
  ctx.beginPath(); ctx.arc(peakX, peakY, 3.5, 0, 2 * Math.PI); ctx.fill();
}

function renderFullSTA() {
  const c = DATA.cells[selectedCell];
  const canvas = document.getElementById("full-canvas");
  const ctx = canvas.getContext("2d");
  const nLags = DATA.n_lags, nPix = DATA.n_pixels;

  const small = document.createElement("canvas");
  small.width = nPix;
  small.height = nLags;
  const sctx = small.getContext("2d");
  const vmax = Math.max(...c.sta.flat().map(Math.abs)) || 1e-9;
  const img = sctx.createImageData(nPix, nLags);
  for (let row = 0; row < nLags; row++) {
    for (let col = 0; col < nPix; col++) {
      const t = c.sta[row][col] / vmax;
      const color = RDBU(t);
      const [r, g, b] = color.match(/\d+/g).map(Number);
      const idx = (row * nPix + col) * 4;
      img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
    }
  }
  sctx.putImageData(img, 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
}

function togglePlay() {
  const btn = document.getElementById("play-btn");
  const slider = document.getElementById("lag-slider");
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    btn.textContent = "▶ PLAY";
    return;
  }
  btn.textContent = "⏸ PAUSE";
  playTimer = setInterval(() => {
    let v = (parseInt(slider.value, 10) + 1) % (DATA.n_lags);
    slider.value = v;
    renderSpatial(v);
  }, 220);
}

function getCSSVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#ff6b6b;padding:24px;">Failed to load: ${err.stack || err}</pre>`;
  console.error(err);
});
