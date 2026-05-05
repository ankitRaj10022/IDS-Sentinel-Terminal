const healthPill = document.getElementById("health-pill");
const datasetSummary = document.getElementById("dataset-summary");
const legacyResults = document.getElementById("legacy-results");
const jobList = document.getElementById("job-list");
const runList = document.getElementById("run-list");
const emptyTemplate = document.getElementById("empty-state");

function cloneEmpty() {
  return emptyTemplate.content.cloneNode(true);
}

function number(value) {
  return Intl.NumberFormat().format(value);
}

function pct(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function formatWhen(value) {
  if (!value) {
    return "Pending";
  }
  return new Date(value).toLocaleString();
}

function setHealth(ok) {
  healthPill.textContent = ok ? "Online" : "Offline";
  healthPill.parentElement.style.borderColor = ok ? "rgba(95, 224, 197, 0.28)" : "rgba(243, 109, 109, 0.28)";
}

function fillNode(target, nodes) {
  target.innerHTML = "";
  if (!nodes.length) {
    target.appendChild(cloneEmpty());
    return;
  }
  nodes.forEach((node) => target.appendChild(node));
}

function datasetCard(title, item) {
  const wrapper = document.createElement("div");
  wrapper.className = "dataset-box";
  wrapper.innerHTML = `
    <h3>${title}</h3>
    <div class="meta-grid">
      <div><span>Path</span><strong>${item.path}</strong></div>
      <div><span>Rows</span><strong>${number(item.rows)}</strong></div>
      <div><span>Columns</span><strong>${item.columns}</strong></div>
      <div><span>Size</span><strong>${item.size_mb} MB</strong></div>
    </div>
    <p class="inline-note">Labels: ${Object.entries(item.label_counts)
      .map(([label, count]) => `${label}=${number(count)}`)
      .join(" | ")}</p>
  `;
  return wrapper;
}

function renderDatasets(datasets) {
  const nodes = [
    datasetCard("Classical Train", datasets.classical_train),
    datasetCard("Classical Test", datasets.classical_test),
    datasetCard("DNN Train Copy", datasets.dnn_train),
    datasetCard("DNN Test Copy", datasets.dnn_test),
  ];

  const duplicate = document.createElement("div");
  duplicate.className = "dataset-box";
  duplicate.innerHTML = `
    <h3>Duplicate Check</h3>
    <p class="inline-note">Train files match: <strong>${datasets.duplicates.train_files_match}</strong></p>
    <p class="inline-note">Test files match: <strong>${datasets.duplicates.test_files_match}</strong></p>
  `;
  nodes.push(duplicate);
  fillNode(datasetSummary, nodes);
}

function renderLegacy(legacy) {
  const groups = [
    { title: "Classical Legacy Artifacts", entries: legacy.classical },
    { title: "DNN Legacy Artifacts", entries: legacy.dnn },
  ];

  const nodes = groups.map((group) => {
    const wrapper = document.createElement("div");
    wrapper.className = "legacy-box";
    const rows = group.entries
      .map(
        (entry) => `
          <tr>
            <td>${entry.label}</td>
            <td>${pct(entry.metrics.accuracy)}</td>
            <td>${pct(entry.metrics.precision)}</td>
            <td>${pct(entry.metrics.recall)}</td>
            <td>${pct(entry.metrics.f1)}</td>
          </tr>
        `
      )
      .join("");

    wrapper.innerHTML = `
      <h3>${group.title}</h3>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
    return wrapper;
  });

  fillNode(legacyResults, nodes);
}

function renderJobs(jobs) {
  const nodes = jobs.map((job) => {
    const wrapper = document.createElement("div");
    wrapper.className = "job-box";
    wrapper.innerHTML = `
      <div class="job-head">
        <strong>${job.kind}</strong>
        <span class="job-badge ${job.status}">${job.status}</span>
      </div>
      <div class="inline-note">Created: ${formatWhen(job.created_at)}</div>
      <div class="inline-note">Started: ${formatWhen(job.started_at)}</div>
      <div class="inline-note">Completed: ${formatWhen(job.completed_at)}</div>
      ${job.run_id ? `<div class="inline-note">Run ID: ${job.run_id}</div>` : ""}
      ${job.error ? `<div class="inline-note">Error: ${job.error}</div>` : ""}
    `;
    return wrapper;
  });
  fillNode(jobList, nodes);
}

function renderRuns(runs) {
  const nodes = runs.map((run) => {
    const wrapper = document.createElement("div");
    wrapper.className = "run-box";
    const best = run.results?.[0];
    wrapper.innerHTML = `
      <div class="run-head">
        <strong>${run.kind}</strong>
        <span class="job-badge completed">${run.run_id}</span>
      </div>
      <h3>${best ? best.label : "No result entries"}</h3>
      <div class="inline-note">Created: ${formatWhen(run.created_at)}</div>
      <div class="inline-note">Train rows: ${number(run.dataset?.train_rows || 0)} | Test rows: ${number(run.dataset?.test_rows || 0)}</div>
      ${
        best
          ? `
            <div class="score-strip">
              <div><span>Accuracy</span><strong>${pct(best.metrics.accuracy)}</strong></div>
              <div><span>Precision</span><strong>${pct(best.metrics.precision)}</strong></div>
              <div><span>Recall</span><strong>${pct(best.metrics.recall)}</strong></div>
              <div><span>F1</span><strong>${pct(best.metrics.f1)}</strong></div>
            </div>
          `
          : ""
      }
    `;
    return wrapper;
  });
  fillNode(runList, nodes);
}

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json();
}

async function loadOverview() {
  try {
    const [health, overview] = await Promise.all([request("/api/health"), request("/api/overview")]);
    setHealth(health.status === "ok");
    renderDatasets(overview.datasets);
    renderLegacy(overview.legacy);
    renderJobs(overview.jobs);
    renderRuns(overview.runs);
  } catch (error) {
    setHealth(false);
    console.error(error);
  }
}

function formToPayload(form) {
  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    if (value === "") {
      continue;
    }
    if (["train_sample", "test_sample", "epochs", "batch_size"].includes(key)) {
      payload[key] = Number(value);
    } else {
      payload[key] = value;
    }
  }
  return payload;
}

async function launch(url, payload = {}) {
  await request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await loadOverview();
}

document.getElementById("refresh-overview").addEventListener("click", loadOverview);
document.getElementById("run-legacy").addEventListener("click", () => launch("/api/jobs/legacy-evaluation"));
document.getElementById("classical-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await launch("/api/jobs/classical", formToPayload(event.currentTarget));
});
document.getElementById("dnn-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await launch("/api/jobs/dnn", formToPayload(event.currentTarget));
});

loadOverview();
setInterval(loadOverview, 6000);

