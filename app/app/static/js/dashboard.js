let confusionChart, shapChart, rocChart, prChart, compareChart, perturbChart;
let allModels = [];

async function loadModels() {
  const res = await fetch("/api/models");
  const data = await res.json();

  allModels = data.models;

  const select = document.getElementById("modelSelect");
  select.innerHTML = "";

  data.models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.model_id;
    opt.textContent = `${m.model_name} (${m.version})`;
    select.appendChild(opt);
  });

  if (data.models.length > 0) {
    await loadEvaluation();
    renderComparison();
  }
}

async function loadEvaluation() {
  const modelId = document.getElementById("modelSelect").value;
  const res = await fetch(`/api/models/${modelId}/evaluation`);
  const model = await res.json();

  renderMetrics(model.metrics);
  renderConfusionMatrix(model.confusion_matrix);
  renderShap(model.shap_summary);
  renderPerturb(model.perturbation_importance);
  renderRoc(model.roc_curve);
  renderPr(model.pr_curve);
}

function renderMetrics(metrics) {
  const grid = document.getElementById("metricsGrid");
  grid.innerHTML = "";

  Object.entries(metrics).forEach(([key, value]) => {
    const box = document.createElement("div");
    box.className = "metric-box";
    box.innerHTML = `<h3>${key}</h3><p>${value}</p>`;
    grid.appendChild(box);
  });
}

function renderConfusionMatrix(cm) {
  const container = document.getElementById("confusionMatrix");
  const [labelA, labelB] = cm.labels;
  const [[tn, fp], [fn, tp]] = cm.matrix;

  const values = [tn, fp, fn, tp];
  const max = Math.max(...values);

  const shade = (v) => {
    const intensity = Math.round((v / max) * 180) + 40; // 40–220
    return `background-color: rgb(230, 240, 255); color: #202122; box-shadow: inset 0 0 0 9999px rgba(0, 0, 128, ${v / max * 0.25})`;
  };

  container.innerHTML = `
    <table class="confusion-matrix">
      <tr>
        <th></th>
        <th colspan="2">Predicted</th>
      </tr>
      <tr>
        <th></th>
        <th>${labelA}</th>
        <th>${labelB}</th>
      </tr>
      <tr>
        <th class="label">${labelA}</th>
        <td style="${shade(tn)}">${tn}</td>
        <td style="${shade(fp)}">${fp}</td>
      </tr>
      <tr>
        <th class="label">${labelB}</th>
        <td style="${shade(fn)}">${fn}</td>
        <td style="${shade(tp)}">${tp}</td>
      </tr>
    </table>
  `;
}

function renderShap(shap) {
  const ctx = document.getElementById("shapChart").getContext("2d");
  if (!shap || !shap.top_features || shap.top_features.length === 0) {
    if (shapChart) shapChart.destroy();
    return;
  }
  const labels = shap.top_features.map(f => f.feature);
  const values = shap.top_features.map(f => f.importance);
  if (shapChart) shapChart.destroy();
  shapChart = new Chart(ctx, { type: "bar", data: { labels, datasets: [{ label: "SHAP", data: values }] } });
}

function renderPerturb(items) {
  const ctx = document.getElementById("perturbChart").getContext("2d");
  if (!items || items.length === 0) {
    if (perturbChart) perturbChart.destroy();
    return;
  }
  const labels = items.map(f => f.feature);
  const values = items.map(f => f.delta);
  if (perturbChart) perturbChart.destroy();
  perturbChart = new Chart(ctx, { type: "bar", data: { labels, datasets: [{ label: "Delta", data: values }] } });
}

function renderRoc(roc) {
  const container = document.getElementById("rocContainer");
  container.innerHTML = "";

  if (roc && roc.fpr && roc.fpr.length > 0 && roc.tpr && roc.tpr.length > 0) {
    const canvas = document.createElement("canvas");
    canvas.id = "rocChart";
    container.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    if (rocChart) rocChart.destroy();
    rocChart = new Chart(ctx, {
      type: "line",
      data: { labels: roc.fpr, datasets: [{ label: "TPR", data: roc.tpr, borderColor: "blue" }] }
    });
    return;
  }

  if (roc && roc.image_url) {
    const img = document.createElement("img");
    img.src = roc.image_url;
    img.alt = "ROC Curve";
    img.style.maxWidth = "100%";
    img.style.border = "1px solid #a2a9b1";
    img.style.background = "#fff";
    container.appendChild(img);
  } else {
    container.innerHTML = "<p>No ROC curve data available.</p>";
  }
}

function renderPr(pr) {
  const container = document.getElementById("prContainer");
  container.innerHTML = "";

  if (pr && pr.recall && pr.recall.length > 0 && pr.precision && pr.precision.length > 0) {
    const canvas = document.createElement("canvas");
    canvas.id = "prChart";
    container.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    if (prChart) prChart.destroy();
    prChart = new Chart(ctx, {
      type: "line",
      data: { labels: pr.recall, datasets: [{ label: "Precision", data: pr.precision, borderColor: "green" }] }
    });
    return;
  }

  if (pr && pr.image_url) {
    const img = document.createElement("img");
    img.src = pr.image_url;
    img.alt = "PR Curve";
    img.style.maxWidth = "100%";
    img.style.border = "1px solid #a2a9b1";
    img.style.background = "#fff";
    container.appendChild(img);
  } else {
    container.innerHTML = "<p>No PR curve data available.</p>";
  }
}

function renderComparison() {
  const ctx = document.getElementById("compareChart").getContext("2d");
  const labels = allModels.map(m => `${m.model_name} (${m.version})`);
  const accuracy = allModels.map(m => m.metrics.accuracy);
  const f1 = allModels.map(m => m.metrics.f1);
  const rocAuc = allModels.map(m => m.metrics.roc_auc);

  if (compareChart) compareChart.destroy();
  compareChart = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [
      { label: "Accuracy", data: accuracy },
      { label: "F1", data: f1 },
      { label: "ROC-AUC", data: rocAuc }
    ]}
  });
}

loadModels();