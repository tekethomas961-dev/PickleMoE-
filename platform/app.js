const data = window.MOE_PLATFORM_DATA;

const formatMetric = (value, digits = 3) => {
  if (value === null || value === undefined || value === "") return "—";
  if (Number.isInteger(value)) return String(value);
  return Number(value).toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
};

const featureGrid = document.querySelector("#featureGrid");
data.features.forEach(([name, desc]) => {
  const item = document.createElement("div");
  item.className = "feature-item";
  item.innerHTML = `<code>${name}</code><span>${desc}</span>`;
  featureGrid.appendChild(item);
});

const topMetrics = document.querySelector("#topMetrics");
[
  ["最终模型数", "5", "含 MLP 与 MoE 消融"],
  ["输入特征", "8", "全部为连续技术特征"],
  ["输出类别", "4", "四类击球标签"],
  ["专家数量", "4", "每类技术模式有解释空间"],
  ["最佳 CV(load)", "0.163", "K=2 + 负载平衡"],
].forEach(([label, value, note]) => {
  const item = document.createElement("div");
  item.className = "metric-card";
  item.innerHTML = `<span>${label}</span><strong>${value}</strong><small>${note}</small>`;
  topMetrics.appendChild(item);
});

const resultTbody = document.querySelector("#resultsTable tbody");
data.runs.forEach((run) => {
  const tr = document.createElement("tr");
  const badge =
    run.k === undefined ? '<span class="model-badge">Baseline</span>' : `<span class="model-badge">K=${run.k}</span>`;
  tr.innerHTML = `
    <td><strong>${run.name}</strong>${badge}</td>
    <td>${run.bestEpoch}</td>
    <td>${formatMetric(run.accuracy, 3)}</td>
    <td>${formatMetric(run.macroF1, 3)}</td>
    <td>${formatMetric(run.balancedAccuracy, 3)}</td>
    <td>${run.activeParams.toLocaleString("zh-CN")}</td>
    <td>${formatMetric(run.cvLoad, 3)}</td>
    <td>${formatMetric(run.maxMean, 3)}</td>
  `;
  resultTbody.appendChild(tr);
});

function renderRouting(key) {
  const routing = data.routing[key];
  const image = document.querySelector("#routingImage");
  const link = document.querySelector("#routingImageLink");
  document.querySelector("#routingTitle").textContent = routing.title;
  image.src = routing.image;
  link.href = routing.image;

  const target = document.querySelector("#dominantExperts");
  target.innerHTML = "";
  routing.matrix.forEach((row, rowIndex) => {
    const max = Math.max(...row);
    const expertIndex = row.indexOf(max);
    const item = document.createElement("div");
    item.className = "dominant-item";
    item.innerHTML = `
      <div>
        <strong>${data.labels[rowIndex]}</strong>
        <span>Expert ${expertIndex} 主导，selected gate 权重 ${formatMetric(max, 3)}</span>
      </div>
      <div class="mini-bars">
        ${row
          .map(
            (value, i) =>
              `<span title="Expert ${i}: ${formatMetric(value, 3)}"><i style="height:${Math.max(
                4,
                value * 72,
              )}px"></i></span>`,
          )
          .join("")}
      </div>
    `;
    target.appendChild(item);
  });
}

document.querySelectorAll("[data-routing]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-routing]").forEach((el) => el.classList.remove("active"));
    button.classList.add("active");
    renderRouting(button.dataset.routing);
  });
});
renderRouting("k1");

const balanceCards = document.querySelector("#balanceCards");
[
  ["K=1", data.runs.find((r) => r.id === "final_moe_k1_lb"), data.runs.find((r) => r.id === "final_moe_k1_no_lb")],
  ["K=2", data.runs.find((r) => r.id === "final_moe_k2_lb"), data.runs.find((r) => r.id === "final_moe_k2_no_lb")],
].forEach(([label, withLb, noLb]) => {
  const reduction = ((1 - withLb.cvLoad / noLb.cvLoad) * 100).toFixed(1);
  const item = document.createElement("article");
  item.className = "balance-card";
  item.innerHTML = `
    <h3>${label} 负载平衡消融</h3>
    <div class="balance-row">
      <span>无负载平衡</span>
      <strong>${formatMetric(noLb.cvLoad, 3)}</strong>
    </div>
    <div class="bar-track"><i style="width:${Math.min(noLb.cvLoad * 100, 100)}%"></i></div>
    <div class="balance-row">
      <span>lambda_lb=0.01</span>
      <strong>${formatMetric(withLb.cvLoad, 3)}</strong>
    </div>
    <div class="bar-track improved"><i style="width:${Math.min(withLb.cvLoad * 100, 100)}%"></i></div>
    <p>CV(load) 降低 ${reduction}%，专家使用更均衡。</p>
  `;
  balanceCards.appendChild(item);
});

const figureTabs = document.querySelector("#figureTabs");
const figureGrid = document.querySelector("#figureGrid");
function renderFigures(groupName) {
  figureGrid.innerHTML = "";
  data.figures[groupName].forEach(([title, src]) => {
    const item = document.createElement("article");
    item.className = "figure-card";
    item.innerHTML = `
      <div class="panel-title">
        <h3>${title}</h3>
        <a href="${src}" target="_blank" rel="noreferrer">打开</a>
      </div>
      <div class="image-frame">
        <img src="${src}" alt="${groupName} ${title}" />
      </div>
    `;
    figureGrid.appendChild(item);
  });
}

Object.keys(data.figures).forEach((groupName, index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `figure-tab${index === 0 ? " active" : ""}`;
  button.textContent = groupName;
  button.addEventListener("click", () => {
    document.querySelectorAll(".figure-tab").forEach((el) => el.classList.remove("active"));
    button.classList.add("active");
    renderFigures(groupName);
  });
  figureTabs.appendChild(button);
});
renderFigures(Object.keys(data.figures)[0]);

const publicTest = data.publicTest;
if (publicTest) {
  document.querySelector("#publicSourceName").textContent = publicTest.sourceName;
  document.querySelector("#publicMethod").textContent = `${publicTest.method} · ${publicTest.datasetShape[0].toLocaleString(
    "zh-CN",
  )} samples · ${publicTest.splitMethod || "train/val/test"} = ${publicTest.split.train}/${publicTest.split.val}/${publicTest.split.test}`;
  const publicSourceLink = document.querySelector("#publicSourceLink");
  publicSourceLink.href = publicTest.sourceUrl;

  const publicResults = document.querySelector("#publicResults");
  publicTest.results.forEach((row) => {
    const card = document.createElement("article");
    card.className = "public-result-card";
    card.innerHTML = `
      <strong>${row.model}</strong>
      <div class="public-metrics">
        <span><b>${formatMetric(row.accuracy, 3)}</b><i>Accuracy</i></span>
        <span><b>${formatMetric(row.macroF1, 3)}</b><i>Macro F1</i></span>
        <span><b>${formatMetric(row.balancedAccuracy, 3)}</b><i>Balanced Acc</i></span>
      </div>
    `;
    publicResults.appendChild(card);
  });

  const publicFigures = document.querySelector("#publicFigures");
  publicTest.figures.forEach(([title, src]) => {
    const card = document.createElement("article");
    card.className = "figure-card";
    card.innerHTML = `
      <div class="panel-title">
        <h3>${title}</h3>
        <a href="${src}" target="_blank" rel="noreferrer">打开</a>
      </div>
      <div class="image-frame">
        <img src="${src}" alt="${title}" />
      </div>
    `;
    publicFigures.appendChild(card);
  });
}
