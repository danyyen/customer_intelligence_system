const money = new Intl.NumberFormat("en-GB", {
  style: "currency",
  currency: "GBP",
  maximumFractionDigits: 0,
});
const percent = new Intl.NumberFormat("en-GB", {
  style: "percent",
  maximumFractionDigits: 1,
});

const byId = (id) => document.getElementById(id);

async function getJSON(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(body.detail || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function setText(id, value) {
  byId(id).textContent = value;
}

function priorityClass(label) {
  const normalized = label.toLowerCase();
  if (normalized.includes("priority 1")) return "priority-high";
  if (normalized.includes("priority 2")) return "priority-medium";
  return "priority-standard";
}

function showDecision(row) {
  setText("result-id", row.customer_id);
  setText("churn-probability", percent.format(row.churn_probability));
  setText(
    "threshold-context",
    row.above_model_threshold
      ? `Above the ${percent.format(row.probability_threshold)} action threshold`
      : `Below the ${percent.format(row.probability_threshold)} action threshold`,
  );
  setText("segment", row.segment);
  setText("frequency", `${row.segment_frequency.toLocaleString()} historical purchases`);
  setText("customer-value", money.format(row.customer_value));
  setText("value-at-risk", money.format(row.value_at_risk_score));
  setText("value-rank", `Value-protection rank #${row.value_at_risk_rank.toLocaleString()}`);
  setText("priority-band", row.active_priority_band);
  setText("recommended-action", row.active_recommended_action);
  setText(
    "decision-meta",
    `Snapshot ${row.snapshot_date} · Model ${row.model_version} · Strategy: ${row.active_strategy.replace("_", " ")}`,
  );
  byId("priority-band").className = `priority-pill ${priorityClass(row.active_priority_band)}`;
  byId("customer-result").hidden = false;
  byId("customer-result").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function lookupCustomer(event) {
  event.preventDefault();
  const customerId = byId("customer-id").value.trim();
  const strategy = byId("strategy").value;
  setText("lookup-message", "Loading customer decision…");
  try {
    const row = await getJSON(
      `/v1/customers/${encodeURIComponent(customerId)}?strategy=${encodeURIComponent(strategy)}`,
    );
    showDecision(row);
    setText("lookup-message", "");
  } catch (error) {
    byId("customer-result").hidden = true;
    setText(
      "lookup-message",
      error.status === 404 ? "Customer ID not found in the latest snapshot." : error.message,
    );
  }
}

async function loadQueue() {
  const strategy = byId("queue-strategy").value;
  const body = byId("queue-body");
  body.replaceChildren();
  setText("queue-message", "Loading campaign queue…");
  try {
    const rows = await getJSON(`/v1/customers?strategy=${strategy}&limit=10`);
    for (const row of rows) {
      const tr = document.createElement("tr");
      const values = [
        row.customer_id,
        row.segment,
        percent.format(row.churn_probability),
        money.format(row.customer_value),
        row.active_priority_band,
      ];
      for (const value of values) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      }
      tr.tabIndex = 0;
      tr.title = "Open this customer";
      tr.addEventListener("click", () => {
        byId("customer-id").value = row.customer_id;
        byId("strategy").value = strategy;
        byId("lookup-form").requestSubmit();
      });
      tr.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") tr.click();
      });
      body.appendChild(tr);
    }
    setText("queue-message", `${rows.length} customers shown from the latest scored snapshot.`);
  } catch (error) {
    setText("queue-message", error.message);
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function setServiceStatus(text, state) {
  const el = byId("service-status");
  el.textContent = text;
  el.classList.remove("status-ready", "status-error", "status-waking");
  if (state) el.classList.add(`status-${state}`);
}

// The free Render instance sleeps after ~15 minutes idle and can take up to a
// minute to answer the first request. Poll readiness with a fixed backoff
// instead of failing on the first cold-start error.
async function waitForReady(attempts = 8, delayMs = 5000) {
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await getJSON("/health/ready");
    } catch (error) {
      if (attempt === attempts) throw error;
      setServiceStatus(
        "Service is waking up — the free host can take up to a minute on the first visit…",
        "waking",
      );
      await sleep(delayMs);
    }
  }
}

async function initialize() {
  setServiceStatus("Connecting to the service…", "waking");
  let health;
  try {
    health = await waitForReady();
  } catch (error) {
    setServiceStatus(
      "Service is temporarily unavailable. Refresh the page to try again.",
      "error",
    );
    return;
  }
  setServiceStatus(`Service ready · ${health.model_version}`, "ready");
  try {
    const model = await getJSON("/v1/model");
    setText(
      "model-summary",
      `Model ${model.model_version} · ${model.observation_window_days}-day history · ${model.prediction_horizon_days}-day prediction horizon`,
    );
  } catch (error) {
    setText("model-summary", "Model information is unavailable.");
  }
  await loadQueue();
}

byId("lookup-form").addEventListener("submit", lookupCustomer);
byId("queue-strategy").addEventListener("change", loadQueue);
initialize();
