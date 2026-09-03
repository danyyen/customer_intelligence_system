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

async function initialize() {
  try {
    const [health, model] = await Promise.all([
      getJSON("/health/ready"),
      getJSON("/v1/model"),
    ]);
    setText("service-status", `Service ready · ${health.model_version}`);
    byId("service-status").classList.add("status-ready");
    setText(
      "model-summary",
      `Model ${model.model_version} · ${model.observation_window_days}-day history · ${model.prediction_horizon_days}-day prediction horizon`,
    );
  } catch (error) {
    setText("service-status", "Service is temporarily unavailable");
    byId("service-status").classList.add("status-error");
  }
  await loadQueue();
}

byId("lookup-form").addEventListener("submit", lookupCustomer);
byId("queue-strategy").addEventListener("change", loadQueue);
initialize();
