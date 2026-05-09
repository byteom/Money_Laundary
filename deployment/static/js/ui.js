/* ═══════════════════════════════════════════════════════════════════
   UI Rendering — maps backend API fields to DOM elements correctly

   Backend response fields:
   - baseline_probability   → Random Forest (RF)
   - logistic_probability   → Logistic Regression (LR)  [optional]
   - graphsage_probability  → GraphSAGE GNN
   - tgn_probability        → Temporal Graph Network
   - ensemble_probability   → Weighted Ensemble
   - ml_ensemble_probability→ ML-only (before rule boost)
   - risk_classification    → "critical" | "high" | "medium" | "low" | "minimal"
   - explainability         → {summary, confidence, top_factors, decision_steps, model_contributions}
   - inference_warnings     → [string]
   - aml_rules_triggered    → [{rule_id, rule_name, severity, detail}]
   - context_summary        → {context_applied, recent_transactions, window_minutes}
   ═══════════════════════════════════════════════════════════════════ */

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return 'N/A';
  }
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function clampPercent(value) {
  return `${Math.max(0, Math.min(100, Number(value) * 100))}%`;
}

function riskClassName(risk) {
  return String(risk || 'minimal').toLowerCase();
}

function setHidden(id, hidden) {
  const el = document.getElementById(id);
  if (el) el.hidden = hidden;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setHtml(id, value) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = value;
}

function formatMoney(value) {
  return `$${Number(value).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 2 })}`;
}

/**
 * Core rendering function shared by Single and Context tabs.
 * Maps real backend API fields to the correct DOM elements.
 */
function renderResultBlock(prefix, result, modelInfo) {
  const risk = riskClassName(result.risk_classification);
  const explained = result.explainability || {};
  const summary = explained.summary || {};

  setHidden(`${prefix}Loading`, true);
  setHidden(`${prefix}Empty`, true);
  setHidden(`${prefix}Results`, false);

  // ── Main probability display ─────────────────────────────────
  const ensembleProb = result.ensemble_probability;
  setText(`${prefix}RiskScore`, pct(ensembleProb));

  // Apply color class based on risk
  const scoreEl = document.getElementById(`${prefix}RiskScore`);
  if (scoreEl) scoreEl.className = `big-prob ${risk}`;

  // Serving model info
  const modelsAvail = modelInfo?.models_available || {};
  const servingText =
    `Serving: ${modelsAvail.baseline || 'Random Forest'} + ${modelsAvail.graphsage || 'GraphSAGE'} + ${modelsAvail.tgn || 'TGN'} ` +
    `| Context: ${result.context_applied ? 'Yes' : 'No'} ` +
    `| AML Rules: ${result.aml_rules_count || 0} triggered`;
  setText(`${prefix}ServingLine`, servingText);

  // Risk pill
  const pill = document.getElementById(`${prefix}RiskPill`);
  if (pill) {
    pill.textContent = risk.toUpperCase();
    pill.className = `risk-pill ${risk}`;
  }

  // ── Model probability bars ───────────────────────────────────
  // Maps: [DOM-key, API-field]
  const barMap = [
    ['Logistic', result.logistic_probability],
    ['RandomForest', result.baseline_probability],
    ['GraphSage', result.graphsage_probability],
    ['Tgn', result.tgn_probability],
    ['Ensemble', result.ensemble_probability],
  ];

  for (const [key, value] of barMap) {
    const valueNode = document.getElementById(`${prefix}${key}Value`);
    const fillNode = document.getElementById(`${prefix}${key}Fill`);
    const rowNode = document.getElementById(`${prefix}${key}Row`);

    const hasValue = value !== null && value !== undefined && !Number.isNaN(Number(value));

    // For LR and RF, hide the row if no data
    if (rowNode) {
      rowNode.hidden = !hasValue;
    }

    if (hasValue && valueNode && fillNode) {
      valueNode.textContent = pct(value);
      fillNode.style.width = clampPercent(value);
    }
  }

  // ── Plain-language summary ───────────────────────────────────
  setText(`${prefix}Headline`, summary.headline || 'Risk explanation unavailable.');
  setText(`${prefix}Drivers`, summary.drivers || 'No major drivers provided.');
  setText(`${prefix}ConfidenceNote`, summary.confidence_note || 'Confidence details unavailable.');
  setText(`${prefix}Action`, summary.recommended_action || 'No action guidance available.');

  // ── Confidence line ──────────────────────────────────────────
  const confidence = explained.confidence || {};
  setText(`${prefix}ConfidenceLine`, `Confidence: ${confidence.label || 'n/a'} (${confidence.score ?? 'n/a'})`);

  // ── Top factors ──────────────────────────────────────────────
  const factorList = document.getElementById(`${prefix}TopFactors`);
  if (factorList) {
    factorList.innerHTML = '';
    const factors = explained.top_factors || [];
    for (const factor of factors) {
      const item = document.createElement('li');
      item.textContent = `${factor.label}: impact ${factor.impact_score}, value ${factor.value}`;
      factorList.appendChild(item);
    }
  }

  // ── Decision steps ───────────────────────────────────────────
  const stepList = document.getElementById(`${prefix}DecisionSteps`);
  if (stepList) {
    stepList.innerHTML = '';
    const steps = explained.decision_steps || [];
    for (const step of steps) {
      const item = document.createElement('li');
      item.textContent = `${step.title} — ${step.detail}`;
      stepList.appendChild(item);
    }
  }

  // ── Warnings ─────────────────────────────────────────────────
  const warnings = Array.isArray(result.inference_warnings) ? result.inference_warnings : [];
  const warningBox = document.getElementById(`${prefix}Warnings`);
  if (warningBox) {
    if (warnings.length > 0) {
      warningBox.hidden = false;
      warningBox.textContent = `Warnings: ${warnings.join(' | ')}`;
    } else {
      warningBox.hidden = true;
    }
  }

  // ── Context history summary (context tab only) ───────────────
  const contextSummaryBox = document.getElementById(`${prefix}HistorySummary`);
  if (contextSummaryBox) {
    const ctxSummary = result.context_summary || {};
    contextSummaryBox.textContent = ctxSummary.context_applied
      ? `Context applied with ${ctxSummary.recent_transactions || 0} recent transactions across a ${ctxSummary.window_minutes || 0} minute window.`
      : 'No temporal context supplied.';
  }
}

export function setLoading(prefix, active) {
  setHidden(`${prefix}Loading`, !active);
  if (active) {
    setHidden(`${prefix}Empty`, true);
    setHidden(`${prefix}Results`, true);
  }
}

export function setEmpty(prefix) {
  setHidden(`${prefix}Loading`, true);
  setHidden(`${prefix}Results`, true);
  setHidden(`${prefix}Empty`, false);
}

export function setError(prefix, message) {
  setEmpty(prefix);
  const emptyBox = document.getElementById(`${prefix}Empty`);
  if (emptyBox) {
    emptyBox.innerHTML = `<i class="fas fa-exclamation-triangle" style="font-size:2rem;margin-bottom:.75rem;opacity:.4;color:var(--red)"></i><p>${message}</p>`;
  }
}

export function renderModelInfo(info) {
  const riskLevels = info?.risk_levels || {};
  const models = info?.models_available || {};
  const weights = info?.ensemble_weights || {};

  const modelParts = [];
  if (models.logistic_regression && models.logistic_regression !== 'Not loaded') modelParts.push('LR');
  if (models.baseline) modelParts.push('RF');
  if (models.graphsage) modelParts.push('GraphSAGE');
  if (models.tgn && models.tgn !== 'Not trained') modelParts.push('TGN');

  const weightParts = Object.entries(weights).map(([k, v]) => `${k}: ${(Number(v) * 100).toFixed(0)}%`).join(', ');
  const riskText = Object.entries(riskLevels).map(([k, v]) => `${k}: ${v}`).join(' | ');

  const statusEl = document.getElementById('modelStatus');
  if (statusEl) {
    statusEl.textContent = `Models: ${modelParts.join(' + ')} | Weights: ${weightParts} | ${riskText}`;
  }
}

export function renderSingleResult(result, modelInfo) {
  renderResultBlock('single', result, modelInfo);
}

export function renderContextResult(result, modelInfo) {
  renderResultBlock('context', result, modelInfo);
}

export function renderBatchResults(payload) {
  setHidden('batchLoading', true);
  setHidden('batchEmpty', true);

  const results = Array.isArray(payload.results) ? payload.results : [];
  const summary = payload.summary || {};

  const stats = [
    `${summary.cases_requested || results.length} cases`,
    `${summary.cases_completed || 0} completed`,
    `${summary.risk_counts ? (summary.risk_counts.high || 0) + (summary.risk_counts.critical || 0) : 0} elevated`,
    `${summary.unique_probability_count || 0} unique scores`,
  ];
  setHtml('batchStats', stats.map((label) => `<span class="stat-chip">${label}</span>`).join(''));

  const summaryLines = [
    `Batch completed with ${summary.cases_completed || 0} scored cases and ${summary.cases_failed || 0} validation failures.`,
    `Probability span: ${summary.min_probability ?? 0} to ${summary.max_probability ?? 0} with ${summary.unique_probability_count || 0} unique ensemble scores.`,
  ];
  setText('batchSummary', summaryLines.join(' '));

  const tableBody = document.getElementById('batchTableBody');
  if (!tableBody) return;

  tableBody.innerHTML = '';
  results.forEach((row, index) => {
    const tr = document.createElement('tr');
    if (row.status === 'error') {
      tr.innerHTML = `
        <td>${index + 1}</td>
        <td>${row.scenario || row.case_id || 'Invalid case'}</td>
        <td colspan="7" style="color:#ffd4dc">${row.error || 'Validation failed'}</td>
        <td><span class="risk-pill low inline">ERROR</span></td>
      `;
      tableBody.appendChild(tr);
      return;
    }

    const risk = riskClassName(row.risk_classification);
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td style="font-size:.75rem">${row.scenario || row.case_id || 'Case'}</td>
      <td>${formatMoney(row.explainability?.inputs?.transaction_amount ?? 0)}</td>
      <td>${row.explainability?.inputs?.transaction_type || 'n/a'}</td>
      <td>${pct(row.logistic_probability)}</td>
      <td>${pct(row.baseline_probability)}</td>
      <td>${pct(row.graphsage_probability)}</td>
      <td>${pct(row.tgn_probability)}</td>
      <td><strong>${pct(row.ensemble_probability)}</strong></td>
      <td><span class="risk-pill ${risk} inline">${risk.toUpperCase()}</span></td>
    `;
    tableBody.appendChild(tr);
  });
}

export function populateForm(data) {
  document.getElementById('sender_id').value = data.sender_id || '';
  document.getElementById('receiver_id').value = data.receiver_id || '';
  document.getElementById('transaction_amount').value = data.transaction_amount || '';
  document.getElementById('timestamp').value = data.timestamp || '';
  document.getElementById('transaction_type').value = data.transaction_type || 'transfer';
}

export function renderCurrentTransactionSummary(payload) {
  setText(
    'singlePreview',
    `${payload.sender_id} → ${payload.receiver_id} | ${formatMoney(payload.transaction_amount)} | ${payload.transaction_type} | ${payload.timestamp}`,
  );
  setText(
    'contextCurrentSummary',
    `Current transaction: ${payload.sender_id} → ${payload.receiver_id} | ${formatMoney(payload.transaction_amount)} | ${payload.transaction_type} | ${payload.timestamp}`,
  );
}

export function toIsoWithZ(datetimeLocal) {
  if (!datetimeLocal) return '';
  return `${datetimeLocal}:00Z`;
}
