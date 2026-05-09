import { predictBatch, predictTransaction, predictWithContext, getModelInfo } from './api.js';
import { state, setActiveTab as setActiveTabState, setBatchResult, setContextResult, setLatestResult, setModelInfo } from './state.js';
import {
  setLoading,
  setEmpty,
  setError,
  renderModelInfo,
  renderBatchResults,
  renderContextResult,
  renderCurrentTransactionSummary,
  renderSingleResult,
  populateForm,
  toIsoWithZ,
} from './ui.js';

/* ═══════════════════════════════════════════════════════════════════
   Scenarios — these match the backend's expected fields:
   sender_id, receiver_id, transaction_amount, timestamp, transaction_type
   ═══════════════════════════════════════════════════════════════════ */

const scenarios = {
  normal: {
    name: 'Normal Grocery',
    sender_id: 'ACC_0123',
    receiver_id: 'ACC_0456',
    transaction_amount: 145.5,
    timestamp: new Date().toISOString().slice(0, 16),
    transaction_type: 'payment',
  },
  utility: {
    name: 'Normal Utility Bill',
    sender_id: 'ACC_0900',
    receiver_id: 'ACC_1100',
    transaction_amount: 82.3,
    timestamp: new Date(Date.now() - 1800000).toISOString().slice(0, 16),
    transaction_type: 'payment',
  },
  smallTransfer: {
    name: 'Small Transfer',
    sender_id: 'ACC_0345',
    receiver_id: 'ACC_0789',
    transaction_amount: 320,
    timestamp: new Date(Date.now() - 7200000).toISOString().slice(0, 16),
    transaction_type: 'transfer',
  },
  dinner: {
    name: 'Restaurant Dinner',
    sender_id: 'ACC_1122',
    receiver_id: 'ACC_9911',
    transaction_amount: 65.5,
    timestamp: new Date(Date.now() - 5400000).toISOString().slice(0, 16),
    transaction_type: 'payment',
  },
  online: {
    name: 'Online Shopping',
    sender_id: 'ACC_1401',
    receiver_id: 'ACC_1520',
    transaction_amount: 310,
    timestamp: new Date(Date.now() - 3600000).toISOString().slice(0, 16),
    transaction_type: 'payment',
  },
  wire: {
    name: 'Large Wire Transfer',
    sender_id: 'ACC_9001',
    receiver_id: 'ACC_9002',
    transaction_amount: 25000,
    timestamp: new Date(Date.now() - 900000).toISOString().slice(0, 16),
    transaction_type: 'transfer',
  },
  atm: {
    name: 'ATM Withdrawal',
    sender_id: 'ACC_7001',
    receiver_id: 'ATM_001',
    transaction_amount: 9500,
    timestamp: new Date(Date.now() - 600000).toISOString().slice(0, 16),
    transaction_type: 'withdrawal',
  },
  travel: {
    name: 'Travel Booking',
    sender_id: 'ACC_3001',
    receiver_id: 'ACC_3002',
    transaction_amount: 1800,
    timestamp: new Date(Date.now() - 7200000).toISOString().slice(0, 16),
    transaction_type: 'payment',
  },
  salary: {
    name: 'Salary Deposit',
    sender_id: 'EMPLOYER_01',
    receiver_id: 'ACC_4455',
    transaction_amount: 5600,
    timestamp: new Date(Date.now() - 86400000).toISOString().slice(0, 16),
    transaction_type: 'deposit',
  },
  suspicious: {
    name: 'Suspicious – Large Cash Out',
    sender_id: 'ACC_0111',
    receiver_id: 'ACC_0222',
    transaction_amount: 9850,
    timestamp: new Date(Date.now() - 900000).toISOString().slice(0, 16),
    transaction_type: 'cash_out',
  },
  rapid: {
    name: 'Suspicious – Rapid Mobile',
    sender_id: 'ACC_9009',
    receiver_id: 'ACC_9010',
    transaction_amount: 7200,
    timestamp: new Date(Date.now() - 120000).toISOString().slice(0, 16),
    transaction_type: 'transfer',
  },
  unknownPair: {
    name: 'Unknown Account Pair',
    sender_id: 'UNK_SENDER_9001',
    receiver_id: 'UNK_RECEIVER_422',
    transaction_amount: 50000,
    timestamp: new Date().toISOString().slice(0, 16),
    transaction_type: 'transfer',
  },
};

/* ═══════════════════════════════════════════════════════════════════
   Batch cases — each wraps its payload in a 'payload' key
   matching the batch-predict endpoint format
   ═══════════════════════════════════════════════════════════════════ */

const batchCases = [
  { case_id: 'batch_01', scenario: 'Grocery $45', payload: { sender_id: 'ACC_CTX_001', receiver_id: 'ACC_NORM_102', transaction_amount: 45.2, timestamp: '2026-05-09T06:15:21Z', transaction_type: 'payment' } },
  { case_id: 'batch_02', scenario: 'Utility $120', payload: { sender_id: 'ACC_CTX_001', receiver_id: 'ACC_NORM_103', transaction_amount: 120, timestamp: '2026-05-09T07:15:21Z', transaction_type: 'payment' } },
  { case_id: 'batch_03', scenario: 'Online $310', payload: { sender_id: 'ACC_CTX_002', receiver_id: 'ACC_NORM_104', transaction_amount: 310, timestamp: '2026-05-09T08:15:21Z', transaction_type: 'payment' } },
  { case_id: 'batch_04', scenario: 'Transfer $250', payload: { sender_id: 'ACC_CTX_003', receiver_id: 'ACC_NORM_105', transaction_amount: 250, timestamp: '2026-05-09T09:15:21Z', transaction_type: 'transfer' } },
  { case_id: 'batch_05', scenario: 'Withdrawal $90', payload: { sender_id: 'ACC_CTX_004', receiver_id: 'ATM_001', transaction_amount: 90, timestamp: '2026-05-09T10:15:21Z', transaction_type: 'withdrawal' } },
  { case_id: 'batch_06', scenario: 'Dinner $65', payload: { sender_id: 'ACC_CTX_005', receiver_id: 'ACC_NORM_106', transaction_amount: 65.5, timestamp: '2026-05-09T11:15:21Z', transaction_type: 'payment' } },
  { case_id: 'batch_07', scenario: 'Travel $1.8k', payload: { sender_id: 'ACC_CTX_006', receiver_id: 'ACC_NORM_107', transaction_amount: 1800, timestamp: '2026-05-09T12:15:21Z', transaction_type: 'payment' } },
  { case_id: 'batch_08', scenario: 'Salary $5.6k', payload: { sender_id: 'EMPLOYER_01', receiver_id: 'ACC_NORM_108', transaction_amount: 5600, timestamp: '2026-05-09T13:15:21Z', transaction_type: 'deposit' } },
  { case_id: 'batch_09', scenario: 'Wire $25k', payload: { sender_id: 'ACC_CTX_007', receiver_id: 'ACC_CTX_008', transaction_amount: 25000, timestamp: '2026-05-09T14:15:21Z', transaction_type: 'transfer' } },
  { case_id: 'batch_10', scenario: 'Near Threshold $9.85k', payload: { sender_id: 'ACC_CTX_009', receiver_id: 'ACC_CTX_010', transaction_amount: 9850, timestamp: '2026-05-09T15:15:21Z', transaction_type: 'cash_out' } },
  { case_id: 'batch_11', scenario: 'Rapid Mobile $7.2k', payload: { sender_id: 'ACC_CTX_011', receiver_id: 'ACC_CTX_012', transaction_amount: 7200, timestamp: '2026-05-09T16:15:21Z', transaction_type: 'transfer' } },
  { case_id: 'batch_12', scenario: 'Micro $12', payload: { sender_id: 'ACC_CTX_013', receiver_id: 'ACC_CTX_014', transaction_amount: 12, timestamp: '2026-05-09T17:15:21Z', transaction_type: 'payment' } },
  { case_id: 'batch_13', scenario: 'Unknown Pair $50k', payload: { sender_id: 'UNK_A_1', receiver_id: 'UNK_B_1', transaction_amount: 50000, timestamp: '2026-05-09T18:15:21Z', transaction_type: 'transfer' } },
  { case_id: 'batch_14', scenario: 'Late Night Cash Out', payload: { sender_id: 'ACC_CTX_015', receiver_id: 'ACC_CTX_016', transaction_amount: 5100, timestamp: '2026-05-09T23:40:00Z', transaction_type: 'cash_out' } },
  { case_id: 'batch_15', scenario: 'Round Amount $10k', payload: { sender_id: 'ACC_CTX_017', receiver_id: 'ACC_CTX_018', transaction_amount: 10000, timestamp: '2026-05-09T19:15:21Z', transaction_type: 'transfer' } },
  { case_id: 'batch_16', scenario: 'Small Deposit $500', payload: { sender_id: 'ACC_CTX_019', receiver_id: 'ACC_CTX_020', transaction_amount: 500, timestamp: '2026-05-09T20:15:21Z', transaction_type: 'deposit' } },
  { case_id: 'batch_17', scenario: 'Foreign Wire $68k', payload: { sender_id: 'ACC_CTX_021', receiver_id: 'ACC_CTX_022', transaction_amount: 68000, timestamp: '2026-05-09T21:15:21Z', transaction_type: 'transfer' } },
  { case_id: 'batch_18', scenario: 'Rent Payment $2.1k', payload: { sender_id: 'ACC_CTX_023', receiver_id: 'LANDLORD_01', transaction_amount: 2100, timestamp: '2026-05-09T22:15:21Z', transaction_type: 'payment' } },
];

/* ═══════════════════════════════════════════════════════════════════
   Context test data
   ═══════════════════════════════════════════════════════════════════ */

const normalContext = [
  { sender_id: 'ACC_CTX_001', receiver_id: 'ACC_NORM_102', transaction_amount: 117, timestamp: '2026-05-09T06:15:21Z', transaction_type: 'payment' },
  { sender_id: 'ACC_CTX_001', receiver_id: 'ACC_NORM_101', transaction_amount: 89, timestamp: '2026-05-09T05:35:21Z', transaction_type: 'payment' },
  { sender_id: 'ACC_CTX_001', receiver_id: 'ACC_NORM_100', transaction_amount: 142, timestamp: '2026-05-09T05:05:21Z', transaction_type: 'payment' },
  { sender_id: 'ACC_CTX_001', receiver_id: 'ACC_NORM_099', transaction_amount: 68, timestamp: '2026-05-09T04:45:21Z', transaction_type: 'transfer' },
];

const riskyContext = [
  { sender_id: 'ACC_CTX_001', receiver_id: 'ACC_RISK_101', transaction_amount: 9800, timestamp: '2026-05-09T05:58:00Z', transaction_type: 'cash_out' },
  { sender_id: 'ACC_CTX_001', receiver_id: 'ACC_RISK_102', transaction_amount: 9950, timestamp: '2026-05-09T06:01:00Z', transaction_type: 'transfer' },
  { sender_id: 'ACC_CTX_001', receiver_id: 'ACC_RISK_103', transaction_amount: 10050, timestamp: '2026-05-09T06:03:00Z', transaction_type: 'transfer' },
  { sender_id: 'ACC_CTX_001', receiver_id: 'ACC_RISK_104', transaction_amount: 9900, timestamp: '2026-05-09T06:07:00Z', transaction_type: 'withdrawal' },
  { sender_id: 'ACC_CTX_001', receiver_id: 'ACC_RISK_105', transaction_amount: 10100, timestamp: '2026-05-09T06:11:00Z', transaction_type: 'cash_out' },
];

/* ═══════════════════════════════════════════════════════════════════
   Helpers
   ═══════════════════════════════════════════════════════════════════ */

function getSinglePayload() {
  return {
    sender_id: document.getElementById('sender_id').value.trim(),
    receiver_id: document.getElementById('receiver_id').value.trim(),
    transaction_amount: Number(document.getElementById('transaction_amount').value),
    timestamp: toIsoWithZ(document.getElementById('timestamp').value),
    transaction_type: document.getElementById('transaction_type').value,
  };
}

function updateSinglePreview() {
  const payload = getSinglePayload();
  if (payload.sender_id && payload.receiver_id && payload.timestamp) {
    renderCurrentTransactionSummary(payload);
  }
}

function validateSinglePayload(payload) {
  if (!payload.sender_id || !payload.receiver_id || !payload.timestamp || Number.isNaN(payload.transaction_amount)) {
    throw new Error('Please complete the transaction form before analyzing.');
  }
}

/* ═══════════════════════════════════════════════════════════════════
   Tab switching — matches the new data-tab="single"|"batch"|"context"
   ═══════════════════════════════════════════════════════════════════ */

function applyActiveTab(tabId) {
  setActiveTabState(tabId);
  for (const panel of document.querySelectorAll('.tab-content')) {
    const panelTabId = panel.id.replace('tab-', '');
    panel.classList.toggle('active', panelTabId === tabId);
  }
  for (const button of document.querySelectorAll('.tab-btn')) {
    button.classList.toggle('active', button.dataset.tab === tabId);
  }
}

/* ═══════════════════════════════════════════════════════════════════
   Core actions
   ═══════════════════════════════════════════════════════════════════ */

async function runSinglePrediction() {
  const payload = getSinglePayload();
  validateSinglePayload(payload);
  setLoading('single', true);
  try {
    const result = await predictTransaction(payload);
    setLatestResult(result);
    renderSingleResult(result, state.modelInfo);
  } catch (error) {
    setError('single', error.message);
  }
}

async function runContextPrediction() {
  const payload = getSinglePayload();
  validateSinglePayload(payload);

  let recentTransactions;
  try {
    recentTransactions = JSON.parse(document.getElementById('recentTransactionsJson').value || '[]');
  } catch (error) {
    setError('context', 'Context JSON must be valid JSON.');
    return;
  }

  setLoading('context', true);
  try {
    const result = await predictWithContext({
      transaction: payload,
      recent_transactions: recentTransactions,
      simulate_only: true,
    });
    setContextResult(result);
    renderContextResult(result, state.modelInfo);
  } catch (error) {
    setError('context', error.message);
  }
}

async function runBatchPrediction() {
  setLoading('batch', true);
  const statusEl = document.getElementById('batchStatus');
  if (statusEl) statusEl.textContent = `Running 0/${batchCases.length}...`;

  try {
    const result = await predictBatch({ cases: batchCases });
    setBatchResult(result);
    renderBatchResults(result);
    if (statusEl) {
      const summary = result.summary || {};
      statusEl.textContent = `Done! ${summary.cases_completed || 0} cases, ${summary.unique_probability_count || 0} unique probabilities.`;
    }
  } catch (error) {
    setError('batch', error.message);
    if (statusEl) statusEl.textContent = `Error: ${error.message}`;
  }
}

/* ═══════════════════════════════════════════════════════════════════
   Scenario + Form management
   ═══════════════════════════════════════════════════════════════════ */

function applyScenario(name) {
  const scenario = scenarios[name];
  if (!scenario) return;
  populateForm(scenario);
  updateSinglePreview();
}

function resetForm() {
  document.getElementById('analysisForm').reset();
  document.getElementById('timestamp').value = new Date().toISOString().slice(0, 16);
  updateSinglePreview();
  setEmpty('single');
}

/* ═══════════════════════════════════════════════════════════════════
   Init
   ═══════════════════════════════════════════════════════════════════ */

function buildScenarioButtons() {
  const container = document.getElementById('scenarioButtons');
  if (!container) return;
  for (const [key, data] of Object.entries(scenarios)) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'scenario-btn';
    btn.textContent = data.name;
    btn.addEventListener('click', () => applyScenario(key));
    container.appendChild(btn);
  }
}

function initTabs() {
  for (const button of document.querySelectorAll('.tab-btn')) {
    button.addEventListener('click', () => applyActiveTab(button.dataset.tab));
  }
}

function initForm() {
  document.getElementById('analysisForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    await runSinglePrediction();
  });

  document.getElementById('resetBtn').addEventListener('click', resetForm);
  document.getElementById('sender_id').addEventListener('input', updateSinglePreview);
  document.getElementById('receiver_id').addEventListener('input', updateSinglePreview);
  document.getElementById('transaction_amount').addEventListener('input', updateSinglePreview);
  document.getElementById('timestamp').addEventListener('change', updateSinglePreview);
  document.getElementById('transaction_type').addEventListener('change', updateSinglePreview);
}

function initBatch() {
  document.getElementById('batchRunButton').addEventListener('click', runBatchPrediction);
}

function renderContextSample(sample) {
  document.getElementById('recentTransactionsJson').value = JSON.stringify(sample, null, 2);
}

function initContext() {
  document.getElementById('contextRunButton').addEventListener('click', runContextPrediction);
  document.getElementById('contextLoadNormal').addEventListener('click', () => renderContextSample(normalContext));
  document.getElementById('contextLoadRisky').addEventListener('click', () => renderContextSample(riskyContext));
}

async function initModelInfo() {
  try {
    const info = await getModelInfo();
    setModelInfo(info);
    renderModelInfo(info);
  } catch (error) {
    const el = document.getElementById('modelStatus');
    if (el) el.textContent = 'Unable to load model metadata.';
  }
}

function init() {
  // Default timestamp
  document.getElementById('timestamp').value = new Date().toISOString().slice(0, 16);

  // Build scenario buttons
  buildScenarioButtons();

  // Load default scenario
  populateForm(scenarios.suspicious);
  updateSinglePreview();

  // Set empty states
  setEmpty('single');
  setEmpty('context');
  setEmpty('batch');

  // Init all event handlers
  initTabs();
  initForm();
  initBatch();
  initContext();
  initModelInfo();

  // Set default active tab
  applyActiveTab('single');

  // Pre-load normal context
  renderContextSample(normalContext);
}

init();
