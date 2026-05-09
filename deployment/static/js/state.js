export const state = {
  activeTab: 'singleTab',
  modelInfo: null,
  latestResult: null,
  contextResult: null,
  batchResult: null,
  history: [],
};

export function setActiveTab(tabId) {
  state.activeTab = tabId;
}

export function setModelInfo(info) {
  state.modelInfo = info;
}

export function setLatestResult(result) {
  state.latestResult = result;
  state.history = [result, ...state.history].slice(0, 8);
}

export function setBatchResult(result) {
  state.batchResult = result;
}

export function setContextResult(result) {
  state.contextResult = result;
  state.history = [result, ...state.history].slice(0, 8);
}
