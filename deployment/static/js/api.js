async function requestJson(url, payload, method = 'POST') {
  const response = await fetch(url, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    const message = data && data.error ? data.error : 'Request failed';
    const details = data && data.details ? data.details : data;
    const error = new Error(message);
    error.details = details;
    throw error;
  }

  return data;
}

export async function predictTransaction(payload) {
  return requestJson('/predict', payload);
}

export async function predictWithContext(payload) {
  return requestJson('/predict-context', payload);
}

export async function predictBatch(payload) {
  return requestJson('/batch-predict', payload);
}

export async function getModelInfo() {
  return requestJson('/model-info', undefined, 'GET');
}
