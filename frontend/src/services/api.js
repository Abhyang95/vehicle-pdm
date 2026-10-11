import axios from "axios";


const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";


const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

// ==========================================================
// VEHICLE HEALTH
// ==========================================================

export async function getVehicleHealth(
  unit,
  historyPoints = 10
) {
  const response = await api.get(
    `/vehicle/${unit}/health`,
    {
      params: {
        history_points: historyPoints,
      },
    }
  );

  return response.data;
}

// ==========================================================
// MODEL INFO
// ==========================================================

export async function getModelInfo() {
  const response =
    await api.get("/model-info");

  return response.data;
}

// ==========================================================
// API HEALTH
// ==========================================================

export async function getApiHealth() {
  const response =
    await api.get("/health");

  return response.data;
}

// ==========================================================
// VEHICLE TELEMETRY
// ==========================================================

export async function getVehicleTelemetry(
  unit
) {
  const response = await api.get(
    `/vehicle/${unit}/telemetry`
  );

  return response.data;
}

// ==========================================================
// DIAGNOSTICS
// IMPORTANT:
// Backend expects:
// {
//   telemetry: [...]
// }
// ==========================================================

export async function getVehicleDiagnostics(
  telemetry
) {
  const response = await api.post(
    "/diagnostics",
    {
      telemetry,
    }
  );

  return response.data;
}

// ==========================================================
// MAINTENANCE
// IMPORTANT:
// Backend expects:
// {
//   telemetry: [...]
// }
// ==========================================================

export async function getMaintenanceRecommendation(
  telemetry
) {
  const response = await api.post(
    "/maintenance",
    {
      telemetry,
    }
  );

  return response.data;
}

// ==========================================================
// AI ASSISTANT
// ==========================================================

export async function askVehicleAssistant(
  unit,
  question,
  sessionId = null,
  cycle = null
) {
  const payload = {
    unit,
    question,
  };

  if (sessionId) {
    payload.session_id =
      sessionId;
  }

  if (cycle != null) {
    payload.cycle = cycle;
  }

  const response = await api.post(
    "/assistant",
    payload
  );

  return response.data;
}

// ==========================================================
// CSV UPLOAD
// ==========================================================

export async function uploadReplayCsv(
  file
) {
  const formData =
    new FormData();

  formData.append(
    "file",
    file
  );

  const response = await api.post(
    "/upload-csv",
    formData,
    {
      headers: {
        "Content-Type":
          "multipart/form-data",
      },
    }
  );

  return response.data;
}

// ==========================================================
// REPLAY SESSION
// ==========================================================

export async function getReplaySession(
  sessionId
) {
  const response = await api.get(
    `/replay/${encodeURIComponent(
      sessionId
    )}`
  );

  return response.data;
}

// ==========================================================
// REPLAY ANALYSIS
// IMPORTANT:
// Backend endpoint is GET with query params.
// ==========================================================

export async function analyzeReplay(
  sessionId,
  unit,
  cycle
) {
  const response = await api.get(
    `/replay/${encodeURIComponent(
      sessionId
    )}/analyze`,
    {
      params: {
        unit,
        cycle,
      },
    }
  );

  return response.data;
}