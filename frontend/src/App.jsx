import { useEffect, useState } from "react";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

import {
  getApiHealth,
  getVehicleHealth,
  getVehicleTelemetry,
  getVehicleDiagnostics,
  getMaintenanceRecommendation,
  askVehicleAssistant,
  uploadReplayCsv,
  analyzeReplay,
} from "./services/api";

import ReactMarkdown from "react-markdown";


import "./App.css";


const NAV_ITEMS = [
   ["overview", "Overview"],
  ["replay", "Replay"],
  ["explainability", "Explainability"],
  ["maintenance", "Maintenance"],
  ["assistant", "Assistant"],
];


function App() {

  // ==========================================================
  // CORE DASHBOARD STATE
  // ==========================================================

  const [apiStatus, setApiStatus] = useState("CHECKING...");
  const [vehicle, setVehicle] = useState(null);
  const [error, setError] = useState("");
  const [diagnostics, setDiagnostics] = useState(null);
  const [maintenance, setMaintenance] = useState(null);

  // ==========================================================
  // AI ASSISTANT STATE
  // ==========================================================

  const [assistantQuestion, setAssistantQuestion] = useState("");
  const [assistantAnswer, setAssistantAnswer] = useState("");
  const [assistantSources, setAssistantSources] = useState([]);
  const [assistantLoading, setAssistantLoading] = useState(false);
  const [assistantError, setAssistantError] = useState("");

  // ==========================================================
  // REPLAY STATE
  // ==========================================================

  const [replayFile, setReplayFile] = useState(null);
  const [replaySession, setReplaySession] = useState(null);

  const [replayUnit, setReplayUnit] = useState(1);

  // Selected cycle in the slider.
  const [replayCycle, setReplayCycle] = useState(1);

  // Cycle that has actually been analyzed.
  const [analyzedReplayCycle, setAnalyzedReplayCycle] =
    useState(null);

  const [replayResult, setReplayResult] = useState(null);

  const [replayLoading, setReplayLoading] = useState(false);
  const [replayUploading, setReplayUploading] = useState(false);
  const [replayError, setReplayError] = useState("");
  const [replayEnabled, setReplayEnabled] = useState(false);

  // ==========================================================
  // INITIAL DASHBOARD LOAD
  // ==========================================================

  useEffect(() => {
    async function loadDashboardData() {
      try {
        setError("");

        const health = await getApiHealth();

        setApiStatus(
          health.status === "healthy"
            ? "ONLINE"
            : "OFFLINE"
        );

        const vehicleData = await getVehicleHealth(1, 10);

        setVehicle(vehicleData);

        const telemetryData =
          await getVehicleTelemetry(1);

        const diagnosticData =
          await getVehicleDiagnostics(
            telemetryData.telemetry
          );

        setDiagnostics(diagnosticData);

        const maintenanceData =
          await getMaintenanceRecommendation(
            telemetryData.telemetry
          );

        setMaintenance(maintenanceData);
      } catch (err) {
        console.error(
          "Dashboard API error:",
          err
        );

        setApiStatus("OFFLINE");

        setError(
          err.response?.data?.detail ||
            err.message ||
            "Unable to connect to FastAPI."
        );
      }
    }

    loadDashboardData();
  }, []);

  // ==========================================================
  // REPLAY CSV UPLOAD
  // ==========================================================

  async function handleReplayUpload() {
    if (!replayFile) {
      setReplayError(
        "Please select a CSV file first."
      );
      return;
    }

    try {
      setReplayUploading(true);
      setReplayError("");

      setReplayResult(null);
      setAnalyzedReplayCycle(null);

      // Clear old assistant result because the data source
      // is changing.
      setAssistantAnswer("");
      setAssistantSources([]);
      setAssistantError("");

      const result =
        await uploadReplayCsv(replayFile);

      setReplaySession(result);
      setReplayEnabled(true);

      const firstUnit =
        result.units?.[0];

      if (firstUnit !== undefined) {
        setReplayUnit(firstUnit);

        const metadata =
          result.unit_metadata?.find(
            (item) =>
              item.unit === firstUnit
          );

        setReplayCycle(
          metadata?.start_cycle ??
            result.cycle_range?.min ??
            1
        );
      } else {
        setReplayCycle(
          result.cycle_range?.min ?? 1
        );
      }
    } catch (err) {
      console.error(
        "Replay upload failed:",
        err
      );

      setReplayError(
        err.response?.data?.detail ||
          err.message ||
          "Unable to upload replay CSV."
      );
    } finally {
      setReplayUploading(false);
    }
  }

  // ==========================================================
  // EXPLICIT REPLAY ANALYSIS
  // ==========================================================

  async function handleAnalyzeReplay() {
    if (!replaySession?.session_id) {
      setReplayError(
        "Please upload a telemetry CSV first."
      );
      return;
    }

    try {
      setReplayLoading(true);
      setReplayError("");

      // Clear old Assistant result because the diagnostic
      // context is about to change.
      setAssistantAnswer("");
      setAssistantSources([]);
      setAssistantError("");

      const result = await analyzeReplay(
        replaySession.session_id,
        replayUnit,
        replayCycle
      );

      setReplayResult(result);
      setAnalyzedReplayCycle(
        result.current_cycle
      );
    } catch (err) {
      console.error(
        "Replay analysis failed:",
        err
      );

      setReplayResult(null);
      setAnalyzedReplayCycle(null);

      setReplayError(
        err.response?.data?.detail ||
          err.message ||
          "Unable to analyze replay cycle."
      );
    } finally {
      setReplayLoading(false);
    }
  }

  // ==========================================================
  // REPLAY UNIT CHANGE
  // ==========================================================

  function handleReplayUnitChange(event) {
    const newUnit =
      Number(event.target.value);

    setReplayUnit(newUnit);

    const metadata =
      replaySession?.unit_metadata?.find(
        (item) =>
          item.unit === newUnit
      );

    const newStartCycle =
      metadata?.start_cycle ??
      replaySession?.cycle_range?.min ??
      1;

    setReplayCycle(newStartCycle);

    // The previously analyzed result belongs to the
    // previous unit, so clear it.
    setReplayResult(null);
    setAnalyzedReplayCycle(null);

    setAssistantAnswer("");
    setAssistantSources([]);
    setAssistantError("");
  }

  // ==========================================================
  // REPLAY CYCLE CHANGE
  // ==========================================================

  function handleReplayCycleChange(event) {
    const newCycle =
      Number(event.target.value);

    setReplayCycle(newCycle);

    // Do not automatically analyze.
    // The user must press "Analyze Cycle".
    setReplayError("");
  }

  // ==========================================================
  // EXIT REPLAY
  // ==========================================================

  function handleExitReplay() {
    setReplayEnabled(false);

    setReplaySession(null);
    setReplayResult(null);

    setReplayFile(null);

    setReplayError("");
    setAnalyzedReplayCycle(null);

    // Clear replay Assistant answer.
    setAssistantAnswer("");
    setAssistantSources([]);
    setAssistantError("");

    setReplayUnit(1);
    setReplayCycle(1);
  }

  // ==========================================================
  // ACTIVE DASHBOARD STATE
  // ==========================================================

  const activeVehicle =
    replayEnabled && replayResult
      ? {
          unit: replayResult.unit,

          current_cycle:
            replayResult.current_cycle,

          predicted_rul:
            replayResult.prediction
              ?.predicted_rul ?? 0,

          failure_probability:
            replayResult.prediction
              ?.failure_probability ?? 0,

          severity:
            replayResult.prediction
              ?.severity ?? "UNKNOWN",

          overall_sensor_health:
            replayResult.overall_health
              ?.score ?? 0,

          overall_health_status:
            replayResult.overall_health
              ?.status ?? "UNKNOWN",

          subsystem_health:
            replayResult.subsystem_health ||
            {},
        }
      : vehicle;

  const activeDiagnostics =
    replayEnabled && replayResult
      ? {
          unit: replayResult.unit,

          current_cycle:
            replayResult.current_cycle,

          failure_probability:
            replayResult.prediction
              ?.failure_probability ?? 0,

          predicted_rul:
            replayResult.prediction
              ?.predicted_rul ?? 0,

          risk_status:
            replayResult.prediction
              ?.risk_status ?? "UNKNOWN",

          severity:
            replayResult.prediction
              ?.severity ?? "UNKNOWN",

          overall_sensor_health:
            replayResult.overall_health
              ?.score ?? 0,

          overall_health_status:
            replayResult.overall_health
              ?.status ?? "UNKNOWN",

          shap_contributors:
            replayResult.shap_contributors ||
            [],

          feature_count:
            replayResult.feature_count ?? 0,
        }
      : diagnostics;

  const activeMaintenance =
    replayEnabled && replayResult
      ? replayResult.maintenance || null
      : maintenance;

  // ==========================================================
  // ACTIVE DATA SOURCE
  // ==========================================================

  const activeDataSource =
    replayEnabled
      ? {
          mode: "replay",

          label: "UPLOADED REPLAY",

          filename:
            replayResult?.data_source
              ?.filename ||
            replaySession?.filename ||
            "Uploaded telemetry",

          session_id:
            replayResult?.data_source
              ?.session_id ||
            replaySession?.session_id ||
            null,
        }
      : {
          mode: "demo",

          label: "DEMO TELEMETRY",

          filename:
            vehicle?.data_source
              ?.filename ||
            "Built-in demonstration trajectory",

          session_id: null,
        };

  const activeDataSourceIsReplay =
    activeDataSource.mode === "replay";

  const replayAwaitingAnalysis =
  replayEnabled && !replayResult;
  // ==========================================================
  // REPLAY METADATA
  // ==========================================================

  const selectedUnitMetadata =
    replaySession?.unit_metadata?.find(
      (item) =>
        item.unit === replayUnit
    );

  const replayStartCycle =
    selectedUnitMetadata?.start_cycle ??
    replaySession?.cycle_range?.min ??
    1;

  const replayEndCycle =
    selectedUnitMetadata?.end_cycle ??
    replaySession?.cycle_range?.max ??
    1;
 
  // ==========================================================
  // REPLAY RUL VALIDATION
  // ==========================================================

  const replayRulHistory =
    replayResult?.rul_history || [];
  const replayRulChartData = replayRulHistory.map(
  (point) => ({
    cycle: Number(point.cycle),
    rul: Number(point.predicted_rul),
  })
);
  const replayGroundTruthAvailable =
    replayResult?.ground_truth_available === true;

  const replayGroundTruthSource =
    replayResult?.ground_truth_source || null;

  
  const currentPredictedRul = Number(
    replayResult?.prediction?.predicted_rul ?? 0
  );

  const currentTrueRul = replayGroundTruthAvailable
    ? replayRulHistory.find(
        (point) => point.cycle === replayResult?.current_cycle
      )?.true_rul
    : null;

  const currentRulError =
    currentTrueRul != null
      ? Math.abs(currentPredictedRul - currentTrueRul)
      : null;

 


  // ==========================================================
  // AI ASSISTANT
  // ==========================================================

  async function handleAssistantSubmit(
    event
  ) {
    event.preventDefault();

    const question =
      assistantQuestion.trim();

    if (!question) {
      setAssistantError(
        "Please enter a vehicle diagnostic question."
      );

      return;
    }

    // Do not allow Assistant to operate on an uploaded
    // replay before a cycle has actually been analyzed.
    if (
      activeDataSourceIsReplay &&
      !replayResult
    ) {
      setAssistantError(
        "Analyze a replay cycle first, then ask the Assistant about that cycle."
      );

      return;
    }

    try {
      setAssistantLoading(true);
      setAssistantError("");

      const assistantUnit =
        activeVehicle?.unit ?? 1;

      const assistantCycle =
        activeVehicle?.current_cycle ??
        null;

      const assistantSessionId =
        activeDataSourceIsReplay
          ? activeDataSource.session_id
          : null;

      const result =
        await askVehicleAssistant(
          assistantUnit,
          question,
          assistantSessionId,
          assistantCycle
        );

      setAssistantAnswer(
        result.answer || ""
      );

      setAssistantSources(
        result.sources || []
      );
    } catch (err) {
      console.error(
        "Assistant request failed:",
        err
      );

      setAssistantAnswer("");
      setAssistantSources([]);

      setAssistantError(
        err.response?.data?.detail ||
          err.message ||
          "Unable to get an answer from the vehicle assistant."
      );
    } finally {
      setAssistantLoading(false);
    }
  }

  // ==========================================================
  // LOADING / ERROR STATE
  // ==========================================================

  if (!vehicle) {
    return (
      <div className="app">
        <header className="topbar">

          <div className="brand-block">

            <div className="brand-mark">
              VP
            </div>

            <div>
              <h1>
                Vehicle Predictive Maintenance
              </h1>

              <p className="subtitle">
                Predictive diagnostics & vehicle
                health monitoring
              </p>
            </div>

          </div>

          <div
            className={`api-status ${
              apiStatus === "ONLINE"
                ? "online"
                : "offline"
            }`}
          >
            <span className="status-dot" />

            API {apiStatus}
          </div>

        </header>

        <main className="loading-shell">

          <div className="loading-card">

            <div className="loading-indicator" />

            {error ? (
              <>
                <span className="eyebrow">
                  SYSTEM CONNECTION
                </span>

                <h2>
                  Unable to load vehicle data
                </h2>

                <p>
                  {error}
                </p>
              </>
            ) : (
              <>
                <span className="eyebrow">
                  INITIALIZING PLATFORM
                </span>

                <h2>
                  Loading vehicle diagnostics
                </h2>

                <p>
                  Connecting to the
                  predictive-maintenance API
                  and loading the current
                  vehicle state.
                </p>
              </>
            )}

          </div>

        </main>
      </div>
    );
  }

  // ==========================================================
  // CHART DATA
  // ==========================================================

  let healthChartData;

  if (
    replayEnabled &&
    replayResult
  ) {
    healthChartData =
      vehicle.history
        .filter(
          (point) =>
            point.cycle <=
            replayResult.current_cycle
        )
        .map((point) => ({
          cycle: point.cycle,

          health:
            point.overall_sensor_health,

          rul:
            point.predicted_rul,

          risk:
            point.failure_probability *
            100,
        }));

    healthChartData.push({
      cycle:
        replayResult.current_cycle,

      health:
        replayResult.overall_health
          ?.score ?? 0,

      rul:
        replayResult.prediction
          ?.predicted_rul ?? 0,

      risk:
        (replayResult.prediction
          ?.failure_probability ?? 0) *
        100,
    });
  } else {
    healthChartData =
      vehicle.history.map(
        (point) => ({
          cycle: point.cycle,

          health:
            point.overall_sensor_health,

          rul:
            point.predicted_rul,

          risk:
            point.failure_probability *
            100,
        })
      );
  }


  // ==========================================================
  // DATA HELPERS
  // ==========================================================

  const subsystemEntries =
    Object.entries(
      activeVehicle.subsystem_health ||
        {}
    );

  const shapContributors =
    activeDiagnostics
      ?.shap_contributors || [];

  const maintenancePriority =
    activeMaintenance?.priority ||
    "N/A";

  const maintenanceAction =
    activeMaintenance?.action ||
    "No maintenance recommendation available.";

  const maintenanceRulContext =
    activeMaintenance?.rul_context ??
    `${activeVehicle.predicted_rul} cycles`;

  const focusSubsystems =
    activeMaintenance
      ?.focus_subsystems || [];

  const failurePercentage =
    activeVehicle.failure_probability *
    100;

  const riskStatus =
    activeDiagnostics?.risk_status ||
    "UNKNOWN";

  const severity =
    activeDiagnostics?.severity ||
    activeVehicle.severity ||
    "UNKNOWN";

  const healthStatus =
    activeDiagnostics
      ?.overall_health_status ||
    activeVehicle
      .overall_health_status ||
    "UNKNOWN";

  const isCritical =
    healthStatus === "CRITICAL";

  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="app">

      {/* ======================================================
          HEADER
      ======================================================= */}

      <header className="topbar">

        <div className="brand-block">

          <div className="brand-mark">
            VP
          </div>

          <div>
            <h1>
              Vehicle Predictive Maintenance
            </h1>

            <p className="subtitle">
              Predictive diagnostics & vehicle
              health monitoring
            </p>
          </div>

        </div>

        <div className="header-right">

          <div className="vehicle-context">

            <span className="eyebrow">
              ACTIVE VEHICLE
            </span>

            
<strong>
  {replayAwaitingAnalysis
    ? "Awaiting analysis"
    : activeDataSourceIsReplay
      ? `Unit ${activeVehicle.unit} of ${
          replaySession?.unit_count ?? 1
        }`
      : `Unit ${activeVehicle.unit}`}
</strong>

          </div>

          <div
            className={`data-source-badge ${
              activeDataSourceIsReplay
                ? "replay"
                : "demo"
            }`}
          >

            <span className="status-dot" />

            <span
  title={
    activeDataSourceIsReplay
      ? activeDataSource.filename
      : activeDataSource.label
  }
>
  {activeDataSourceIsReplay
    ? activeDataSource.filename
    : activeDataSource.label}
</span>

          </div>

          <div
            className={`api-status ${
              apiStatus === "ONLINE"
                ? "online"
                : "offline"
            }`}
          >

            <span className="status-dot" />

            API {apiStatus}

          </div>

        </div>

      
      </header>

      <nav className="section-nav" aria-label="Dashboard sections">
        {NAV_ITEMS.map(([id, label]) => (
          <a key={id} href={`#${id}`}>
            {label}
          </a>
        ))}
      </nav>

      <main className="dashboard">


        {/* ====================================================
            PAGE INTRO
        ===================================================== */}

        <section className="page-intro"id="overview">

          <div>

            <span className="eyebrow">
              PREDICTIVE MAINTENANCE PLATFORM
            </span>

            <h2>
              Vehicle health intelligence
            </h2>

            <p>
              Machine-learning based
              failure-risk prediction,
              remaining useful life
              estimation, explainable
              diagnostics and AI-assisted
              maintenance analysis.
            </p>

          </div>

          <div className="intro-status">

            <span className="status-dot" />

            <div>

              <span>
                {activeDataSource.label}
              </span>

              <strong>
                Cycle{" "}
                {activeVehicle.current_cycle}
              </strong>

            </div>

          </div>

        </section>

        {/* ====================================================
            ACTIVE DATA SOURCE
        ===================================================== */}

        <section
          className={`data-source-panel ${
            activeDataSourceIsReplay
              ? "replay"
              : ""
          }`}
        >

          <div className="data-source-main">

            <div
              className={`data-source-indicator ${
                activeDataSourceIsReplay
                  ? "replay"
                  : ""
              }`}
            >
              <span className="status-dot" />
            </div>

            <div>

              <span className="eyebrow">
                DATA SOURCE
              </span>

              <h3>
                {activeDataSource.label}
              </h3>

              <p>
                {activeDataSourceIsReplay
                  ? activeDataSource.filename
                  : "Built-in demonstration trajectory"}
              </p>

            </div>

          </div>

          <div className="data-source-details">

            <div>
              <span>
                VEHICLE
              </span>

              <strong>
  {replayAwaitingAnalysis
    ? "Awaiting analysis"
    : `Unit ${activeVehicle.unit}`}
</strong>
            </div>

            <div>
              <span>
                CURRENT CYCLE
              </span>

              <strong>
                {activeVehicle.current_cycle}
              </strong>
            </div>

            <div>
              <span>
                {activeDataSourceIsReplay
                  ? "SESSION"
                  : "MODE"}
              </span>

              <strong>
                {activeDataSourceIsReplay
                  ? "REPLAY"
                  : "DEMO"}
              </strong>
            </div>

          </div>

        </section>
        {replayAwaitingAnalysis && (
  <div className="replay-pending-notice" role="status">
    <strong>Replay uploaded — analysis pending</strong>
    <p>
      Select a unit and cycle, then click Analyze.
      Dashboard metrics are still showing demonstration data
      until the replay analysis completes.
    </p>
  </div>
)}
        {/* ====================================================
            KPI GRID
        ===================================================== */}

        {!replayAwaitingAnalysis && (
          <section className="metrics-grid">

          <div className="metric-card metric-primary">

            <div className="metric-top">

              <span className="metric-label">
                PREDICTED RUL
              </span>

              <span className="metric-icon">
                RUL
              </span>

            </div>

            <div className="metric-value">
              {Number(
                activeVehicle.predicted_rul
              ).toFixed(2)}
            </div>

            <div className="metric-footer">

              <span>
                cycles remaining
              </span>

              <span className="metric-accent">
                MODEL OUTPUT
              </span>

            </div>

          </div>

          <div className="metric-card">

            <div className="metric-top">

              <span className="metric-label">
                FAILURE RISK
              </span>

              <span className="metric-icon">
                RISK
              </span>

            </div>

            <div className="metric-value">
              {failurePercentage.toFixed(2)}
              <small>%</small>
            </div>

            <div className="metric-footer">

              <span>
                probability estimate
              </span>

              <span
                className={`severity-text ${severity.toLowerCase()}`}
              >
                {severity}
              </span>

            </div>
             <div className="metric-note">
    Model-derived score, not a calibrated real-world probability.
  </div>
          </div>

          <div className="metric-card">

            <div className="metric-top">

              <span className="metric-label">
                CURRENT CYCLE
              </span>

              <span className="metric-icon">
                CYCLE
              </span>

            </div>

            <div className="metric-value">
              {activeVehicle.current_cycle}
            </div>

            <div className="metric-footer">

              <span>
                operating cycle
              </span>

              <span className="metric-neutral">
                UNIT{" "}
                {activeVehicle.unit}
              </span>

            </div>

          </div>

          <div
            className={`metric-card ${
              isCritical
                ? "metric-warning"
                : ""
            }`}
          >

            <div className="metric-top">

              <span className="metric-label">
                SENSOR HEALTH
              </span>

              <span className="metric-icon">
                HEALTH
              </span>

            </div>

            <div className="metric-value">
              {Number(
                activeVehicle
                  .overall_sensor_health
              ).toFixed(2)}
            </div>

            <div className="metric-footer">

              <span>
                telemetry health index
              </span>

              <span
  className={`severity-text ${
    String(healthStatus || "unknown").toLowerCase()
  }`}
>
  {healthStatus}
</span>

            </div>

          </div>

          </section>
)}

        {/* ====================================================
            TELEMETRY REPLAY
        ===================================================== */}

        <section className="panel replay-panel" id="replay">

          <div className="panel-header">

            <div>

              <span className="eyebrow">
                DATA EXPLORATION
              </span>

              <h2>
                Telemetry Replay
              </h2>

              <p>
                Upload telemetry and inspect
                predictive health
                cycle-by-cycle.
              </p>

            </div>

            {replayEnabled &&
              replayResult && (
                <div className="cycle-badge">
                  ANALYZED CYCLE{" "}
                  {replayResult.current_cycle}
                </div>
              )}

          </div>

          <div className="replay-content">

            {/* ==================================================
                FILE UPLOAD
            =================================================== */}

            <div className="upload-layout">

              <div className="file-control">

                <label
                  htmlFor="replay-csv"
                  className="detail-label"
                >
                  TELEMETRY CSV
                </label>

                <input
                  id="replay-csv"
                  className="file-input"
                  type="file"
                  accept=".csv"
                  onChange={(event) => {
                    setReplayFile(
                      event.target.files?.[0] ||
                        null
                    );

                    setReplayError("");
                  }}
                />

              </div>

              <button
                type="button"
                className="primary-button"
                onClick={handleReplayUpload}
                disabled={
                  replayUploading ||
                  !replayFile
                }
              >
                {replayUploading
                  ? "Uploading..."
                  : "Upload CSV"}
              </button>

              {replayEnabled && (
                <button
                  type="button"
                  className="secondary-button"
                  onClick={handleExitReplay}
                >
                  Exit Replay
                </button>
              )}

            </div>

            {/* ==================================================
                SESSION INFORMATION
            =================================================== */}

            {replaySession && (
              <div className="replay-session">

                <div className="session-item">

                  <span className="detail-label">
                    FILE
                  </span>

                  <strong>
                    {replaySession.filename}
                  </strong>

                </div>

                <div className="session-item">

                  <span className="detail-label">
                    ROWS
                  </span>

                  <strong>
                    {replaySession.row_count}
                  </strong>

                </div>

                <div className="session-item">

                  <span className="detail-label">
                    VEHICLES
                  </span>

                  <strong>
                    {replaySession.unit_count}
                  </strong>

                </div>

                <div className="session-item">

                  <span className="detail-label">
                    CYCLE RANGE
                  </span>

                  <strong>
                    {replaySession.cycle_range?.min}
                    {" — "}
                    {replaySession.cycle_range?.max}
                  </strong>

                </div>

              </div>
            )}

            {/* ==================================================
                REPLAY CONTROLS
            =================================================== */}

            {replaySession && (
              <div className="replay-controls">

                <div className="replay-control-top">

                  <div>

                    <span className="detail-label">
                      VEHICLE UNIT
                    </span>

                    <select
                      className="replay-select"
                      value={replayUnit}
                      onChange={
                        handleReplayUnitChange
                      }
                    >
                      {(
                        replaySession.units ||
                        []
                      ).map((unit) => (
                        <option
                          key={unit}
                          value={unit}
                        >
                          Unit {unit}
                        </option>
                      ))}
                    </select>

                  </div>

                  <div className="replay-cycle-display">

                    <span className="detail-label">
                      SELECTED CYCLE
                    </span>

                    <strong>
                      {replayCycle}
                    </strong>

                  </div>

                  <div className="replay-cycle-display">

                    <span className="detail-label">
                      ANALYZED CYCLE
                    </span>

                    <strong>
                      {analyzedReplayCycle ??
                        "—"}
                    </strong>

                  </div>

                </div>

                {/* ==================================================
                    SLIDER
                =================================================== */}

                <div className="slider-area">

                  <input
                    className="replay-slider"
                    type="range"
                    min={replayStartCycle}
                    max={replayEndCycle}
                    value={replayCycle}
                    onChange={
                      handleReplayCycleChange
                    }
                  />

                  <div className="slider-labels">

                    <span>
                      Cycle{" "}
                      {replayStartCycle}
                    </span>

                    <span>
                      Cycle{" "}
                      {replayEndCycle}
                    </span>

                  </div>

                </div>

                {/* ==================================================
                    ANALYZE BUTTON
                =================================================== */}

                <div className="replay-analysis-action">

                  <button
                    type="button"
                    className="primary-button replay-button"
                    onClick={
                      handleAnalyzeReplay
                    }
                    disabled={replayLoading}
                  >
                    {replayLoading
                      ? `Analyzing Cycle ${replayCycle}...`
                      : `Analyze Cycle ${replayCycle}`}
                  </button>

                  <span className="replay-analysis-hint">
                    Select a cycle and run
                    analysis to update the
                    dashboard.
                  </span>

                </div>

                {replayResult && (
                  <div className="inline-status success-status">
                    <span className="status-dot" />

                    Cycle{" "}
                    {replayResult.current_cycle}
                    {" "}
                    analyzed successfully.
                  </div>
                )}

                
                {replayError && (
                  <div className="error-box">
                    {replayError}
                  </div>
                )}

                {/* ==================================================
                    REPLAY RUL VALIDATION
                ================================================== */}

                {replayResult &&
                  replayRulHistory.length > 0 && (
                    <div className="replay-rul-validation">

                      <div className="replay-rul-header">

                        <div>
                          <span className="eyebrow">
                            MODEL VALIDATION
                          </span>

                          <h3>
                            Replay RUL Validation
                          </h3>

                          <p>
  {replayGroundTruthAvailable
    ? "Predicted remaining useful life compared with benchmark ground truth across the replay trajectory."
    : "Predicted remaining useful life across the uploaded telemetry trajectory. Benchmark ground truth is unavailable for this file."}
</p>
                        </div>

                        <div className="replay-rul-status">

                          {replayGroundTruthAvailable ? (
                            <>
                              <span className="validation-badge">
                                GROUND TRUTH AVAILABLE
                              </span>

                              <span className="validation-source">
                                {replayGroundTruthSource}
                              </span>
                            </>
                          ) : (
                            <span className="validation-badge neutral">
                              PREDICTION ONLY
                            </span>
                          )}

                        </div>

                      </div>

                      <div className="replay-rul-meta">

                        <div className="replay-rul-meta-item">
                          <span>
                            Analyzed Cycle
                          </span>

                          <strong>
                            {replayResult.current_cycle}
                          </strong>
                        </div>

                        <div className="replay-rul-meta-item">
                          <span>
                            Predicted RUL
                          </span>

                          <strong>
                            {Number(
                              replayResult.prediction
                                ?.predicted_rul ?? 0
                            ).toFixed(1)}
                            {" "}cycles
                          </strong>
                        </div>

                        

<div className="replay-rul-meta-item">
  <span>Ground Truth RUL</span>
  <strong>
    {currentTrueRul != null
      ? `${Number(currentTrueRul).toFixed(1)} cycles`
      : "Unavailable"}
  </strong>
</div>

<div className="replay-rul-meta-item">
  <span>Absolute Prediction Error</span>
  <strong>
    {currentRulError != null
      ? `${currentRulError.toFixed(2)} cycles`
      : "N/A"}
  </strong>
</div>



                      </div>

                      <div className="replay-rul-chart">

                        <ResponsiveContainer
                          width="100%"
                          height={300}
                        >

                          <LineChart
                            data={replayRulHistory}
                            margin={{
                              top: 10,
                              right: 20,
                              left: 5,
                              bottom: 5,
                            }}
                          >

                            
                             <CartesianGrid strokeDasharray="3 3" stroke="#232a30" vertical={false} />
                            <XAxis
                              dataKey="cycle"
                              tick={{ fontSize: 11 }}
                              label={{
                                value: "Cycle",
                                position: "insideBottom",
                                offset: -2,
                              }}
                            />

                            <YAxis
  domain={[0, 125]}
  ticks={[0, 25, 50, 75, 100, 125]}
  allowDataOverflow
  tick={{ fontSize: 11 }}
  label={{
    value: "RUL (cycles)",
    angle: -90,
    position: "insideLeft",
  }}
/>

                            <Tooltip
                              formatter={(
                                value,
                                name
                              ) => [
                                `${Number(value).toFixed(1)} cycles`,
                                name,
                              ]}
                              labelFormatter={(cycle) =>
                                `Cycle ${cycle}`
                              }
                            />

                            <Line
                              type="monotone"
                              dataKey="predicted_rul"
                              name="Predicted RUL"
                              stroke="#8fa6ba"
                              strokeWidth={2}
                              dot={false}
                              activeDot={{ r: 4 }}
                            />

                            {replayGroundTruthAvailable && (
                              <Line
                                type="monotone"
                                dataKey="true_rul"
                                name="True RUL"
                                stroke="#dfa65c"
                                strokeWidth={2}
                                dot={false}
                                activeDot={{ r: 4 }}
                              />
                            )}

                          </LineChart>

                        </ResponsiveContainer>

                      </div>

                      <div className="replay-rul-legend">

                        <span>
                          <span className="legend-line predicted" />
                          Predicted RUL
                        </span>

                        {replayGroundTruthAvailable && (
                          <span>
                            <span className="legend-line truth" />
                            True RUL
                          </span>
                        )}

                      </div>

                      {!replayGroundTruthAvailable && (
                        <p className="replay-rul-note">
                          True RUL is shown only for the recognized
                          NASA C-MAPSS FD001 benchmark replay.
                          Arbitrary uploaded telemetry does not
                          receive benchmark ground-truth labels.
                        </p>
                      )}

                    </div>
                  )}

              </div>
            )}


            {!replaySession && (
              <div className="empty-state">

                <span className="empty-icon">
                  CSV
                </span>

                <div>

                  <strong>
                    No replay session loaded
                  </strong>

                  <p>
                    Upload a compatible
                    telemetry CSV to enable
                    cycle-by-cycle analysis.
                  </p>

                </div>

              </div>
            )}

          </div>

        </section>

        {/* ====================================================
            HEALTH HISTORY
        ===================================================== */}

        <section className="panel chart-panel">

          <div className="panel-header">

            <div>

              <span className="eyebrow">
                VEHICLE CONDITION
              </span>

              <h2>
                Vehicle Health History
              </h2>

              <p>
                Telemetry-derived health
                trajectory across operating
                cycles.
              </p>

            </div>

            <div className="chart-stat">

              <span>
                CURRENT CYCLE
              </span>

              <strong>
                {activeVehicle.current_cycle}
              </strong>

            </div>

          </div>

          <div className="chart-container large-chart">

            <ResponsiveContainer
              width="100%"
              height={340}
            >

              <LineChart
                 data={
    replayEnabled && replayRulHistory.length > 0
      ? replayRulHistory.map((point) => ({
          cycle: Number(point.cycle),
          health: Number(point.overall_health),
        }))
      : healthChartData
  }
                margin={{
                  top: 12,
                  right: 20,
                  left: 0,
                  bottom: 8,
                }}
              >

                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#252c32"
                  vertical={false}
                />

                <XAxis
                  dataKey="cycle"
                  stroke="#65717b"
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 11 }}
                />

                <YAxis
                  stroke="#65717b"
                  domain={[0, 100]}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 11 }}
                />

                <Tooltip
                  contentStyle={{
                    background:
                      "#12171b",
                    border:
                      "1px solid #303940",
                    borderRadius: "8px",
                    color: "#e7edf1",
                    boxShadow:
                      "0 12px 30px rgba(0,0,0,.35)",
                  }}
                />

                <Line
                  type="monotone"
                  dataKey="health"
                  name="Overall Health"
                  stroke="#8fa6ba"
                  strokeWidth={3}
                  dot={false}
                  activeDot={{
                    r: 5,
                    strokeWidth: 2,
                  }}
                />

              </LineChart>

            </ResponsiveContainer>

          </div>

        </section>

        {/* ====================================================
            TREND GRID
        ===================================================== */}

        <section className="trend-grid">

          {/* RUL */}

          <div className="panel chart-panel">

            <div className="panel-header">

              <div>

                <span className="eyebrow">
                  PROGNOSTICS
                </span>

                <h2>
                  Predicted RUL Trend
                </h2>

                <p>
                  Estimated remaining useful
                  life across operating cycles.
                </p>

              </div>

            </div>

            <div className="chart-container compact-chart">

              <ResponsiveContainer
                width="100%"
                height={280}
              >

                <LineChart
                  data={
  replayEnabled && replayRulChartData.length > 0
    ? replayRulChartData
    : healthChartData
}
                  margin={{
                    top: 10,
                    right: 12,
                    left: 0,
                    bottom: 8,
                  }}
                >

                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#252c32"
                    vertical={false}
                  />

                  <XAxis
                    dataKey="cycle"
                    stroke="#65717b"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 10 }}
                  />

                  <YAxis
                    stroke="#65717b"
                    domain={[0, 125]}
                    ticks={[0, 25, 50, 75, 100, 125]}
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 10 }}
                  />

                  <Tooltip
                    formatter={(value) => [
                      `${Number(value).toFixed(2)} cycles`,
                      "Predicted RUL",
                    ]}
                    contentStyle={{
                      background:
                        "#12171b",
                      border:
                        "1px solid #303940",
                      borderRadius: "8px",
                    }}
                  />

                  <Line
                    type="monotone"
                    dataKey="rul"
                    name="Predicted RUL"
                    stroke="#8fa6ba"
                    strokeWidth={2.5}
                    dot={{r: 3}}
                    activeDot={{ r: 5 }}
                  />

                </LineChart>

              </ResponsiveContainer>

            </div>

          </div>

          {/* FAILURE RISK */}

          <div className="panel chart-panel">

            <div className="panel-header">

              <div>

                <span className="eyebrow">
                  RISK MODEL
                </span>

                <h2>
                  Failure Risk Trend
                </h2>

                <p>
                  Model-estimated probability
                  of failure.
                </p>

              </div>

            </div>

            <div className="chart-container compact-chart">

              <ResponsiveContainer
                width="100%"
                height={280}
              >

                <LineChart
                  data={
    replayEnabled && replayRulHistory.length > 0
      ? replayRulHistory.map((point) => ({
          cycle: Number(point.cycle),
          risk: Number(point.failure_risk) * 100,
        }))
      : healthChartData
  }
                  margin={{
                    top: 10,
                    right: 12,
                    left: 0,
                    bottom: 8,
                  }}
                >

                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#252c32"
                    vertical={false}
                  />

                  <XAxis
                    dataKey="cycle"
                    stroke="#65717b"
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 10 }}
                  />

                  <YAxis
                    stroke="#65717b"
                    domain={[0, 100]}
                    tickFormatter={(value) =>
                      `${value}%`
                    }
                    tickLine={false}
                    axisLine={false}
                    tick={{ fontSize: 10 }}
                  />

                  <Tooltip
                    formatter={(value) => [
                      `${Number(value).toFixed(2)}%`,
                      "Failure Risk",
                    ]}
                    contentStyle={{
                      background:
                        "#12171b",
                      border:
                        "1px solid #303940",
                      borderRadius: "8px",
                    }}
                  />

                  <Line
                    type="monotone"
                    dataKey="risk"
                    name="Failure Risk"
                    stroke="#dfa65c"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5 }}
                  />

                </LineChart>

              </ResponsiveContainer>

            </div>

          </div>

        </section>

        {/* ====================================================
            SUBSYSTEM HEALTH
        ===================================================== */}

        <section className="panel" id="explainability">

          <div className="panel-header">

            <div>

              <span className="eyebrow">
                TELEMETRY DIAGNOSTICS
              </span>

              <h2>
                Subsystem Health
              </h2>

              <p>
                Approximate telemetry-derived
                subsystem indicators.
              </p>

            </div>

            <div className="panel-note">
              PROJECT DEFINED
            </div>

          </div>

          <div className="subsystem-grid">

            {subsystemEntries.map(
              ([name, details]) => (
                <div
                  className="subsystem-card"
                  key={name}
                >

                  <div className="subsystem-top">

                    <span className="subsystem-name">
                      {name
                        .replace("_", " ")
                        .toUpperCase()}
                    </span>

                    <span
                      className={`health-status ${
                        details.status?.toLowerCase()
                      }`}
                    >
                      {details.status}
                    </span>

                  </div>

                  <div className="subsystem-score-row">

                    <span className="subsystem-health">
                      {Number(
                        details.health
                      ).toFixed(2)}
                    </span>

                    <span className="score-unit">
                      / 100
                    </span>

                  </div>

                  <div className="health-bar">

                    <div
                      className="health-fill"
                      style={{
                        width: `${Math.max(
                          0,
                          Math.min(
                            100,
                            details.health
                          )
                        )}%`,
                      }}
                    />

                  </div>

                  <div className="subsystem-foot">
                    Telemetry health index
                  </div>

                </div>
              )
            )}

          </div>

          <div className="section-disclaimer">
            These subsystem indicators are
            derived from sensor-group behavior
            in the simulated C-MAPSS benchmark
            and do not represent OEM component
            diagnostics.
          </div>

        </section>

        {/* ====================================================
    DIAGNOSTIC SUMMARY
===================================================== */}

<section className="panel">

  <div className="panel-header">

    <div>

      <span className="eyebrow">
        DECISION STATE
      </span>

      <h2>
        Diagnostic Summary
      </h2>

      <p>
        Current model and telemetry
        diagnostic state.
      </p>

    </div>

    <div
      className={`status-pill ${
        riskStatus.toLowerCase()
      }`}
    >
      {riskStatus}
    </div>

  </div>

  {!replayAwaitingAnalysis && (
    <div className="diagnostic-grid">

      <div className="diagnostic-item">

        <span className="detail-label">
          VEHICLE
        </span>

        <strong>
          Unit {activeVehicle.unit}
        </strong>

      </div>

      <div className="diagnostic-item">

        <span className="detail-label">
          CURRENT CYCLE
        </span>

        <strong>
          {activeVehicle.current_cycle}
        </strong>

      </div>

      <div className="diagnostic-item">

        <span className="detail-label">
          MODEL RISK
        </span>

        <strong>
          {riskStatus}
        </strong>

      </div>

      <div className="diagnostic-item">

        <span className="detail-label">
          SEVERITY
        </span>

        <strong
          className={severity.toLowerCase()}
        >
          {severity}
        </strong>

      </div>

      <div className="diagnostic-item">

        <span className="detail-label">
          FAILURE PROBABILITY
        </span>

        <strong>
          {failurePercentage.toFixed(2)}%
        </strong>

      </div>

      <div className="diagnostic-item">

        <span className="detail-label">
          PREDICTED RUL
        </span>

        <strong>
          {Number(
            activeVehicle.predicted_rul
          ).toFixed(2)}
        </strong>

        <span className="diagnostic-unit">
          cycles
        </span>

      </div>

      <div className="diagnostic-item">

        <span className="detail-label">
          SENSOR HEALTH
        </span>

        <strong>
          {Number(
            activeVehicle.overall_sensor_health
          ).toFixed(2)}
        </strong>

        <span className="diagnostic-unit">
          index
        </span>

      </div>

      <div className="diagnostic-item">

        <span className="detail-label">
          HEALTH STATUS
        </span>

        <strong
          className={healthStatus.toLowerCase()}
        >
          {healthStatus}
        </strong>

      </div>

    </div>
  )}

  {replayAwaitingAnalysis && (
    <div className="replay-pending-notice" role="status">
      <strong>Replay uploaded - analysis pending</strong>
      <p>
        Select a unit and cycle, then click Analyze.
        Diagnostic values will appear after the replay analysis completes.
      </p>
    </div>
  )}

          {/* ==================================================
              DIAGNOSTIC INDICATOR GUIDE
          ================================================== */}

          <div className="diagnostic-guide">

            <div className="diagnostic-guide-header">

              <span className="eyebrow">
                HOW TO READ THESE INDICATORS
              </span>

              <p>
                These indicators describe different aspects
                of vehicle condition and are not interchangeable.
              </p>

            </div>

            <div className="diagnostic-guide-grid">

              <div className="diagnostic-guide-item">

                <strong>
                  FAILURE RISK
                </strong>

                <span>
                  Model-estimated probability that the vehicle
                  is approaching the defined failure condition.
                </span>

                <small>
                  LOW_RISK &lt; 15% · WATCH 15–29.99% ·
                  HIGH_RISK ≥ 30%
                </small>

              </div>

              <div className="diagnostic-guide-item">

                <strong>
                  SEVERITY
                </strong>

                <span>
                  Decision urgency derived from failure
                  probability and predicted RUL.
                </span>

                <small>
                  LOW · MEDIUM · HIGH · CRITICAL
                </small>

              </div>

              <div className="diagnostic-guide-item">

                <strong>
                  SENSOR HEALTH
                </strong>

                <span>
                  Telemetry-derived condition index from
                  0–100. It is not a failure probability.
                </span>

                <small>
                  HEALTHY ≥ 80 · WATCH 60–79.99 ·
                  DEGRADED 40–59.99 · CRITICAL &lt; 40
                </small>

              </div>

              <div className="diagnostic-guide-item">

                <strong>
                  MAINTENANCE PRIORITY
                </strong>

                <span>
                  Recommended action based on severity and
                  approximate subsystem condition.
                </span>

                <small>
                  ROUTINE · PLANNED · URGENT · IMMEDIATE
                </small>

              </div>

            </div>

          </div>

        </section>


        {/* ====================================================
            MODEL EXPLAINABILITY
        ===================================================== */}

        {activeDiagnostics && (
          <section className="panel">

            <div className="panel-header">

              <div>

                <span className="eyebrow">
                  EXPLAINABLE AI
                </span>

                <h2>
                  Model Diagnostics
                </h2>

                <p>
                  SHAP contributors explaining
                  the current failure-risk
                  prediction.
                </p>

              </div>

              <div className="feature-count">

                <strong>
                  {activeDiagnostics.feature_count}
                </strong>

                <span>
                  MODEL FEATURES
                </span>

              </div>

            </div>

            <div className="shap-header">

              <span>
                FEATURE
              </span>

              <span>
                DIRECTION
              </span>

              <span>
                IMPACT
              </span>

            </div>

            <div className="shap-list">

              {shapContributors.map(
                (item, index) => (
                  <div
                    className="shap-row"
                    key={`${item.feature}-${index}`}
                  >

                    <div className="shap-feature">
                      {item.feature}
                    </div>

                    <div
                      className={`shap-direction ${
                        item.direction ===
                        "INCREASES_RISK"
                          ? "risk-up"
                          : "risk-down"
                      }`}
                    >

                      <span>
                        {item.direction ===
                        "INCREASES_RISK"
                          ? "↑"
                          : "↓"}
                      </span>

                      <small>
                        {item.direction ===
                        "INCREASES_RISK"
                          ? "risk"
                          : "protective"}
                      </small>

                    </div>

                    <div className="shap-impact">

                      {item.impact > 0
                        ? "+"
                        : ""}

                      {Number(
                        item.impact
                      ).toFixed(4)}

                    </div>

                  </div>
                )
              )}

            </div>

          </section>
        )}

        {/* ====================================================
            MAINTENANCE
        ===================================================== */}

        {activeMaintenance && (
          <section className="panel maintenance-panel"id="maintenance">

            <div className="panel-header">

              <div>

                <span className="eyebrow">
                  DECISION SUPPORT
                </span>

                <h2>
                  Maintenance Recommendation
                </h2>

                <p>
                  Rule-based recommendation
                  generated from current
                  diagnostic state.
                </p>

              </div>

              <div
                className={`maintenance-priority ${
                  maintenancePriority.toLowerCase()
                }`}
              >
                {maintenancePriority}
              </div>

            </div>

            <div className="maintenance-content">

              <div className="maintenance-action">

                <div className="maintenance-action-icon">
                  →
                </div>

                <div>

                  <span className="detail-label">
                    RECOMMENDED ACTION
                  </span>

                  <strong>
                    {maintenanceAction}
                  </strong>

                </div>

              </div>

              <div className="maintenance-grid">

                <div>

                  <span className="detail-label">
                    RUL CONTEXT
                  </span>

                  <strong>
                    {maintenanceRulContext}
                  </strong>

                </div>

                <div>

                  <span className="detail-label">
                    FOCUS SUBSYSTEMS
                  </span>

                  <strong>

                    {focusSubsystems.length >
                    0
                      ? focusSubsystems
                          .map((item) => {
                            if (
                              typeof item ===
                              "string"
                            ) {
                              return item.replace(
                                "_",
                                " "
                              );
                            }

                            return item.subsystem;
                          })
                          .join(", ")
                      : "None"}

                  </strong>

                </div>

              </div>

              <div className="maintenance-note">

                <span className="note-mark">
                  i
                </span>

                <span>
                  Maintenance recommendations
                  are decision-support outputs
                  based on the project's
                  predictive models and
                  rule-based diagnostic logic.
                </span>

              </div>

            </div>

          </section>
        )}

        {/* ====================================================
            AI ASSISTANT
        ===================================================== */}

        <section className="panel assistant-panel" id="assistant">

          <div className="assistant-hero">

            <div>

              <span className="eyebrow">
                AI-ASSISTED DIAGNOSTICS
              </span>

              <h2>
                Vehicle Diagnostic Assistant
              </h2>

              <p>
                Ask questions about vehicle
                health, predicted RUL,
                failure risk, subsystem
                condition, diagnostics or
                maintenance.
              </p>

            </div>

            <div className="assistant-badge">

              <span className="assistant-dot" />

              RAG + GEMINI

            </div>

          </div>

          <div className="assistant-body">

            {/* ==================================================
                ASSISTANT CONTEXT
            =================================================== */}

            <div className="assistant-context">

              <div className="assistant-context-main">

                <span className="assistant-context-indicator">
                  <span className="status-dot" />
                </span>

                <div>

                  <span className="eyebrow">
                    ANALYSIS CONTEXT
                  </span>

                  <strong>
                    {activeDataSource.label}
                  </strong>

                  <p>
                    Unit{" "}
                    {activeVehicle.unit}
                    {" · "}
                    Cycle{" "}
                    {activeVehicle.current_cycle}
                  </p>

                </div>

              </div>

              <div className="assistant-context-source">

                <span>
                  SOURCE
                </span>

                <strong>
                  {activeDataSourceIsReplay
                    ? activeDataSource.filename
                    : "Built-in demonstration trajectory"}
                </strong>

              </div>

            </div>

            <form
              className="assistant-form"
              onSubmit={
                handleAssistantSubmit
              }
            >

              <label
                htmlFor="assistant-question"
                className="detail-label"
              >
                VEHICLE DIAGNOSTIC QUESTION
              </label>

              <textarea
                id="assistant-question"
                value={assistantQuestion}
                onChange={(event) =>
                  setAssistantQuestion(
                    event.target.value
                  )
                }
                placeholder="Example: Why is the vehicle health critical?"
                rows={4}
                maxLength={2000}
                disabled={
                  assistantLoading
                }
              />

              <div className="assistant-form-footer">

                <span className="assistant-hint">

                  {activeDataSource.label}
                  {" · "}
                  Unit{" "}
                  {activeVehicle?.unit ??
                    1}
                  {" · "}
                  Cycle{" "}
                  {activeVehicle?.current_cycle ??
                    "—"}

                </span>

                <button
                  type="submit"
                  className="primary-button assistant-submit"
                  disabled={
                    assistantLoading
                  }
                >
                  {assistantLoading
                    ? "Analyzing..."
                    : "Ask Assistant"}
                </button>

              </div>

            </form>

            {assistantError && (
              <div className="assistant-error">

                <strong>
                  Assistant Error
                </strong>

                <p>
                  {assistantError}
                </p>

              </div>
            )}

            {assistantLoading && (
              <div className="assistant-loading">

                <span className="loading-dot" />

                <span>
                  Analyzing vehicle
                  diagnostics and retrieving
                  relevant knowledge...
                </span>

              </div>
            )}

            {assistantAnswer &&
              !assistantLoading && (
                <div className="assistant-result">

                  <div className="assistant-answer-section">

                    <div className="assistant-result-heading">

                      <div>

                        <span className="eyebrow">
                          GENERATED ANALYSIS
                        </span>

                        <h3>
                          Diagnostic response
                        </h3>

                      </div>

                      <span className="grounded-pill">
                        GROUNDED
                      </span>

                    </div>

                    <div className="assistant-answer">

                      <ReactMarkdown>
                        {assistantAnswer}
                      </ReactMarkdown>

                    </div>

                  </div>

                  {assistantSources.length >
                    0 && (
                    <div className="assistant-sources">

                      <div className="assistant-sources-heading">

                        <div>

                          <span className="eyebrow">
                            RETRIEVAL TRACE
                          </span>

                          <h3>
                            Retrieved Knowledge
                          </h3>

                        </div>

                        <span>
                          {
                            assistantSources.length
                          }{" "}
                          sources
                        </span>

                      </div>

                      <div className="assistant-source-list">

                        {assistantSources.map(
                          (
                            source,
                            index
                          ) => (
                            <div
                              className="assistant-source"
                              key={`${source.source}-${source.chunk_index}-${index}`}
                            >

                              <div className="source-index">
                                {String(
                                  index + 1
                                ).padStart(
                                  2,
                                  "0"
                                )}
                              </div>

                              <div className="source-main">

                                <div className="assistant-source-top">

                                  <strong>
                                    {
                                      source.source
                                    }
                                  </strong>

                                  <span>
                                    Distance{" "}
                                    {
                                      source.distance
                                    }
                                  </span>

                                </div>

                                <div className="assistant-source-meta">
                                  Knowledge
                                  chunk{" "}
                                  {
                                    source.chunk_index
                                  }
                                </div>

                              </div>

                            </div>
                          )
                        )}

                      </div>

                    </div>
                  )}

                </div>
              )}

          </div>

        </section>

      </main>

      {/* ======================================================
          FOOTER
      ======================================================= */}

      <footer className="site-footer">

        <div>

          <strong>
            Vehicle Predictive Maintenance
          </strong>

          <span>
            Diagnostics Platform
          </span>

        </div>

        <div>

          NASA C-MAPSS benchmark data

          <span className="footer-divider">
            •
          </span>

          Simulated predictive-maintenance
          research

        </div>

      </footer>

    </div>
  );
}

export default App;
