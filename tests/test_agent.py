from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from agent.orchestrator import Orchestrator, AgentTimeoutError
from drift.report import DriftReport, StatisticalDriftResult, StatisticalTestResult, EvidentlyFeatureDrift, EvidentlyTargetDrift


@pytest.fixture
def drift_report(run_id):
    stat = StatisticalDriftResult(
        results=[
            StatisticalTestResult("feature_0", "ks", 0.45, 0.001, True, 0.8),
            StatisticalTestResult("feature_1", "psi", 0.25, None, True, 0.7),
        ],
        overall_drift_detected=True,
        drifted_feature_names=["feature_0", "feature_1"],
        n_drifted=2,
        n_features=10,
    )
    feat = EvidentlyFeatureDrift(
        dataset_drift=True,
        share_of_drifted_features=0.3,
        per_feature_scores={},
        per_feature_drift={},
    )
    tgt = EvidentlyTargetDrift(
        target_drift_detected=False,
        kl_divergence=0.02,
        mean_shift_sigma=None,
        statistic=None,
        p_value=None,
    )
    return DriftReport(
        run_id=run_id,
        timestamp="2026-01-01T00:00:00",
        statistical=stat,
        feature_drift=feat,
        target_drift=tgt,
        overall_severity=0.4,
        drifted_feature_names=["feature_0", "feature_1"],
        n_reference_samples=1000,
        n_current_samples=500,
    )


def _make_anthropic_response(tool_name: str, tool_input: dict, stop_reason: str = "tool_use"):
    """Build a mock Anthropic API response with a single tool_use block."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = f"Analysing and calling {tool_name}."

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = f"tool_{tool_name}_1"
    tool_block.name = tool_name
    tool_block.input = tool_input

    response = MagicMock()
    response.content = [text_block, tool_block]
    response.stop_reason = stop_reason
    return response


def _make_write_summary_response():
    """Final response that calls write_summary."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "No significant drift requiring retrain. Writing summary."

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_write_summary_1"
    tool_block.name = "write_summary"
    tool_block.input = {
        "reasoning": "Drift is minor. No retrain needed.",
        "retrain_triggered": False,
        "champion_promoted": False,
    }

    response = MagicMock()
    response.content = [text_block, tool_block]
    response.stop_reason = "tool_use"
    return response


def test_orchestrator_no_retrain_path(store, run_id, drift_report, loader):
    """Orchestrator completes without retraining when agent calls write_summary directly."""
    store.create_run(run_id, "manual_api")

    X_ref, y_ref = loader.load_reference()
    X_cur, y_cur = loader.load_current_window()

    with patch("anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        # Agent calls write_summary immediately (no retrain)
        mock_client.messages.create.return_value = _make_write_summary_response()

        orchestrator = Orchestrator(store, X_ref, y_ref, X_cur, y_cur, "classification")
        result = orchestrator.run(drift_report, run_id)

    assert result is None  # no retrain → no EvalReport
    steps = store.get_agent_steps(run_id)
    assert len(steps) > 0
    step_types = [s["step_type"] for s in steps]
    assert "llm_reasoning" in step_types
    assert "tool_call" in step_types


def test_orchestrator_records_reasoning_before_tool_calls(store, run_id, drift_report, loader):
    """Text blocks are captured as llm_reasoning before tool_use blocks are processed."""
    store.create_run(run_id, "manual_api")

    X_ref, y_ref = loader.load_reference()
    X_cur, y_cur = loader.load_current_window()

    with patch("anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value = _make_write_summary_response()

        orchestrator = Orchestrator(store, X_ref, y_ref, X_cur, y_cur, "classification")
        orchestrator.run(drift_report, run_id)

    steps = store.get_agent_steps(run_id)
    reasoning_steps = [s for s in steps if s["step_type"] == "llm_reasoning"]
    tool_call_steps = [s for s in steps if s["step_type"] == "tool_call"]
    assert len(reasoning_steps) > 0
    # Reasoning step_index must be lower than the tool call step_index that follows
    if tool_call_steps:
        assert reasoning_steps[0]["step_index"] < tool_call_steps[0]["step_index"]


def test_orchestrator_timeout(store, run_id, drift_report, loader):
    """AgentTimeoutError raised when max turns exceeded without write_summary."""
    store.create_run(run_id, "manual_api")

    X_ref, y_ref = loader.load_reference()
    X_cur, y_cur = loader.load_current_window()

    # Response that never calls write_summary, just keeps calling stat drift
    never_ending = _make_anthropic_response("run_statistical_drift", {"run_id": run_id})

    with patch("anthropic.Anthropic") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.messages.create.return_value = never_ending

        orchestrator = Orchestrator(store, X_ref, y_ref, X_cur, y_cur, "classification")
        orchestrator.max_turns = 2  # force quick timeout

        with pytest.raises(AgentTimeoutError):
            orchestrator.run(drift_report, run_id)
