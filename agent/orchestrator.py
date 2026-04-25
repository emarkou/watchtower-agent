from __future__ import annotations

import os
import time
from typing import Optional

import anthropic

from agent.run_store import RunStore
from drift.report import DriftReport
from training.config import RetrainingConfig, MLflowRunRef
from evaluation.summary import EvalReport


SYSTEM_PROMPT = (
    "You are a senior MLOps engineer responsible for deciding whether to retrain a production ML model.\n"
    "You have access to tools for drift detection, retraining, evaluation, and model promotion.\n"
    "\n"
    "Decision principles:\n"
    "- Be sceptical of unnecessary retrains. Retraining has a cost; only trigger it when drift is meaningful.\n"
    "- Consider both statistical significance AND practical significance of drift.\n"
    "- Prefer keeping the current champion unless the new model shows clear improvement (> min threshold).\n"
    "- Document ALL reasoning in your notes field — future auditors will read this.\n"
    "\n"
    "You MUST call write_summary as the final step of every run, even if no retrain occurred.\n"
    "You MUST NOT take any action not covered by your available tools."
)

TOOLS = [
    {
        "name": "run_statistical_drift",
        "description": "Runs statistical drift tests (KS, PSI, chi-squared) on all features and records results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The current run ID."},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "run_feature_drift",
        "description": "Runs Evidently feature drift detection across all features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The current run ID."},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "run_target_drift",
        "description": "Runs target/label drift detection to identify shifts in the output distribution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The current run ID."},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "trigger_retrain",
        "description": "Triggers a model retraining DAG via the Airflow REST API.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dag_id": {"type": "string", "description": "Airflow DAG ID to trigger."},
                "notes": {"type": "string", "description": "Human-readable notes for auditors explaining why retraining was triggered."},
                "feature_subset": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Optional subset of features to use for retraining.",
                },
                "class_weights": {
                    "type": ["object", "null"],
                    "description": "Optional class weights dict.",
                },
                "random_seed": {"type": "integer", "description": "Random seed for reproducibility.", "default": 42},
                "extra_conf": {"type": "object", "description": "Any extra configuration to pass to the DAG."},
            },
            "required": ["dag_id", "notes", "random_seed", "extra_conf"],
        },
    },
    {
        "name": "evaluate_model",
        "description": "Polls MLflow until the training run completes, then evaluates the new model against the eval set.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dag_run_id": {"type": "string", "description": "Airflow DAG run ID returned by trigger_retrain."},
                "mlflow_run_url": {"type": "string", "description": "MLflow run URL returned by trigger_retrain."},
            },
            "required": ["dag_run_id", "mlflow_run_url"],
        },
    },
    {
        "name": "promote_model",
        "description": "Registers the evaluated model in the MLflow registry and promotes it to Production.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dag_run_id": {"type": "string", "description": "Airflow DAG run ID."},
                "mlflow_run_url": {"type": "string", "description": "MLflow run URL of the model to promote."},
                "justification": {"type": "string", "description": "Explanation of why this model should replace the current champion."},
            },
            "required": ["dag_run_id", "mlflow_run_url", "justification"],
        },
    },
    {
        "name": "write_summary",
        "description": "Writes the final LLM-generated summary to RunStore. MUST be the last tool call in every run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Full reasoning and summary of decisions made during this run."},
                "retrain_triggered": {"type": "boolean", "description": "Whether retraining was triggered."},
                "champion_promoted": {"type": "boolean", "description": "Whether a new champion was promoted."},
            },
            "required": ["reasoning", "retrain_triggered", "champion_promoted"],
        },
    },
]


class AgentTimeoutError(Exception):
    pass


class Orchestrator:
    def __init__(
        self,
        store: RunStore,
        X_ref,
        y_ref,
        X_cur,
        y_cur,
        task_type: str,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.store = store
        self.X_ref = X_ref
        self.y_ref = y_ref
        self.X_cur = X_cur
        self.y_cur = y_cur
        self.task_type = task_type
        self.max_turns = int(os.getenv("AGENT_MAX_TURNS", "20"))

    def _eval_split(self):
        n = len(self.X_cur)
        split = int(n * 0.8)
        X_eval = self.X_cur.iloc[split:]
        y_eval = self.y_cur.iloc[split:]
        return X_eval, y_eval

    def run(self, drift_report: DriftReport, run_id: str) -> Optional[EvalReport]:

        feature_names = list(self.X_ref.columns)
        X_eval, y_eval = self._eval_split()

        messages = [
            {
                "role": "user",
                "content": (
                    f"Here is the drift report for run {run_id}:\n\n"
                    f"{drift_report.to_json()}\n\n"
                    "Please analyse the drift and decide whether to retrain."
                ),
            }
        ]

        step_index = 0
        retrain_triggered = False
        champion_promoted = False
        eval_report: Optional[EvalReport] = None
        mlflow_run_ref: Optional[MLflowRunRef] = None

        for _ in range(self.max_turns):
            response = self.client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            for block in response.content:
                if block.type == "text":
                    self.store.record_step(
                        run_id=run_id,
                        step_index=step_index,
                        step_type="llm_reasoning",
                        tool_name=None,
                        input_payload=None,
                        output_payload={"text": block.text},
                    )
                    step_index += 1

            if response.stop_reason == "end_turn":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input
                start_ms = time.time()

                self.store.record_step(
                    run_id=run_id,
                    step_index=step_index,
                    step_type="tool_call",
                    tool_name=tool_name,
                    input_payload=tool_input,
                    output_payload=None,
                )
                step_index += 1

                try:
                    result = self._dispatch_tool(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        run_id=run_id,
                        mlflow_run_ref=mlflow_run_ref,
                        retrain_triggered=retrain_triggered,
                        champion_promoted=champion_promoted,
                        eval_report=eval_report,
                        feature_names=feature_names,
                        X_eval=X_eval,
                        y_eval=y_eval,
                    )

                    if tool_name == "trigger_retrain":
                        retrain_triggered = True
                        mlflow_run_ref = MLflowRunRef(
                            dag_run_id=result["dag_run_id"],
                            mlflow_run_url=result["mlflow_run_url"],
                        )
                    elif tool_name == "evaluate_model":
                        eval_report = EvalReport(**result)
                    elif tool_name == "promote_model":
                        champion_promoted = result.get("promoted", False)

                    duration_ms = int((time.time() - start_ms) * 1000)
                    self.store.record_step(
                        run_id=run_id,
                        step_index=step_index,
                        step_type="tool_result",
                        tool_name=tool_name,
                        input_payload=None,
                        output_payload=result,
                        duration_ms=duration_ms,
                    )
                    step_index += 1

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

                    if tool_name == "write_summary":
                        return eval_report

                except Exception as e:
                    error_result = {"error": str(e), "tool": tool_name}
                    self.store.record_step(
                        run_id=run_id,
                        step_index=step_index,
                        step_type="tool_result",
                        tool_name=tool_name,
                        input_payload=None,
                        output_payload=error_result,
                    )
                    step_index += 1
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {e}",
                        "is_error": True,
                    })

            messages.append({"role": "assistant", "content": response.content})
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        raise AgentTimeoutError(f"Agent exceeded {self.max_turns} turns without completing")

    def _dispatch_tool(
        self,
        tool_name: str,
        tool_input: dict,
        run_id: str,
        mlflow_run_ref: Optional[MLflowRunRef],
        retrain_triggered: bool,
        champion_promoted: bool,
        eval_report: Optional[EvalReport],
        feature_names: list,
        X_eval,
        y_eval,
    ) -> dict:
        from agent.tools.drift_tools import (
            run_statistical_drift_tool,
            run_feature_drift_tool,
            run_target_drift_tool,
        )
        from agent.tools.retrain_tools import trigger_retrain_tool
        from agent.tools.eval_tools import (
            evaluate_model_tool,
            promote_model_tool,
            write_summary_tool,
        )

        if tool_name == "run_statistical_drift":
            return run_statistical_drift_tool(
                run_id=run_id,
                store=self.store,
                X_ref=self.X_ref,
                X_cur=self.X_cur,
                feature_names=feature_names,
                task_type=self.task_type,
            )

        if tool_name == "run_feature_drift":
            return run_feature_drift_tool(
                run_id=run_id,
                store=self.store,
                X_ref=self.X_ref,
                X_cur=self.X_cur,
            )

        if tool_name == "run_target_drift":
            return run_target_drift_tool(
                run_id=run_id,
                store=self.store,
                y_ref=self.y_ref,
                y_cur=self.y_cur,
                task_type=self.task_type,
            )

        if tool_name == "trigger_retrain":
            config = RetrainingConfig(
                dag_id=tool_input["dag_id"],
                notes=tool_input.get("notes", ""),
                feature_subset=tool_input.get("feature_subset"),
                class_weights=tool_input.get("class_weights"),
                random_seed=tool_input.get("random_seed", 42),
                extra_conf=tool_input.get("extra_conf", {}),
            )
            return trigger_retrain_tool(config=config, run_id=run_id, store=self.store)

        if tool_name == "evaluate_model":
            ref = MLflowRunRef(
                dag_run_id=tool_input["dag_run_id"],
                mlflow_run_url=tool_input["mlflow_run_url"],
            )
            return evaluate_model_tool(
                mlflow_run_ref=ref,
                store=self.store,
                run_id=run_id,
                X_eval=X_eval,
                y_eval=y_eval,
                task_type=self.task_type,
            )

        if tool_name == "promote_model":
            if mlflow_run_ref is None:
                raise RuntimeError("No mlflow_run_ref available; trigger_retrain must be called first.")
            ref = MLflowRunRef(
                dag_run_id=tool_input["dag_run_id"],
                mlflow_run_url=tool_input["mlflow_run_url"],
                mlflow_run_id=mlflow_run_ref.mlflow_run_id,
                artifact_uri=mlflow_run_ref.artifact_uri,
            )
            if eval_report is None:
                raise RuntimeError("evaluate_model must be called before promote_model.")
            return promote_model_tool(
                mlflow_run_ref=ref,
                eval_report=eval_report,
                justification=tool_input["justification"],
                store=self.store,
                run_id=run_id,
            )

        if tool_name == "write_summary":
            return write_summary_tool(
                eval_report=eval_report,
                reasoning=tool_input["reasoning"],
                store=self.store,
                run_id=run_id,
                retrain_triggered=retrain_triggered,
                champion_promoted=champion_promoted,
            )

        raise ValueError(f"Unknown tool: {tool_name}")
