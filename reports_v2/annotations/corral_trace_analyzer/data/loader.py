"""
Data loader for trace annotations
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from .schema import StepData, ToolCallData, TraceData


class TraceDataLoader:
    """Loads and parses trace annotation data from parquet files"""

    def __init__(self, parquet_path: str):
        """
        Initialize the data loader

        Args:
            parquet_path: Path to the parquet file containing trace annotations
        """
        self.parquet_path = Path(parquet_path)
        self.raw_df = None
        self.traces_df = None
        self.steps_df = None
        self.tools_df = None

    def load(self) -> pd.DataFrame:
        """
        Load the parquet file

        Returns:
            Raw DataFrame from parquet
        """
        logger.info(f"Loading data from {self.parquet_path}...")
        self.raw_df = pd.read_parquet(self.parquet_path)
        logger.info(f"Loaded {len(self.raw_df)} traces")
        return self.raw_df

    def _parse_json_field(self, value: Any) -> Any:
        """
        Parse a field that might be a JSON string

        Args:
            value: Value that might be a JSON string or already parsed

        Returns:
            Parsed value
        """
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
        return value

    def parse_trace(self, row: pd.Series) -> TraceData:
        """
        Parse a single trace row into a TraceData object

        Args:
            row: DataFrame row containing trace data

        Returns:
            TraceData object
        """
        # Extract trial info - parse JSON if needed
        trial_info = self._parse_json_field(row.get("trial_info", {}))
        if not isinstance(trial_info, dict):
            trial_info = {}

        token_usage = self._parse_json_field(trial_info.get("token_usage", {}))
        if not isinstance(token_usage, dict):
            token_usage = {}

        error_types = self._parse_json_field(trial_info.get("error_types", {}))
        if not isinstance(error_types, dict):
            error_types = {}

        # Parse steps
        steps = []
        file_nodes = self._parse_json_field(row.get("fileNodes", []))
        if not isinstance(file_nodes, list):
            file_nodes = []
        for idx, node in enumerate(file_nodes):
            # Ensure markers is a list
            markers = node.get("markers", [])
            if not isinstance(markers, list):
                markers = []

            step = StepData(
                trace_id=row["fileId"],
                step_id=node.get("id", str(idx)),
                step_index=idx,
                node_type=node.get("type", ""),
                annotatable=node.get("annotatable", False),
                message=node.get("message") or "",  # Handle None
                markers=markers,
                notes=node.get("notes") or "",  # Handle None
            )
            steps.append(step)

        # Parse tool calls
        tool_calls = []
        tool_calls_data = self._parse_json_field(trial_info.get("tool_calls", []))
        if not isinstance(tool_calls_data, list):
            tool_calls_data = []

        for tool_call in tool_calls_data:
            tc = ToolCallData(
                tool_name=tool_call.get("tool_name", ""),
                arguments=tool_call.get("arguments", {}),
                result=tool_call.get("result", ""),
                status=tool_call.get("status", ""),
                error_message=tool_call.get("error_message"),
                duration=tool_call.get("duration", 0.0),
                timestamp=tool_call.get("timestamp", ""),
            )
            tool_calls.append(tc)

        # Create TraceData object
        trace = TraceData(
            trace_id=row["fileId"],
            file_id=row["fileId"],
            annotator=row.get("annotator", ""),
            model=row.get("model", ""),
            environment=row.get("environment", ""),
            agent_type=row.get("agent_type", ""),
            level=row.get("level", 0),
            task_id=row.get("task_id", ""),
            timestamp=row.get("timestamp", ""),
            trial_id=trial_info.get("trial_id", ""),
            score=trial_info.get("score", 0.0),
            success=trial_info.get("success", False),
            submitted_answer=trial_info.get("submitted_answer", ""),
            tool_execution_duration=trial_info.get("tool_execution_duration", 0.0),
            prompt_tokens=token_usage.get("prompt_tokens", 0),
            completion_tokens=token_usage.get("completion_tokens", 0),
            total_tokens=token_usage.get("total_tokens", 0),
            total_calls=trial_info.get("total_calls", 0),
            successful_calls=trial_info.get("successful_calls", 0),
            failed_calls=trial_info.get("failed_calls", 0),
            tools_used=trial_info.get("tools_used", [])
            if isinstance(trial_info.get("tools_used", []), list)
            else [],
            invalid_tool_errors=error_types.get("invalid_tool", 0),
            invalid_args_errors=error_types.get("invalid_args", 0),
            execution_errors=error_types.get("execution_error", 0),
            positive_marker_count=row.get("positive_marker_count", 0),
            negative_marker_count=row.get("negative_marker_count", 0),
            neutral_marker_count=row.get("neutral_marker_count", 0),
            validation_attempt_count=row.get("validation_attempt_count", 0),
            backtrack_trigger_count=row.get("backtrack_trigger_count", 0),
            planning_statement_count=row.get("planning_statement_count", 0),
            reasoning_statement_count=row.get("reasoning_statement_count", 0),
            correct_submission_count=row.get("correct_submission_count", 0),
            neutral_count=row.get("neutral_count", 0),
            iteration_limit_count=row.get("iteration_limit_count", 0),
            missing_validation_count=row.get("missing_validation_count", 0),
            unnecessary_tool_use_count=row.get("unnecessary_tool_use_count", 0),
            non_sense_count=row.get("non_sense_count", 0),
            loop_instance_count=row.get("loop_instance_count", 0),
            hallucination_count=row.get("hallucination_count", 0),
            wrong_planning_count=row.get("wrong_planning_count", 0),
            wrong_reasoning_count=row.get("wrong_reasoning_count", 0),
            syntax_error_count=row.get("syntax_error_count", 0),
            early_final_answer_count=row.get("early_final_answer_count", 0),
            give_up_count=row.get("give_up_count", 0),
            inefficient_tool_call_count=row.get("inefficient_tool_call_count", 0),
            misunderstood_tool_count=row.get("misunderstood_tool_count", 0),
            qa_score=row.get("qa_score", 0.0),
            steps=steps,
            tool_calls=tool_calls,
        )

        return trace

    def create_traces_dataframe(self) -> pd.DataFrame:
        """
        Create a flat dataframe with one row per trace

        Returns:
            DataFrame with trace-level data
        """
        if self.raw_df is None:
            self.load()

        logger.info("Creating traces dataframe...")
        traces_data = []

        for idx, row in self.raw_df.iterrows():
            trace = self.parse_trace(row)

            trace_dict = {
                "trace_id": trace.trace_id,
                "file_id": trace.file_id,
                "annotator": trace.annotator,
                "model": trace.model,
                "environment": trace.environment,
                "agent_type": trace.agent_type,
                "level": trace.level,
                "task_id": trace.task_id,
                "timestamp": trace.timestamp,
                "trial_id": trace.trial_id,
                "score": trace.score,
                "success": trace.success,
                "tool_execution_duration": trace.tool_execution_duration,
                "prompt_tokens": trace.prompt_tokens,
                "completion_tokens": trace.completion_tokens,
                "total_tokens": trace.total_tokens,
                "total_calls": trace.total_calls,
                "successful_calls": trace.successful_calls,
                "failed_calls": trace.failed_calls,
                "tools_used_list": trace.tools_used if trace.tools_used else [],
                "tools_used_count": len(trace.tools_used) if trace.tools_used else 0,
                "invalid_tool_errors": trace.invalid_tool_errors,
                "invalid_args_errors": trace.invalid_args_errors,
                "execution_errors": trace.execution_errors,
                "total_errors": trace.invalid_tool_errors
                + trace.invalid_args_errors
                + trace.execution_errors,
                "positive_marker_count": trace.positive_marker_count,
                "negative_marker_count": trace.negative_marker_count,
                "neutral_marker_count": trace.neutral_marker_count,
                "validation_attempt_count": trace.validation_attempt_count,
                "backtrack_trigger_count": trace.backtrack_trigger_count,
                "planning_statement_count": trace.planning_statement_count,
                "reasoning_statement_count": trace.reasoning_statement_count,
                "correct_submission_count": trace.correct_submission_count,
                "neutral_count": trace.neutral_count,
                "iteration_limit_count": trace.iteration_limit_count,
                "missing_validation_count": trace.missing_validation_count,
                "unnecessary_tool_use_count": trace.unnecessary_tool_use_count,
                "non_sense_count": trace.non_sense_count,
                "loop_instance_count": trace.loop_instance_count,
                "hallucination_count": trace.hallucination_count,
                "wrong_planning_count": trace.wrong_planning_count,
                "wrong_reasoning_count": trace.wrong_reasoning_count,
                "syntax_error_count": trace.syntax_error_count,
                "early_final_answer_count": trace.early_final_answer_count,
                "give_up_count": trace.give_up_count,
                "inefficient_tool_call_count": trace.inefficient_tool_call_count,
                "misunderstood_tool_count": trace.misunderstood_tool_count,
                "qa_score": trace.qa_score,
                "step_count": len(trace.steps),
            }

            traces_data.append(trace_dict)

        self.traces_df = pd.DataFrame(traces_data)
        logger.info(f"Created traces dataframe with {len(self.traces_df)} rows")
        return self.traces_df

    def create_steps_dataframe(self) -> pd.DataFrame:
        """
        Create a flat dataframe with one row per step

        Returns:
            DataFrame with step-level data
        """
        if self.raw_df is None:
            self.load()

        logger.info("Creating steps dataframe...")
        steps_data = []

        for idx, row in self.raw_df.iterrows():
            trace = self.parse_trace(row)

            for step in trace.steps:
                # Handle None values
                message = step.message if step.message is not None else ""
                markers = step.markers if step.markers is not None else []

                step_dict = {
                    "trace_id": trace.trace_id,
                    "step_id": step.step_id,
                    "step_index": step.step_index,
                    "node_type": step.node_type,
                    "annotatable": step.annotatable,
                    "message": message,
                    "message_length": len(message),
                    "markers": markers,
                    "marker_count": len(markers),
                    "has_positive": "positive" in markers,
                    "has_negative": "negative" in markers,
                    "has_neutral": "neutral" in markers,
                    "has_planning": "planning_statement" in markers,
                    "has_reasoning": "reasoning_statement" in markers,
                    "notes": step.notes if step.notes is not None else "",
                    # Add metadata from trace
                    "model": trace.model,
                    "environment": trace.environment,
                    "agent_type": trace.agent_type,
                    "score": trace.score,
                }

                steps_data.append(step_dict)

        self.steps_df = pd.DataFrame(steps_data)
        logger.info(f"Created steps dataframe with {len(self.steps_df)} rows")
        return self.steps_df

    def create_tools_dataframe(self) -> pd.DataFrame:
        """
        Create a flat dataframe with one row per tool call

        Returns:
            DataFrame with tool-level data
        """
        if self.raw_df is None:
            self.load()

        logger.info("Creating tools dataframe...")
        tools_data = []

        for idx, row in self.raw_df.iterrows():
            trace = self.parse_trace(row)

            for tool_idx, tool in enumerate(trace.tool_calls):
                tool_dict = {
                    "trace_id": trace.trace_id,
                    "tool_index": tool_idx,
                    "tool_name": tool.tool_name,
                    "status": tool.status,
                    "is_success": tool.status == "success",
                    "has_error": tool.error_message is not None,
                    "error_message": tool.error_message,
                    "duration": tool.duration,
                    "timestamp": tool.timestamp,
                    # Add metadata from trace
                    "model": trace.model,
                    "environment": trace.environment,
                    "agent_type": trace.agent_type,
                    "score": trace.score,
                }

                tools_data.append(tool_dict)

        self.tools_df = pd.DataFrame(tools_data)
        logger.info(f"Created tools dataframe with {len(self.tools_df)} rows")
        return self.tools_df

    def load_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all dataframes

        Returns:
            tuple of (traces_df, steps_df, tools_df)
        """
        self.load()
        self.create_traces_dataframe()
        self.create_steps_dataframe()
        self.create_tools_dataframe()

        return self.traces_df, self.steps_df, self.tools_df

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the loaded data

        Returns:
            dictionary with summary statistics
        """
        if self.traces_df is None:
            self.load_all()

        return {
            "total_traces": len(self.traces_df),
            "total_steps": len(self.steps_df),
            "total_tool_calls": len(self.tools_df),
            "unique_models": self.traces_df["model"].nunique(),
            "unique_environments": self.traces_df["environment"].nunique(),
            "unique_agent_types": self.traces_df["agent_type"].nunique(),
            "models": self.traces_df["model"].unique().tolist(),
            "environments": self.traces_df["environment"].unique().tolist(),
            "agent_types": self.traces_df["agent_type"].unique().tolist(),
            "success_rate": self.traces_df["success"].mean(),
            "avg_score": self.traces_df["score"].mean(),
            "avg_steps_per_trace": len(self.steps_df) / len(self.traces_df),
            "avg_tools_per_trace": len(self.tools_df) / len(self.traces_df),
        }
