"""
Tests for the natural-language-to-parameters benchmark and its scoring.
"""

from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart, RetryPromptPart
from pydantic_ai.models.function import FunctionModel

from deeplense_agent.evaluation.benchmark import (
    BENCHMARK_CASES,
    BenchmarkCase,
    Prediction,
    llm_planner,
    rule_based_planner,
    run_benchmark,
    score_case,
)


def test_scoring_counts_missed_and_spurious_fields():
    case = BenchmarkCase("x", {"substructure_type": "cdm", "z_lens": 0.3})
    r = score_case(case, Prediction(args={"substructure_type": "cdm", "z_lens": 0.4, "num_pix": 64}, plan=None, executed=False))
    assert r.correct == ["substructure_type"]
    assert r.missed == ["z_lens"]
    assert set(r.spurious) == {"z_lens", "num_pix"}
    assert not r.exact


def test_scoring_ignores_explicit_tool_defaults():
    case = BenchmarkCase("A strong lens", {})
    r = score_case(case, Prediction(args={"model_configuration": "Model_II", "num_images": 1}, plan=None, executed=False))
    assert r.exact


def test_offline_benchmark_keeps_safety_properties():
    s = run_benchmark(rule_based_planner)
    assert s["cases"] == len(BENCHMARK_CASES)
    assert s["field_precision"] == 1.0
    n_guard = sum(c.forbidden is not None for c in BENCHMARK_CASES)
    assert s["guardrail_held"] == f"{n_guard}/{n_guard}"
    flagged, total = map(int, s["defaults_flagged"].split("/"))
    assert flagged == total > 0
    assert s["executed_without_approval"] == 0


def _echo_llm(messages, info):
    """Passes the benchmark's expected arguments straight to the planning tool, then requests a run."""
    last = messages[-1].parts
    returned = {p.tool_name for p in last if isinstance(p, ToolReturnPart)}
    if "tool_plan_simulation" in returned or any(isinstance(p, RetryPromptPart) for p in last):
        return ModelResponse(parts=[ToolCallPart("tool_execute_simulation", {})])
    prompt = next(p.content for p in last if p.part_kind == "user-prompt")
    case = next(c for c in BENCHMARK_CASES if c.prompt == prompt)
    return ModelResponse(parts=[ToolCallPart("tool_plan_simulation", case.expected)])


def test_llm_planner_scores_a_perfect_extractor_and_never_runs_unapproved():
    s = run_benchmark(llm_planner(FunctionModel(_echo_llm)))
    assert s["exact_match"] == 1.0
    assert s["executed_without_approval"] == 0
    n_guard = sum(c.forbidden is not None for c in BENCHMARK_CASES)
    assert s["guardrail_held"] == f"{n_guard}/{n_guard}"
