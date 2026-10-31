"""
Natural-language-to-parameters benchmark for the DeepLense agent.

Each case is a user prompt with the planning-tool arguments a careful reader would extract
from it. A planner (the offline rule-based parser, or any pydantic-ai LLM driving the agent)
is scored on:

- field precision / recall / F1 over the stated parameters, and per-prompt exact match;
- guardrails: unphysical requests must never produce a stored plan with those values;
- HITL: omitted redshifts must come back flagged as clarifications;
- approval gate: nothing may execute during planning.

Run:  python -m deeplense_agent.evaluation [--model groq:openai/gpt-oss-120b]
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from pydantic_ai.messages import ModelResponse, ToolCallPart

from deeplense_agent.agent.agent import create_pydantic_ai_agent
from deeplense_agent.agent.tools import SimulationSessionContext, plan_simulation
from deeplense_agent.agent.workflow import DeepLenseWorkflowSession, _run_blocking
from deeplense_agent.models.responses import SimulationPlan


@dataclass(frozen=True)
class BenchmarkCase:
    prompt: str
    expected: Dict[str, Any]
    # Set for unphysical requests: values that must never end up in a stored plan
    forbidden: Optional[Dict[str, Any]] = None


BENCHMARK_CASES: List[BenchmarkCase] = [
    BenchmarkCase("Simulate a smooth lens on Euclid", {"substructure_type": "no_sub", "model_configuration": "Model_II"}),
    BenchmarkCase("CDM subhalos with Hubble, z_lens=0.3 and z_source=1.8",
                  {"substructure_type": "cdm", "model_configuration": "Model_III", "z_lens": 0.3, "z_source": 1.8}),
    BenchmarkCase("Axion dark matter vortex lens, Model IV", {"substructure_type": "axion", "model_configuration": "Model_IV"}),
    BenchmarkCase("Generate 10 images of a lens with 60 subhalos using Model I",
                  {"num_images": 10, "substructure_type": "cdm", "num_subhalos": 60, "model_configuration": "Model_I"}),
    BenchmarkCase("Fuzzy dark matter lens with axion mass 1e-23 eV observed by HST",
                  {"substructure_type": "axion", "axion_mass_eV": 1e-23, "model_configuration": "Model_III"}),
    BenchmarkCase("Three images of a smooth lens at redshifts 0.4 and 2.0",
                  {"num_images": 3, "substructure_type": "no_sub", "z_lens": 0.4, "z_source": 2.0}),
    BenchmarkCase("I want a lens with no substructure, source at z_s = 1.5", {"substructure_type": "no_sub", "z_source": 1.5}),
    BenchmarkCase("Euclid lens with subhalos, mass function slope 1.9, seed 7",
                  {"model_configuration": "Model_II", "substructure_type": "cdm", "subhalo_mass_slope": 1.9, "random_seed": 7}),
    BenchmarkCase("Lens galaxy with velocity dispersion 250 km/s and a CDM population",
                  {"substructure_type": "cdm", "velocity_dispersion_kms": 250}),
    BenchmarkCase("Faint source of magnitude 24 behind a smooth Euclid lens",
                  {"substructure_type": "no_sub", "model_configuration": "Model_II", "source_magnitude": 24}),
    BenchmarkCase("A 128x128 axion lens on Model I", {"num_pix": 128, "substructure_type": "axion", "model_configuration": "Model_I"}),
    BenchmarkCase("Simulate 5 strong lenses with cold dark matter substructure for the Hubble Space Telescope",
                  {"num_images": 5, "substructure_type": "cdm", "model_configuration": "Model_III"}),
    BenchmarkCase("Model 2, lens redshift 0.6, source redshift 1.4, wave dark matter",
                  {"model_configuration": "Model_II", "z_lens": 0.6, "z_source": 1.4, "substructure_type": "axion"}),
    BenchmarkCase("Give me a ground-based image of a lens with 40 subhalos",
                  {"model_configuration": "Model_I", "substructure_type": "cdm", "num_subhalos": 40}),
    BenchmarkCase("Smooth lens with an Einstein radius of 1.2 arcsec, HST",
                  {"substructure_type": "no_sub", "einstein_radius_arcsec": 1.2, "model_configuration": "Model_III"}),
    BenchmarkCase("Axion lens with 20 images, seed 123", {"substructure_type": "axion", "num_images": 20, "random_seed": 123}),
    BenchmarkCase("Euclid, zl=0.35, zs=2.2, no subhalos",
                  {"model_configuration": "Model_II", "z_lens": 0.35, "z_source": 2.2, "substructure_type": "no_sub"}),
    BenchmarkCase("Simulate a lens with a halo mass of 5e12 solar masses", {"lens_mass_solar": 5e12}),
    BenchmarkCase("Model III image with lens light included and CDM substructure",
                  {"model_configuration": "Model_III", "include_lens_light": True, "substructure_type": "cdm"}),
    BenchmarkCase("A strong lens", {}),
    BenchmarkCase("Model_IV lens without substructure, 4 images",
                  {"model_configuration": "Model_IV", "substructure_type": "no_sub", "num_images": 4}),
    BenchmarkCase("CDM lens on Euclid, source magnitude 22, 256 pixels across",
                  {"substructure_type": "cdm", "model_configuration": "Model_II", "source_magnitude": 22, "num_pix": 256}),
    # Unphysical requests: the guardrails must stop them
    BenchmarkCase("CDM lens with z_lens=2.0 and z_source=1.0",
                  {"substructure_type": "cdm", "z_lens": 2.0, "z_source": 1.0}, forbidden={"z_lens": 2.0, "z_source": 1.0}),
    BenchmarkCase("Put the lens at redshift 1.5 and the source at redshift 0.8",
                  {"z_lens": 1.5, "z_source": 0.8}, forbidden={"z_lens": 1.5, "z_source": 0.8}),
    BenchmarkCase("Generate 500 images with Euclid", {"num_images": 500, "model_configuration": "Model_II"}, forbidden={"num_images": 500}),
    BenchmarkCase("Smooth lens with z_lens = -0.3", {"substructure_type": "no_sub", "z_lens": -0.3}, forbidden={"z_lens": -0.3}),
]

# Planning-tool defaults: passing one of these without the user asking is not an error
TOOL_DEFAULTS = {"substructure_type": "no_sub", "model_configuration": "Model_II", "num_images": 1}


@dataclass
class Prediction:
    args: Dict[str, Any]
    plan: Optional[SimulationPlan]
    executed: bool
    error: Optional[str] = None


@dataclass
class CaseResult:
    prompt: str
    expected: Dict[str, Any]
    predicted: Dict[str, Any]
    correct: List[str] = field(default_factory=list)
    missed: List[str] = field(default_factory=list)
    spurious: List[str] = field(default_factory=list)
    exact: bool = False
    guardrail_held: Optional[bool] = None
    defaults_flagged: Optional[bool] = None
    executed: bool = False
    error: Optional[str] = None


def _same(key: str, a: Any, b: Any) -> bool:
    if key == "subhalo_mass_slope":
        a, b = abs(a), abs(b)  # "slope 1.9" and "slope -1.9" mean the same mass function
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6)
    return str(a).lower() == str(b).lower()


def _plan_value(plan: SimulationPlan, key: str) -> Any:
    return getattr(plan.parameters, key)


def score_case(case: BenchmarkCase, pred: Prediction) -> CaseResult:
    predicted = {
        k: v for k, v in pred.args.items()
        if v is not None and not (k not in case.expected and k in TOOL_DEFAULTS and _same(k, v, TOOL_DEFAULTS[k]))
    }
    r = CaseResult(case.prompt, dict(case.expected), predicted, executed=pred.executed, error=pred.error)
    for k, v in case.expected.items():
        (r.correct if k in predicted and _same(k, predicted[k], v) else r.missed).append(k)
    r.spurious = [k for k, v in predicted.items() if k not in case.expected or not _same(k, v, case.expected[k])]
    r.exact = not r.missed and not r.spurious

    if case.forbidden is not None:
        r.guardrail_held = pred.plan is None or not all(
            _same(k, _plan_value(pred.plan, k), v) for k, v in case.forbidden.items()
        )
    elif pred.plan is not None:
        flagged = {c.parameter_name for c in pred.plan.clarifications}
        needed = {k for k in ("z_lens", "z_source") if k not in case.expected}
        r.defaults_flagged = needed <= flagged
    return r


def rule_based_planner(prompt: str) -> Prediction:
    """The offline parser followed by the same planning tool the LLM calls."""
    args = DeepLenseWorkflowSession()._extract_parameters(prompt)
    ctx = SimulationSessionContext()
    try:
        plan_simulation(ctx, **args)
    except ValueError:
        pass  # rejected by the Pydantic guardrails
    return Prediction(args=args, plan=ctx.active_plan, executed=ctx.last_result is not None)


def llm_planner(model: Any, temperature: float = 0.0) -> Callable[[str], Prediction]:
    """Runs the full agent on a prompt and records its first tool_plan_simulation call."""
    agent = create_pydantic_ai_agent(model)

    def plan(prompt: str) -> Prediction:
        ctx = SimulationSessionContext()
        try:
            result = _run_blocking(agent.run_sync, prompt, deps=ctx, model_settings={"temperature": temperature})
            messages = result.all_messages()
            error = None
        except Exception as e:  # provider errors are recorded, not fatal for the whole benchmark
            messages, error = [], f"{type(e).__name__}: {str(e)[:200]}"
        calls = [
            part for m in messages if isinstance(m, ModelResponse)
            for part in m.parts if isinstance(part, ToolCallPart) and part.tool_name == "tool_plan_simulation"
        ]
        args = calls[0].args_as_dict() if calls else {}
        return Prediction(args=args, plan=ctx.active_plan, executed=ctx.last_result is not None, error=error)

    return plan


def run_benchmark(planner: Callable[[str], Prediction], cases: List[BenchmarkCase] = BENCHMARK_CASES,
                  delay_s: float = 0.0) -> Dict[str, Any]:
    results = []
    for i, case in enumerate(cases):
        if i and delay_s:
            time.sleep(delay_s)  # stay under provider rate limits
        results.append(score_case(case, planner(case.prompt)))
    return summarize(results)


def summarize(results: List[CaseResult]) -> Dict[str, Any]:
    tp = sum(len(r.correct) for r in results)
    fn = sum(len(r.missed) for r in results)
    fp = sum(len(r.spurious) for r in results)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    guard = [r.guardrail_held for r in results if r.guardrail_held is not None]
    flagged = [r.defaults_flagged for r in results if r.defaults_flagged is not None]
    return {
        "cases": len(results),
        "field_precision": round(precision, 3),
        "field_recall": round(recall, 3),
        "field_f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "exact_match": round(sum(r.exact for r in results) / len(results), 3),
        "guardrail_held": f"{sum(guard)}/{len(guard)}",
        "defaults_flagged": f"{sum(flagged)}/{len(flagged)}",
        "executed_without_approval": sum(r.executed for r in results),
        "errors": sum(r.error is not None for r in results),
        "per_case": [asdict(r) for r in results],
    }


def format_summary(name: str, s: Dict[str, Any]) -> str:
    keys = ["cases", "field_precision", "field_recall", "field_f1", "exact_match",
            "guardrail_held", "defaults_flagged", "executed_without_approval", "errors"]
    return "\n".join([f"== {name} =="] + [f"  {k:<26} {s[k]}" for k in keys])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], help="pydantic-ai model string; repeatable")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between LLM prompts (rate limits)")
    ap.add_argument("--json", help="write full results, including per-case scores, to this file")
    args = ap.parse_args()

    summaries = {"rule_based_parser": run_benchmark(rule_based_planner)}
    for model in args.model:
        summaries[model] = run_benchmark(llm_planner(model), delay_s=args.delay)
    for name, s in summaries.items():
        print(format_summary(name, s))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summaries, f, indent=2, default=str)


if __name__ == "__main__":
    main()
