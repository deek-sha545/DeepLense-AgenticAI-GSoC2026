"""
Unit Tests for Agent Workflow and Human-in-the-Loop Interaction.
"""

import pytest
from deeplense_agent.agent import (
    DeepLenseWorkflowSession,
    WorkflowState,
    create_pydantic_ai_agent,
)
from deeplense_agent.agent.tools import SimulationSessionContext, inspect_simulation_quality


def test_agent_tools_registration():
    agent = create_pydantic_ai_agent()
    tool_names = set(agent._function_toolset.tools.keys())
    assert "tool_plan_simulation" in tool_names
    assert "tool_refine_parameters" in tool_names
    assert "tool_execute_simulation" in tool_names
    assert "tool_inspect_diagnostics" in tool_names


def test_workflow_session_hitl_clarification_cycle():
    session = DeepLenseWorkflowSession()
    assert session.state == WorkflowState.IDLE

    # Turn 1: Underspecified request -> should trigger CLARIFICATION_PENDING
    resp1 = session.chat("Please simulate a lens with CDM subhalos.")
    assert session.state == WorkflowState.CLARIFICATION_PENDING
    assert len(resp1.clarifications) >= 2
    assert resp1.plan is not None
    assert resp1.plan.parameters.substructure.substructure_type.value == "cdm"

    # Turn 2: User confirms defaults -> should execute simulation
    resp2 = session.chat("Proceed with these parameters.")
    assert session.state == WorkflowState.COMPLETED
    assert resp2.bundle is not None
    assert resp2.execution_output is not None
    assert resp2.execution_output.success is True


def test_workflow_session_parameter_refinement():
    session = DeepLenseWorkflowSession()

    # Turn 1: Initial request
    session.chat("Simulate a strong lens.")
    assert session.state == WorkflowState.CLARIFICATION_PENDING

    # Turn 2: Refine model to Model_I and z_source to 2.5
    resp2 = session.chat("Use Model_I with z_lens=0.4 and z_source=2.5, run simulation now.")
    assert session.state == WorkflowState.COMPLETED
    assert resp2.bundle.parameters.model_configuration.value == "Model_I"
    assert resp2.bundle.parameters.z_lens == 0.4
    assert resp2.bundle.parameters.z_source == 2.5
    assert resp2.bundle.primary_image.shape == (150, 150)


# --- LLM-driven path, using a scripted FunctionModel in place of a real LLM ---

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, RetryPromptPart
from pydantic_ai.models.function import FunctionModel


def _scripted_llm(messages, info):
    """Plans on the first user turn, then requests execution, then reports."""
    last = messages[-1].parts
    returned = {p.tool_name for p in last if isinstance(p, ToolReturnPart)}
    if any(isinstance(p, RetryPromptPart) for p in last):
        return ModelResponse(parts=[TextPart("z_lens must be below z_source. Try z_lens=0.5, z_source=1.5?")])
    if "tool_execute_simulation" in returned:
        content = next(p.content for p in last if isinstance(p, ToolReturnPart))
        if isinstance(content, str) and "did not approve" in content:
            return ModelResponse(parts=[TextPart("Understood, not running it.")])
        return ModelResponse(parts=[TextPart("Done.")])
    if "tool_plan_simulation" in returned:
        return ModelResponse(parts=[ToolCallPart("tool_execute_simulation", {})])
    prompt = next(p.content for p in last if p.part_kind == "user-prompt")
    if "bad" in prompt:
        args = {"substructure_type": "cdm", "z_lens": 2.0, "z_source": 1.0}
    else:
        args = {"substructure_type": "cdm", "model_configuration": "Model_III", "z_lens": 0.4, "z_source": 2.0}
    return ModelResponse(parts=[ToolCallPart("tool_plan_simulation", args)])


def test_llm_execution_requires_approval():
    session = DeepLenseWorkflowSession(agent_llm=FunctionModel(_scripted_llm))
    resp1 = session.chat("CDM lens on HST, zl 0.4 zs 2")
    assert resp1.state == WorkflowState.READY_TO_EXECUTE
    assert session.ctx.last_result is None  # nothing ran before approval
    assert resp1.plan.parameters.model_configuration.value == "Model_III"

    resp2 = session.approve()
    assert resp2.state == WorkflowState.COMPLETED
    assert resp2.bundle.primary_image.shape == (64, 64)


def test_llm_rejection_does_not_execute():
    session = DeepLenseWorkflowSession(agent_llm=FunctionModel(_scripted_llm))
    session.chat("CDM lens on HST, zl 0.4 zs 2")
    resp = session.chat("no, wait")
    assert session.ctx.last_result is None
    assert resp.state == WorkflowState.CLARIFICATION_PENDING
    assert "not running" in resp.message


def test_llm_invalid_physics_is_fed_back_to_model():
    session = DeepLenseWorkflowSession(agent_llm=FunctionModel(_scripted_llm))
    resp = session.chat("bad redshifts please")
    assert session.ctx.active_plan is None
    assert "z_lens must be below z_source" in resp.message


@pytest.mark.parametrize("prompt, expected_sub, expected_model", [
    ("smooth lens with no subhalos", "no_sub", "Model_II"),
    ("cdm lens with low background noise", "cdm", "Model_II"),
    ("axion lens on hubble", "axion", "Model_III"),
    ("ground-based model I with subhalos", "cdm", "Model_I"),
])
def test_rule_based_parser(prompt, expected_sub, expected_model):
    parsed = DeepLenseWorkflowSession()._extract_parameters(prompt)
    assert parsed.get("substructure_type") == expected_sub
    assert parsed.get("model_configuration", "Model_II") == expected_model


def test_rule_based_parser_physics_knobs():
    parsed = DeepLenseWorkflowSession()._extract_parameters(
        "Model IV, 60 subhalos with slope 1.9, velocity dispersion 280 km/s, magnitude 23, axion 1e-23 eV, seed 42"
    )
    assert parsed["num_subhalos"] == 60
    assert parsed["subhalo_mass_slope"] == 1.9
    assert parsed["velocity_dispersion_kms"] == 280
    assert parsed["source_magnitude"] == 23
    assert parsed["axion_mass_eV"] == 1e-23
    assert parsed["random_seed"] == 42


def test_plain_yes_runs_pending_plan_once_with_llm():
    session = DeepLenseWorkflowSession(agent_llm=FunctionModel(_scripted_llm))
    session.chat("CDM lens on HST, zl 0.4 zs 2")
    resp = session.chat("yes")
    assert resp.state == WorkflowState.COMPLETED and resp.bundle is not None


def test_llm_session_works_inside_running_event_loop():
    import asyncio

    async def inside_loop():
        session = DeepLenseWorkflowSession(agent_llm=FunctionModel(_scripted_llm))
        return session.chat("CDM lens on HST, zl 0.4 zs 2")

    resp = asyncio.run(inside_loop())  # e.g. Jupyter
    assert resp.state == WorkflowState.READY_TO_EXECUTE
