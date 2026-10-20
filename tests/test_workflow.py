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
