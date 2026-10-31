"""
Interactive Multi-Turn Human-in-the-Loop (HITL) Workflow Controller.
Orchestrates conversation state, parameter extraction, clarification queries, and simulation execution.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Optional, List, Dict, Any
import re
import matplotlib.pyplot as plt
from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from deeplense_agent.models.responses import (
    SimulationPlan,
    ClarificationQuestion,
    SimulationExecutionOutput,
)
from deeplense_agent.agent.tools import (
    SimulationSessionContext,
    plan_simulation,
    refine_simulation_parameters,
    execute_deeplense_simulation,
)
from deeplense_agent.agent.agent import create_pydantic_ai_agent
from deeplense_agent.simulation.core import SimulationResultBundle
from deeplense_agent.visualization.plotting import plot_lensing_diagnostic


_AFFIRMATIVE = re.compile(
    r"\b(yes|yep|yeah|sure|proceed|run|execute|confirm(ed)?|go ahead|looks good|ok|okay|do it|approve[d]?)\b"
)


def _is_affirmative(text: str) -> bool:
    return bool(_AFFIRMATIVE.search(text.lower()))


class WorkflowState(str, Enum):
    IDLE = "IDLE"
    CLARIFICATION_PENDING = "CLARIFICATION_PENDING"
    READY_TO_EXECUTE = "READY_TO_EXECUTE"
    COMPLETED = "COMPLETED"


class AgentTurnResponse:
    """Structured response container for each conversational turn."""
    def __init__(
        self,
        message: str,
        state: WorkflowState,
        plan: Optional[SimulationPlan] = None,
        clarifications: Optional[List[ClarificationQuestion]] = None,
        execution_output: Optional[SimulationExecutionOutput] = None,
        bundle: Optional[SimulationResultBundle] = None,
    ):
        self.message = message
        self.state = state
        self.plan = plan
        self.clarifications = clarifications or []
        self.execution_output = execution_output
        self.bundle = bundle

    def __repr__(self) -> str:
        return f"<AgentTurnResponse state={self.state.value} message='{self.message[:60]}...'>"


class DeepLenseWorkflowSession:
    """
    High-level interactive session controller for the DeepLense Agent.

    With `agent_llm` set (any pydantic-ai model or model string), each turn is
    handled by the Pydantic AI agent: the LLM plans and refines via tools, and
    execution pauses on a deferred approval request until the user approves.
    Without an LLM, a deterministic rule-based parser drives the same tools so
    the workflow still runs offline (tests, demos without API keys).
    """

    def __init__(self, agent_llm: Optional[Any] = None):
        self.ctx = SimulationSessionContext()
        self.state = WorkflowState.IDLE
        self.agent_llm = agent_llm
        self.agent = create_pydantic_ai_agent(agent_llm) if agent_llm is not None else None
        self.message_history: List[ModelMessage] = []
        self.pending_approval: Optional[DeferredToolRequests] = None

    @property
    def uses_llm(self) -> bool:
        return self.agent is not None

    @property
    def has_unexecuted_plan(self) -> bool:
        plan, bundle = self.ctx.active_plan, self.ctx.last_result
        return plan is not None and (bundle is None or bundle.parameters is not plan.parameters)

    def chat(self, user_input: str) -> AgentTurnResponse:
        """
        Process a user input turn in the multi-turn HITL dialogue.
        """
        text = user_input.strip()
        if self.agent is None:
            return self._chat_rule_based(text)

        plain_yes = _is_affirmative(text) and not self._extract_parameters(text)
        if self.pending_approval is not None:
            return self.approve() if plain_yes else self.reject(text)
        if plain_yes and self.has_unexecuted_plan:
            # The user's "yes" is the approval; don't make the agent ask a second time
            return self.run_active_plan()

        return self._run_agent(text)

    def run_active_plan(self) -> AgentTurnResponse:
        """
        Run the active plan on an explicit user action (e.g. a Run button).
        If the agent is waiting on approval, this approves it; otherwise the plan executes directly.
        """
        if self.pending_approval is not None:
            return self.approve()

        output = execute_deeplense_simulation(self.ctx)
        bundle = self.ctx.last_result
        self.state = WorkflowState.COMPLETED
        if self.agent is not None:
            # Record the user-approved run so the agent knows about it on later turns
            self.message_history = self.message_history + [
                ModelRequest(parts=[UserPromptPart(content="Run the current plan. (Approved by the user.)")]),
                ModelResponse(parts=[TextPart(content=f"Run complete. {output.summary}")]),
            ]

        meta = bundle.output_metadata.images_metadata[0]
        msg = (
            f"Simulation complete.\n\n"
            f"- **Configuration**: {bundle.parameters.model_configuration.value}\n"
            f"- **Substructure**: {bundle.parameters.substructure.substructure_type.value}\n"
            f"- **Redshifts**: z_lens = {bundle.parameters.z_lens}, z_source = {bundle.parameters.z_source}\n"
            f"- **Einstein Radius (θ_E)**: {meta.einstein_radius_arcsec:.3f}\"\n"
            f"- **Resolution**: {meta.dimensions[0]}x{meta.dimensions[1]} px ({meta.pixel_scale_arcsec}\"/px)\n"
            f"- **Arc SNR**: {meta.snr:.1f} integrated, {meta.peak_snr:.1f} peak pixel\n"
            f"- **Substructure Count**: {meta.subhalo_count} components\n"
            f"- **Seed**: {output.random_seed}\n\n"
            f"Images and convergence maps are ready."
        )
        return AgentTurnResponse(
            message=msg,
            state=self.state,
            plan=self.ctx.active_plan,
            execution_output=output,
            bundle=bundle,
        )

    def approve(self) -> AgentTurnResponse:
        """Approve the agent's pending simulation run and let it continue."""
        return self._resolve_pending(True)

    def reject(self, reason: str = "") -> AgentTurnResponse:
        """Deny the pending run; the reason is passed back to the agent so it can revise the plan."""
        message = f"The user did not approve this run. Their reply: {reason}" if reason else "The user declined this run."
        return self._resolve_pending(ToolDenied(message))

    def _resolve_pending(self, decision: Any) -> AgentTurnResponse:
        if self.agent is None or self.pending_approval is None:
            raise RuntimeError("There is no simulation run waiting for approval.")
        results = DeferredToolResults(
            approvals={call.tool_call_id: decision for call in self.pending_approval.approvals}
        )
        self.pending_approval = None
        return self._run_agent(None, deferred_tool_results=results)

    def _run_agent(self, text: Optional[str], **kwargs: Any) -> AgentTurnResponse:
        previous_result = self.ctx.last_result
        try:
            result = _run_blocking(
                self.agent.run_sync, text, deps=self.ctx, message_history=self.message_history, **kwargs
            )
        except (UnexpectedModelBehavior, ModelHTTPError, ModelAPIError) as e:
            # Keep the session usable: history is unchanged, so the user can rephrase or retry
            return AgentTurnResponse(
                message=f"The language model failed on that turn ({type(e).__name__}: {str(e)[:200]}). Please rephrase or try again.",
                state=self.state,
                plan=self.ctx.active_plan,
            )
        return self._handle_agent_result(result, previous_result)

    def _handle_agent_result(
        self, result: Any, previous_result: Optional[SimulationResultBundle]
    ) -> AgentTurnResponse:
        new_messages = result.new_messages()
        self.message_history = result.all_messages()
        plan = self.ctx.active_plan

        if isinstance(result.output, DeferredToolRequests):
            self.pending_approval = result.output
            self.state = WorkflowState.READY_TO_EXECUTE
            text = _last_text(new_messages)
            if not text and plan is not None:
                text = f"Ready to run: {plan.summary}"
            return AgentTurnResponse(message=text + "\n\nApprove this run?", state=self.state, plan=plan)

        bundle = self.ctx.last_result
        if bundle is not None and bundle is not previous_result:
            self.state = WorkflowState.COMPLETED
            return AgentTurnResponse(
                message=result.output,
                state=self.state,
                plan=plan,
                execution_output=bundle.output_metadata,
                bundle=bundle,
            )

        # A plan that differs from what was last executed is still waiting on the user.
        if self.has_unexecuted_plan:
            self.state = WorkflowState.CLARIFICATION_PENDING
        return AgentTurnResponse(
            message=result.output,
            state=self.state,
            plan=plan,
            clarifications=plan.clarifications if plan else [],
        )

    def _chat_rule_based(self, text: str) -> AgentTurnResponse:
        """Deterministic offline fallback that drives the same tools without an LLM."""
        try:
            return self._rule_based_turn(text)
        except ValueError as e:
            # Pydantic guardrail violations: report the physics problem instead of raising
            msg = e.errors()[0]["msg"].removeprefix("Value error, ") if hasattr(e, "errors") else str(e)
            return AgentTurnResponse(
                message=f"I can't use those values: {msg}",
                state=self.state,
                plan=self.ctx.active_plan,
            )

    def _rule_based_turn(self, text: str) -> AgentTurnResponse:
        # 1. State: Awaiting confirmation / clarification
        if self.state in (WorkflowState.CLARIFICATION_PENDING, WorkflowState.READY_TO_EXECUTE):
            is_affirmative = _is_affirmative(text)
            has_refinements = bool(self._extract_parameters(text))

            if is_affirmative and not has_refinements:
                return self.run_active_plan()

            # Otherwise, user is refining or overriding parameters
            parsed = self._extract_parameters(text)
            refined_plan = refine_simulation_parameters(self.ctx, **parsed)

            # Check if user also asked to execute immediately
            if is_affirmative:
                output = execute_deeplense_simulation(self.ctx)
                bundle = self.ctx.last_result
                self.state = WorkflowState.COMPLETED
                msg = f"Parameters updated and simulation executed!\n\n{output.summary}"
                return AgentTurnResponse(
                    message=msg,
                    state=self.state,
                    plan=refined_plan,
                    execution_output=output,
                    bundle=bundle,
                )

            self.state = WorkflowState.READY_TO_EXECUTE
            msg = (
                f"Parameters updated based on your input:\n"
                f"- **Model**: {refined_plan.parameters.model_configuration.value}\n"
                f"- **Substructure**: {refined_plan.parameters.substructure.substructure_type.value}\n"
                f"- **Redshifts**: z_lens = {refined_plan.parameters.z_lens}, z_source = {refined_plan.parameters.z_source}\n"
                f"- **Images**: {refined_plan.parameters.num_images}\n\n"
                f"Would you like me to proceed with generating the simulation?"
            )
            return AgentTurnResponse(
                message=msg,
                state=self.state,
                plan=refined_plan,
            )

        # 2. State: Initial or New Simulation Request
        parsed = self._extract_parameters(text)
        
        plan = plan_simulation(self.ctx, **parsed)

        if plan.requires_user_confirmation and plan.clarifications:
            self.state = WorkflowState.CLARIFICATION_PENDING
            clarification_text = "\n".join([f"- **{c.parameter_name}**: {c.question_text} *(Rationale: {c.scientific_rationale})*" for c in plan.clarifications])
            msg = (
                f"I have initialized a simulation plan based on your request:\n\n"
                f"**Summary**: {plan.summary}\n\n"
                f"Before I run anything, please check these:\n\n"
                f"{clarification_text}\n\n"
                f"Reply with 'proceed' to run with these defaults, or specify your custom values."
            )
            return AgentTurnResponse(
                message=msg,
                state=self.state,
                plan=plan,
                clarifications=plan.clarifications,
            )
        else:
            self.state = WorkflowState.READY_TO_EXECUTE
            msg = (
                f"Simulation plan prepared:\n\n"
                f"**Summary**: {plan.summary}\n\n"
                f"All parameters are fully specified. Would you like to execute the simulation now?"
            )
            return AgentTurnResponse(
                message=msg,
                state=self.state,
                plan=plan,
            )

    def plot_latest(self, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Convenience method to render the astrophysical diagnostic suite for the last generated result."""
        if self.ctx.last_result is None:
            print("No simulation result available to plot.")
            return None
        return plot_lensing_diagnostic(self.ctx.last_result, save_path=save_path)

    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extracts simulation parameters using domain-specific heuristics and patterns."""
        p: Dict[str, Any] = {}
        lower = text.lower()

        def has(pattern: str) -> bool:
            return re.search(pattern, lower) is not None

        # 1. Substructure Type (negations first: "no subhalos" means a smooth lens)
        if has(r"\bno[_ ]sub|\b(no|without)\s+(sub|substructure|subhalos?)\b|\bsmooth\b"):
            p["substructure_type"] = "no_sub"
        elif has(r"\bcdm\b|\bsubhalos?\b|\bpoint mass"):
            p["substructure_type"] = "cdm"
        elif has(r"\baxions?\b|\bvortex|\bwave dark|\bfuzzy\b"):
            p["substructure_type"] = "axion"

        # 2. Model Configuration
        if has(r"\bmodel[_ ]?(iv|4)\b"):
            p["model_configuration"] = "Model_IV"
        elif has(r"\bmodel[_ ]?(iii|3)\b|\bhst\b|\bhubble\b"):
            p["model_configuration"] = "Model_III"
        elif has(r"\bmodel[_ ]?(ii|2)\b|\beuclid\b"):
            p["model_configuration"] = "Model_II"
        elif has(r"\bmodel[_ ]?(i|1)\b|\bground[- ]based\b"):
            p["model_configuration"] = "Model_I"

        # 3. Redshifts
        # Look for z_lens or zl
        zl_match = re.search(r'(?:z_?l(?:ens)?)\s*[:=]?\s*([0-9]+\.?[0-9]*)', lower)
        if zl_match:
            p["z_lens"] = float(zl_match.group(1))

        # Look for z_source or zs
        zs_match = re.search(r'(?:z_?s(?:ource)?)\s*[:=]?\s*([0-9]+\.?[0-9]*)', lower)
        if zs_match:
            p["z_source"] = float(zs_match.group(1))

        # Look for pair e.g. "redshifts 0.5 and 1.5"
        if "z_lens" not in p and "z_source" not in p:
            z_pair = re.search(r'redshifts?\s*([0-9]+\.?[0-9]*)\s*(?:and|,)\s*([0-9]+\.?[0-9]*)', lower)
            if z_pair:
                z1 = float(z_pair.group(1))
                z2 = float(z_pair.group(2))
                p["z_lens"] = min(z1, z2)
                p["z_source"] = max(z1, z2)

        # 4. Number of images
        n_match = re.search(r'(\d+)\s*(?:images?|simulations?|samples?)', lower)
        if n_match:
            p["num_images"] = int(n_match.group(1))

        # 5. Number of subhalos
        sub_match = re.search(r'(\d+)\s*subhalos?', lower)
        if sub_match:
            p["num_subhalos"] = int(sub_match.group(1))

        # 6. Resolution
        res_match = re.search(r'(\d+)\s*x\s*\1', lower)
        if res_match:
            p["num_pix"] = int(res_match.group(1))

        # 7. Physics knobs
        num = r'(-?\d+(?:\.\d+)?(?:e-?\d+)?)'
        slope = re.search(r'slope\s*(?:of\s*)?' + num, lower)
        if slope:
            p["subhalo_mass_slope"] = float(slope.group(1))
        mag = re.search(r'(?:magnitude|mag)\s*(?:of\s*)?' + num, lower)
        if mag:
            p["source_magnitude"] = float(mag.group(1))
        sigma = re.search(r'(?:velocity dispersion|sigma_?v?)\s*(?:of\s*)?' + num + r'\s*km', lower)
        if sigma:
            p["velocity_dispersion_kms"] = float(sigma.group(1))
        m_a = re.search(r'(\d+(?:\.\d+)?e-\d+)\s*ev', lower)
        if m_a:
            p["axion_mass_eV"] = float(m_a.group(1))

        # 8. Seed
        seed_match = re.search(r'seed\s*[:=]?\s*(\d+)', lower)
        if seed_match:
            p["random_seed"] = int(seed_match.group(1))

        return p


def _run_blocking(fn, *args, **kwargs):
    """
    Call a synchronous agent method. Inside a running event loop (Jupyter, async apps)
    `run_sync` cannot nest, so run it on a worker thread instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn(*args, **kwargs)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args, **kwargs).result()


def _last_text(messages: List[ModelMessage]) -> str:
    """Text the model wrote in its most recent response, if any."""
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            texts = [part.content for part in msg.parts if isinstance(part, TextPart)]
            if texts:
                return "\n".join(texts).strip()
    return ""
