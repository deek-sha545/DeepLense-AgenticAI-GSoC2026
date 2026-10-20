"""
Interactive Multi-Turn Human-in-the-Loop (HITL) Workflow Controller.
Orchestrates conversation state, parameter extraction, clarification queries, and simulation execution.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import re
import os
import matplotlib.pyplot as plt

from deeplense_agent.models.parameters import (
    ModelConfiguration,
    SubstructureType,
    SimulationParameters,
)
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
    inspect_simulation_quality,
)
from deeplense_agent.simulation.core import SimulationResultBundle
from deeplense_agent.visualization.plotting import plot_lensing_diagnostic


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
    Manages multi-turn natural language dialogues with intelligent parameter parsing,
    HITL confirmation and clarification, and simulation invocation.
    """

    def __init__(self, agent_llm: Optional[Any] = None):
        self.ctx = SimulationSessionContext()
        self.state = WorkflowState.IDLE
        self.agent_llm = agent_llm

    def chat(self, user_input: str) -> AgentTurnResponse:
        """
        Process a user input turn in the multi-turn HITL dialogue.
        """
        text = user_input.strip()
        lower_text = text.lower()

        # 1. State: Awaiting confirmation / clarification
        if self.state in (WorkflowState.CLARIFICATION_PENDING, WorkflowState.READY_TO_EXECUTE):
            # Check for affirmative execution confirmation
            is_affirmative = any(word in lower_text for word in ["yes", "proceed", "run", "execute", "confirm", "go ahead", "looks good", "ok", "okay"])
            has_refinements = any(k in lower_text for k in ["z_lens", "z_source", "redshift", "model", "subhalo", "axion", "images", "resolution"])

            if is_affirmative and not has_refinements:
                # User confirms the current plan -> execute simulation!
                output = execute_deeplense_simulation(self.ctx)
                bundle = self.ctx.last_result
                self.state = WorkflowState.COMPLETED

                meta = bundle.output_metadata.images_metadata[0]
                msg = (
                    f"Simulation successfully executed!\n\n"
                    f"**Configuration**: {bundle.parameters.model_configuration.value}\n"
                    f"**Substructure**: {bundle.parameters.substructure.substructure_type.value}\n"
                    f"**Redshifts**: z_lens = {bundle.parameters.z_lens}, z_source = {bundle.parameters.z_source}\n"
                    f"**Einstein Radius (θ_E)**: {meta.einstein_radius_arcsec:.3f}\"\n"
                    f"**Resolution**: {meta.dimensions[0]}x{meta.dimensions[1]} px ({meta.pixel_scale_arcsec}\"/px)\n"
                    f"**Estimated SNR**: {meta.estimated_snr:.1f}\n"
                    f"**Substructure Count**: {meta.subhalo_count} components\n\n"
                    f"Images and dark matter convergence maps are stored and ready for visualization."
                )
                return AgentTurnResponse(
                    message=msg,
                    state=self.state,
                    plan=self.ctx.active_plan,
                    execution_output=output,
                    bundle=bundle,
                )

            # Otherwise, user is refining or overriding parameters
            parsed = self._extract_parameters(text)
            refined_plan = refine_simulation_parameters(
                self.ctx,
                z_lens=parsed.get("z_lens"),
                z_source=parsed.get("z_source"),
                substructure_type=parsed.get("substructure_type"),
                model_configuration=parsed.get("model_configuration"),
                num_images=parsed.get("num_images"),
            )

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
        
        # Check if redshifts or models were explicitly given or need clarification
        explicit_redshifts = ("z_lens" in parsed and "z_source" in parsed)
        
        plan = plan_simulation(
            self.ctx,
            substructure_type=parsed.get("substructure_type", "no_sub"),
            model_configuration=parsed.get("model_configuration", "Model_II"),
            z_lens=parsed.get("z_lens"),
            z_source=parsed.get("z_source"),
            num_images=parsed.get("num_images", 1),
            num_subhalos=parsed.get("num_subhalos"),
            custom_resolution_pix=parsed.get("resolution_pix"),
            require_clarification_on_defaults=not explicit_redshifts,
        )

        if plan.requires_user_confirmation and plan.clarifications:
            self.state = WorkflowState.CLARIFICATION_PENDING
            clarification_text = "\n".join([f"- **{c.parameter_name}**: {c.question_text} *(Rationale: {c.scientific_rationale})*" for c in plan.clarifications])
            msg = (
                f"I have initialized a simulation plan based on your request:\n\n"
                f"**Summary**: {plan.summary}\n\n"
                f"### Clarification & Refinement (Human-in-the-Loop):\n"
                f"Before I run the compute, please review these parameters:\n"
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

        # 1. Substructure Type
        if "cdm" in lower or "subhalo" in lower or "point mass" in lower:
            p["substructure_type"] = "cdm"
        elif "axion" in lower or "vortex" in lower or "wave" in lower or "fuzzy" in lower:
            p["substructure_type"] = "axion"
        elif "no_sub" in lower or "no sub" in lower or "smooth" in lower:
            p["substructure_type"] = "no_sub"

        # 2. Model Configuration
        if "model_iv" in lower or "model iv" in lower or "model 4" in lower:
            p["model_configuration"] = "Model_IV"
        elif "model_iii" in lower or "model iii" in lower or "model 3" in lower or "hst" in lower or "hubble" in lower:
            p["model_configuration"] = "Model_III"
        elif "model_ii" in lower or "model ii" in lower or "model 2" in lower or "euclid" in lower:
            p["model_configuration"] = "Model_II"
        elif "model_i" in lower or "model i" in lower or "model 1" in lower or "ground" in lower:
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
            p["resolution_pix"] = int(res_match.group(1))

        return p
