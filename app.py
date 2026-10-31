"""
Streamlit front end for the DeepLense simulation agent.

Run locally:  streamlit run app.py
"""

import io
import json
import os
import tempfile

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from astropy.io import fits
from matplotlib.colors import LogNorm

from deeplense_agent import DeepLenseWorkflowSession, WorkflowState
from deeplense_agent.agent.agent import explain_error
from deeplense_agent.agent.tools import parameter_facts, refine_simulation_parameters

st.set_page_config(page_title="DeepLense", page_icon=":material/blur_circular:", layout="wide")

OFFLINE = "Offline parser (no API key)"
AGENTS = {
    "GPT-OSS 120B · Groq": "groq:openai/gpt-oss-120b",
    "GPT-OSS 20B · Groq (faster)": "groq:openai/gpt-oss-20b",
    "Claude Sonnet 5.5": "anthropic:claude-sonnet-5-5",
    "Claude Haiku 5.5": "anthropic:claude-haiku-5-5",
    "Custom model string": "custom",
    OFFLINE: None,
}
KEY_ENV = {"groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google-gla": "GEMINI_API_KEY"}
MODELS = ["Model_I", "Model_II", "Model_III", "Model_IV"]
MODEL_LABELS = {
    "Model_I": "Model I · generic optical",
    "Model_II": "Model II · Euclid VIS",
    "Model_III": "Model III · HST WFC3",
    "Model_IV": "Model IV · Euclid, real galaxy",
}
SUBSTRUCTURES = {"no_sub": "None (smooth SIE)", "cdm": "CDM subhalos", "axion": "Axion vortex"}
EXAMPLES = [
    "Simulate a lens with CDM subhalos",
    "Axion vortex on HST, z_lens 0.4, z_source 2.0",
    "Model IV, 6 images, 60 subhalos with slope 1.9, seed 7",
]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"], .stMarkdown, .stButton button, input, textarea, select {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .block-container { padding-top: 2.2rem; max-width: 1280px; }
    #MainMenu, footer { visibility: hidden; }
    h1 { font-weight: 600; letter-spacing: -0.02em; font-size: 1.9rem; margin-bottom: 0; }
    .section { font-weight: 500; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em;
               color: #5b5a55; border-bottom: 1px solid #e2dfd6; padding-bottom: 0.35rem; margin: 0.4rem 0 0.8rem 0; }
    [data-testid="stChatMessage"] h3 { font-size: 1rem; font-weight: 600; margin-top: 0.6rem; }
    .subtitle { color: #6b6a64; margin: 0.2rem 0 1.4rem 0; }
    .kv { display: grid; grid-template-columns: max-content 1fr; gap: 0.3rem 1.4rem; font-size: 0.93rem; }
    .kv .k { color: #6b6a64; }
    .kv .v { font-family: 'IBM Plex Mono', monospace; }
    .stats { display: flex; flex-wrap: wrap; gap: 0 2.2rem; margin: 0.4rem 0 0.8rem 0; }
    .stats div { display: flex; flex-direction: column; }
    .stats .k { font-size: 0.78rem; color: #6b6a64; text-transform: uppercase; letter-spacing: 0.05em; }
    .stats .v { font-family: 'IBM Plex Mono', monospace; font-size: 1.25rem; }
    .question { border-left: 2px solid #c8a45c; padding: 0.15rem 0 0.15rem 0.8rem; margin: 0.55rem 0; }
    .question small { color: #6b6a64; display: block; margin-top: 0.15rem; }
    .thumbs { font-size: 0.78rem; color: #6b6a64; }
    .status { font-size: 0.85rem; color: #6b6a64; }
    .status b { color: #24557a; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- session ----------

def secret(name: str) -> Optional[str]:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def build_model(model_string: str, api_key: Optional[str]):
    """Build a pydantic-ai model with an explicit key so keys never leak between app users."""
    provider, _, name = model_string.partition(":")
    if provider == "groq":
        from pydantic_ai.models.groq import GroqModel, GroqModelSettings
        from pydantic_ai.providers.groq import GroqProvider
        settings = GroqModelSettings(temperature=0.2)
        if name.startswith("openai/gpt-oss"):
            # Low reasoning effort keeps turns fast; the tools do the physics
            settings["groq_reasoning_effort"] = "low"
        return GroqModel(name, provider=GroqProvider(api_key=api_key), settings=settings)
    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        return AnthropicModel(name, provider=AnthropicProvider(api_key=api_key))
    if provider == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        return OpenAIChatModel(name, provider=OpenAIProvider(api_key=api_key))
    if provider == "google-gla":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
        return GoogleModel(name, provider=GoogleProvider(api_key=api_key))
    raise ValueError(f"Unsupported provider '{provider}'. Use groq:, anthropic:, openai: or google-gla:.")


def reset_session(model=None):
    st.session_state.session = DeepLenseWorkflowSession(agent_llm=model)
    st.session_state.messages = []
    st.session_state.image_idx = 0


with st.sidebar:
    st.markdown("**Agent**")
    choice = st.selectbox("Language model", list(AGENTS), label_visibility="collapsed")
    model_string = AGENTS[choice]
    if model_string == "custom":
        model_string = st.text_input("Model", placeholder="openai:gpt-4o-mini")

    model, model_error = None, None
    if model_string:
        provider = model_string.partition(":")[0]
        env_name = KEY_ENV.get(provider, "")
        stored_key = secret(env_name) if env_name else None
        typed_key = st.text_input(
            "API key",
            type="password",
            placeholder="Using server key" if stored_key else env_name,
            help="Kept only in this browser session.",
        )
        key = typed_key or stored_key
        if key:
            try:
                model = build_model(model_string, key)
            except Exception as e:
                model_error = str(e)
        else:
            model_error = "Add an API key to use this model. Falling back to the offline parser."
    if model_error:
        st.caption(model_error)

    agent_id = (model_string or "offline") if model is not None else "offline"
    if st.session_state.get("agent_id") != agent_id:
        st.session_state.agent_id = agent_id
        reset_session(model)

    st.divider()
    if st.button("New session", width="stretch"):
        reset_session(model)
        st.rerun()
    st.caption(
        "The agent turns your request into a validated simulation plan. "
        "Nothing runs until you approve it."
    )

session: DeepLenseWorkflowSession = st.session_state.session
messages: list = st.session_state.messages


def send(text: str):
    messages.append(("user", text))
    try:
        resp = session.chat(text)
        messages.append(("assistant", resp.message))
    except Exception as e:
        messages.append(("assistant", f"Something went wrong: `{e}`"))


def run_plan():
    try:
        resp = session.run_active_plan()
        messages.append(("assistant", resp.message))
    except Exception as e:
        messages.append(("assistant", f"Run failed: `{e}`"))
    st.session_state.image_idx = 0


def decline_plan():
    resp = session.reject("Declined from the plan panel.")
    messages.append(("assistant", resp.message))


# ---------- rendering helpers ----------

def show_image(arr: np.ndarray, extent, cmap: str, log: bool = False, label: str = ""):
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=130)
    norm = None
    if log:
        lo = max(float(np.percentile(arr, 5)), 1e-3)
        hi = float(np.percentile(arr, 99.5))
        norm = LogNorm(vmin=lo, vmax=hi) if lo < hi else None
    im = ax.imshow(arr, origin="lower", cmap=cmap, extent=extent, norm=norm)
    ax.set_xlabel("arcsec", fontsize=8, color="#555")
    ax.set_ylabel("arcsec", fontsize=8, color="#555")
    ax.tick_params(labelsize=7, colors="#555")
    for spine in ax.spines.values():
        spine.set_color("#bbb")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7, colors="#555")
    cb.outline.set_edgecolor("#bbb")
    if label:
        cb.set_label(label, fontsize=8, color="#555")
    fig.tight_layout()
    st.pyplot(fig, width="content")
    plt.close(fig)


def thumbnail(arr: np.ndarray) -> np.ndarray:
    """Observed image scaled to 0..1 with an asinh stretch for the batch strip."""
    a = np.arcsinh(np.clip(arr, 0, None) / (np.percentile(arr, 99.5) * 0.1 + 1e-12))
    rgb = plt.get_cmap("afmhot")(a / (a.max() + 1e-12))[..., :3]
    return (rgb * 255).astype(np.uint8)


def fits_bytes(bundle) -> bytes:
    hdus = [fits.PrimaryHDU(bundle.images_observed.astype(np.float32))]
    hdus[0].header["EXTNAME"] = "OBSERVED"
    for name, arr in [("NOISELESS", bundle.images_noiseless), ("KAPPA", bundle.kappa_maps), ("SOURCE", bundle.source_planes)]:
        hdus.append(fits.ImageHDU(arr.astype(np.float32), name=name))
    meta = bundle.output_metadata.images_metadata[0]
    hdus[0].header["MODEL"] = meta.model_configuration.value
    hdus[0].header["SUBSTR"] = meta.substructure_type.value
    hdus[0].header["Z_LENS"] = meta.z_lens
    hdus[0].header["Z_SRC"] = meta.z_source
    hdus[0].header["PIXSCALE"] = (meta.pixel_scale_arcsec, "arcsec/pixel")
    hdus[0].header["THETA_E"] = (meta.einstein_radius_arcsec, "arcsec")
    hdus[0].header["SEED"] = bundle.output_metadata.random_seed
    buf = io.BytesIO()
    fits.HDUList(hdus).writeto(buf)
    return buf.getvalue()


def section(title: str):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


def kv(rows):
    cells = "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in rows)
    st.markdown(f'<div class="kv">{cells}</div>', unsafe_allow_html=True)


# ---------- page ----------

st.markdown("# DeepLense")
st.markdown(
    '<p class="subtitle">Strong gravitational lensing simulations from plain-language requests. '
    "Describe a lens, check the plan, then run it.</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([5, 7], gap="large")

with left:
    section("Conversation")
    mode = "offline parser" if not session.uses_llm else (model_string or "")
    st.markdown(f'<p class="status">Agent: <b>{mode}</b></p>', unsafe_allow_html=True)

    chat_box = st.container(height=560, border=False)
    with chat_box:
        if not messages:
            st.caption("Try one of these, or write your own below.")
            for ex in EXAMPLES:
                if st.button(ex, key=f"ex_{ex}"):
                    send(ex)
                    st.rerun()
        for role, text in messages:
            with st.chat_message(role, avatar=":material/person:" if role == "user" else ":material/blur_circular:"):
                st.markdown(text)

prompt = st.chat_input("Describe the lens you want to simulate")
if prompt:
    with st.spinner("Thinking"):
        send(prompt)
    st.rerun()

plan = session.ctx.active_plan
bundle = session.ctx.last_result

with right:
    section("Plan")
    if plan is None:
        st.caption("No plan yet. The agent will draft one from your first message.")
    else:
        p = plan.parameters
        facts = parameter_facts(p)
        kv([
            ("Instrument", MODEL_LABELS[p.model_configuration.value]),
            ("Grid", facts["grid"].replace("x", " × ", 1)),
            ("Substructure", facts["substructure"]),
            ("Redshifts", f"z_l {p.z_lens}  ·  z_s {p.z_source}"),
            ("Einstein radius", facts["einstein_radius"]),
            ("Source", facts["source"]),
            ("Lens light", facts["lens_light"]),
            ("Images · seed", f"{p.num_images}  ·  {facts['seed']}"),
        ])

        executed = bundle is not None and bundle.parameters is p
        if plan.clarifications and not executed:
            st.markdown("")
            st.markdown("**Worth checking before you run**")
            for c in plan.clarifications:
                st.markdown(
                    f'<div class="question">{c.question_text}<small>{c.scientific_rationale}</small></div>',
                    unsafe_allow_html=True,
                )

        with st.expander("Edit parameters"):
            with st.form("edit_plan", border=False):
                c1, c2 = st.columns(2)
                m = c1.selectbox("Instrument", MODELS, index=MODELS.index(p.model_configuration.value),
                                 format_func=lambda v: MODEL_LABELS[v])
                subs = list(SUBSTRUCTURES)
                s = c2.selectbox("Substructure", subs, index=subs.index(p.substructure.substructure_type.value),
                                 format_func=lambda v: SUBSTRUCTURES[v])
                c3, c4, c5 = st.columns(3)
                zl = c3.number_input("z lens", 0.01, 5.0, float(p.z_lens), 0.05, format="%.2f")
                zs = c4.number_input("z source", 0.05, 10.0, float(p.z_source), 0.05, format="%.2f")
                n = c5.number_input("Images", 1, 50, int(p.num_images))
                cdm = p.substructure.cdm_params
                axion = p.substructure.axion_params
                c6, c7, c8 = st.columns(3)
                n_sub = c6.number_input("Subhalos (CDM)", 1, 200, cdm.n_sub if cdm else 25)
                log_ma = c7.number_input(
                    "log₁₀ axion mass (eV)", -25.0, -19.0,
                    float(np.log10(axion.axion_mass_eV)) if axion and axion.axion_mass_eV else None,
                    0.5, format="%.1f", placeholder="random",
                )
                mag = c8.number_input("Source magnitude", 12.0, 30.0, float(p.source_params.magnitude), 0.5,
                                      format="%.1f", disabled=p.model_configuration.value == "Model_I")
                c9, c10, _ = st.columns(3)
                seed = c9.number_input("Seed", 0, 2**32 - 1, p.random_seed, placeholder="random")
                light = c10.toggle("Lens light", p.lens_light_enabled)
                if st.form_submit_button("Update plan"):
                    try:
                        refine_simulation_parameters(
                            session.ctx, z_lens=zl, z_source=zs, substructure_type=s,
                            model_configuration=m, num_images=int(n),
                            num_subhalos=int(n_sub) if s == "cdm" else None,
                            axion_mass_eV=10 ** log_ma if s == "axion" and log_ma is not None else None,
                            source_magnitude=mag if m != "Model_I" else None,
                            random_seed=int(seed) if seed is not None else None,
                            include_lens_light=light if light != p.lens_light_enabled else None,
                        )
                        if not session.uses_llm:
                            session.state = WorkflowState.READY_TO_EXECUTE
                        st.rerun()
                    except ValueError as e:
                        st.error(explain_error(e))

        if p.model_configuration.value == "Model_IV":
            upload = st.file_uploader(
                "Source galaxy image (optional)", type=["png", "jpg", "jpeg", "fits", "npy"],
                help="Lensed as the Model IV source. Without one, a procedural spiral galaxy is used. "
                     "A Galaxy10_DECals.h5 path can be set from Python.",
            )
            if upload is not None and st.session_state.get("uploaded_name") != upload.name:
                folder = st.session_state.setdefault("upload_dir", tempfile.mkdtemp(prefix="deeplense_"))
                path = os.path.join(folder, os.path.basename(upload.name))
                with open(path, "wb") as fh:
                    fh.write(upload.getbuffer())
                try:
                    refine_simulation_parameters(session.ctx, galaxy_image_path=path)
                    st.session_state.uploaded_name = upload.name
                    st.rerun()
                except ValueError as e:
                    st.error(explain_error(e))

        if not executed or session.pending_approval is not None:
            b1, b2, _ = st.columns([1.6, 1, 2])
            label = "Approve and run" if session.pending_approval is not None else "Run simulation"
            if b1.button(label, type="primary", width="stretch"):
                with st.spinner("Simulating"):
                    run_plan()
                st.rerun()
            if session.pending_approval is not None and b2.button("Decline", width="stretch"):
                decline_plan()
                st.rerun()

    st.markdown("")
    section("Result")
    if bundle is None:
        st.caption("Results show up here after a run.")
    else:
        n_img = len(bundle.images_observed)
        idx = 0
        if n_img > 1:
            thumbs = [thumbnail(img) for img in bundle.images_observed]
            st.image(thumbs, width=72, caption=[str(i + 1) for i in range(n_img)])
            idx = st.slider("Image", 1, n_img, min(st.session_state.image_idx, n_img - 1) + 1) - 1
            st.session_state.image_idx = idx
        meta = bundle.output_metadata.images_metadata[idx]
        extra = ""
        if meta.axion_mass_eV:
            extra = f'<div><span class="k">Axion mass</span><span class="v">{meta.axion_mass_eV:.1e}</span></div>'
        st.markdown(
            '<div class="stats">'
            f'<div><span class="k">Einstein radius</span><span class="v">{meta.einstein_radius_arcsec:.3f}"</span></div>'
            f'<div><span class="k">Arc SNR</span><span class="v">{meta.snr:,.0f}</span></div>'
            f'<div><span class="k">Peak pixel SNR</span><span class="v">{meta.peak_snr:.0f}</span></div>'
            f'<div><span class="k">Substructure</span><span class="v">{meta.subhalo_count}</span></div>'
            f"{extra}"
            f'<div><span class="k">Seed</span><span class="v">{bundle.output_metadata.random_seed}</span></div>'
            "</div>",
            unsafe_allow_html=True,
        )

        half = meta.dimensions[0] * meta.pixel_scale_arcsec / 2
        extent = [-half, half, -half, half]
        tabs = st.tabs(["Observed", "Convergence κ", "Noiseless", "Source plane"])
        with tabs[0]:
            show_image(bundle.images_observed[idx], extent, "afmhot", label="counts")
        with tabs[1]:
            show_image(bundle.kappa_maps[idx], extent, "viridis", log=True, label="κ, pixel-averaged (log)")
        with tabs[2]:
            show_image(bundle.images_noiseless[idx], extent, "afmhot", label="counts")
        with tabs[3]:
            show_image(bundle.source_planes[idx], extent, "afmhot", label="surface brightness")

        metadata = {
            "parameters": json.loads(bundle.parameters.model_dump_json()),
            "output": json.loads(bundle.output_metadata.model_dump_json()),
        }
        npz = io.BytesIO()
        np.savez_compressed(
            npz, observed=bundle.images_observed, noiseless=bundle.images_noiseless,
            kappa=bundle.kappa_maps, source=bundle.source_planes,
        )
        stem = f"deeplense_{bundle.output_metadata.plan_id}"
        d1, d2, d3, _ = st.columns([1, 1, 1, 2])
        d1.download_button("FITS", fits_bytes(bundle), f"{stem}.fits", width="stretch")
        d2.download_button("NumPy", npz.getvalue(), f"{stem}.npz", width="stretch")
        d3.download_button("Metadata", json.dumps(metadata, indent=2), f"{stem}.json", width="stretch")
