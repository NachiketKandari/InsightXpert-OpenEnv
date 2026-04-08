"""FastAPI application for the BIRD Text-to-SQL OpenEnv environment."""

import os

# Disable the default openenv web interface — we mount our own Gradio UI below.
os.environ["ENABLE_WEB_INTERFACE"] = "false"

from openenv.core.env_server import create_app

try:
    from ..models import BirdSQLAction, BirdSQLObservation
except ImportError:
    from models import BirdSQLAction, BirdSQLObservation

try:
    from .bird_environment import BirdEnvironment
except ImportError:
    from server.bird_environment import BirdEnvironment

try:
    from .gradio_app import build_custom_gradio_app
except ImportError:
    from server.gradio_app import build_custom_gradio_app

# Create the base FastAPI app (REST API only: /reset, /step, /state, /health, /ws)
app = create_app(
    BirdEnvironment,
    BirdSQLAction,
    BirdSQLObservation,
    env_name="InsightXpert-OpenEnv",
    max_concurrent_envs=64,
)

# ── Mount custom Gradio UI at /web ──────────────────────────────────────────

import gradio as gr
from fastapi.responses import RedirectResponse
from openenv.core.env_server.web_interface import (
    WebInterfaceManager,
    load_environment_metadata,
    _extract_action_fields,
    _is_chat_env,
    get_quick_start_markdown,
)

_metadata = load_environment_metadata(BirdEnvironment, "InsightXpert-OpenEnv")
_env = BirdEnvironment()
_web_manager = WebInterfaceManager(_env, BirdSQLAction, BirdSQLObservation, _metadata)
_action_fields = _extract_action_fields(BirdSQLAction)
_is_chat = _is_chat_env(BirdSQLAction)
_quick_start = get_quick_start_markdown(_metadata, BirdSQLAction, BirdSQLObservation)

_blocks = build_custom_gradio_app(
    _web_manager, _action_fields, _metadata, _is_chat,
    _metadata.name if _metadata else "InsightXpert-OpenEnv",
    _quick_start,
)


@app.get("/", include_in_schema=False)
async def _root_redirect():
    return RedirectResponse(url="/web/")


try:
    from .gradio_app import CSS as _CUSTOM_CSS
except ImportError:
    from server.gradio_app import CSS as _CUSTOM_CSS

app = gr.mount_gradio_app(app, _blocks, path="/web", css=_CUSTOM_CSS)


def main():
    """Entry point for running the server directly."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
