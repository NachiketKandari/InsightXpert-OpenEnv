"""FastAPI application for the BIRD Text-to-SQL OpenEnv environment."""

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

app = create_app(
    BirdEnvironment,
    BirdSQLAction,
    BirdSQLObservation,
    env_name="InsightXpert-OpenEnv",
    max_concurrent_envs=64,
    gradio_builder=build_custom_gradio_app,
)


def main():
    """Entry point for running the server directly."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
