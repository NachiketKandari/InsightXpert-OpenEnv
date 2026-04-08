"""FastAPI application for the BIRD Text-to-SQL OpenEnv environment."""

from openenv.core.env_server.http_server import create_app

from ..models import BirdSQLAction, BirdSQLObservation
from .bird_environment import BirdEnvironment

app = create_app(
    BirdEnvironment,
    BirdSQLAction,
    BirdSQLObservation,
    env_name="bird-text2sql-env",
)
