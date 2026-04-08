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

app = create_app(
    BirdEnvironment,
    BirdSQLAction,
    BirdSQLObservation,
    env_name="bird-text2sql-env",
)
