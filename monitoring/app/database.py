from logging import getLogger

from sqlmodel import Session, create_engine

from .models import PredictionRecord, Species


# Instanciate logger
logger = getLogger(__name__)


class DatabaseManager:
    """Class that will manage database operations."""

    def __init__(self, database_url: str = "sqlite:///predictions.db"):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=True)

    def create_db_and_tables(self):
        """Initialize database schema."""
        from .models import SQLModel

        SQLModel.metadata.create_all(self.engine)

    def save_prediction(
        self,
        features_input: dict,
        predicted_species: Species,
    ) -> PredictionRecord:
        """Save a prediction record to the database."""
        record = PredictionRecord(**features_input, predicted_species=predicted_species)

        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def get_prediction(self, prediction_id: int) -> PredictionRecord | None:
        """Retrieve a specific prediction by ID."""
        with Session(self.engine) as session:
            return session.get(PredictionRecord, prediction_id)
