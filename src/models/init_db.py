from src.models.database import engine
from src.models.schema import Base
from src.models.seed import seed_data


def initialize_database():

    # Create all tables
    Base.metadata.create_all(bind=engine)

    print(" Tables created")

    # Insert dummy data
    seed_data()

    print(" Database initialized")