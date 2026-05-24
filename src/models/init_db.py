from src.models.database import (
    engine,
    Base
)
from src.models.seed import seed_data
from src.models.document_schema import (
    DocumentMetadata
)
from src.models.schema import (
    Department,
    Employee,
    Project,
    EmployeeProject,
    EmploymentHistory
)

def initialize_database():

    # Create all tables
    Base.metadata.create_all(bind=engine)

    print(" Tables created")

    # Insert dummy data
    seed_data()
    print(Base.metadata.tables.keys())

    print(" Database initialized")