from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.models.database import get_db
from src.services.organization_service import OrganizationService

router = APIRouter(
    prefix="/organization",
    tags=["Organization"]
)


@router.get("/employees")
def get_employees(
        db: Session = Depends(get_db)
):

    return OrganizationService.get_all_employees(db)


@router.get("/projects")
def get_projects(
        db: Session = Depends(get_db)
):

    return OrganizationService.get_all_projects(db)


@router.get("/assets")
def get_assets(
        db: Session = Depends(get_db)
):

    return OrganizationService.get_all_assets(db)