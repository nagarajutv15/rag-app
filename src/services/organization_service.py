from sqlalchemy.orm import Session

from src.models.orgaization import (
    Employee,
    Department,
    Project,
    Asset
)


class OrganizationService:

    @staticmethod
    def get_all_employees(db: Session):

        employees = db.query(Employee).all()

        result = []

        for emp in employees:

            result.append({
                "employee_id": emp.employee_id,
                "employee_code": emp.employee_code,
                "employee_name": emp.employee_name,
                "email": emp.email,
                "designation": emp.designation,
                "department": emp.department.department_name if emp.department else None,
                "manager": emp.manager.employee_name if emp.manager else None,
                "hr": emp.hr.employee_name if emp.hr else None,
                "project": emp.project.project_name if emp.project else None,
                "status": emp.status
            })

        return result

    @staticmethod
    def get_all_projects(db: Session):

        projects = db.query(Project).all()

        result = []

        for project in projects:

            result.append({
                "project_id": project.project_id,
                "project_name": project.project_name,
                "status": project.status,
                "client": project.client.client_name if project.client else None,
                "project_manager": (
                    project.project_manager.employee_name
                    if project.project_manager
                    else None
                )
            })

        return result

    @staticmethod
    def get_all_assets(db: Session):

        assets = db.query(Asset).all()

        result = []

        for asset in assets:

            result.append({
                "asset_id": asset.asset_id,
                "asset_code": asset.asset_code,
                "asset_name": asset.asset_name,
                "asset_type": asset.asset_type,
                "assigned_to": (
                    asset.employee.employee_name
                    if asset.employee
                    else None
                ),
                "status": asset.status
            })

        return result