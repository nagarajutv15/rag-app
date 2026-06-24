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
    

    @staticmethod
    def get_employees_reporting_to(
            db: Session,
            manager_name: str
    ):

        manager = (
            db.query(Employee)
            .filter(
                Employee.employee_name.ilike(
                    manager_name
                )
            )
            .first()
        )

        if not manager:
            return []

        employees = (
            db.query(Employee)
            .filter(
                Employee.manager_id ==
                manager.employee_id
            )
            .all()
        )

        return [
            employee.employee_name
            for employee in employees
        ]
    

    @staticmethod
    def get_asset_owner(
            db: Session,
            asset_code: str
    ):

        asset = (
            db.query(Asset)
            .filter(
                Asset.asset_code == asset_code
            )
            .first()
        )

        if not asset:
            return None

        return {
            "asset_code": asset.asset_code,
            "owner": (
                asset.employee.employee_name
                if asset.employee
                else None
            )
        }
    
    @staticmethod
    def get_employees_by_department(
            db: Session,
            department_name: str
    ):

        department = (
            db.query(Department)
            .filter(
                Department.department_name.ilike(
                    department_name
                )
            )
            .first()
        )

        if not department:
            return []

        return [
            employee.employee_name
            for employee in department.employees
        ]    
    
    @staticmethod
    def get_projects_by_client(
            db: Session,
            client_name: str
    ):

        projects = (
            db.query(Project)
            .join(Project.client)
            .filter(
                Project.client.has(
                    client_name=client_name
                )
            )
            .all()
        )

        return [
            project.project_name
            for project in projects
        ]    
    
    