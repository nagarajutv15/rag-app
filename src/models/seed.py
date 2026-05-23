"""
Data seeding script to populate the database with initial dummy data for testing and development.
"""
from datetime import date
from src.models.database import SessionLocal
from src.models.schema import (
    Department,
    Employee,
    Project,
    EmployeeProject,
    EmploymentHistory
)


def seed_data():

    db = SessionLocal()

    try:

        # Prevent duplicate seeding
        employee_exists = db.query(Employee).first()

        if employee_exists:
            print("Data already exists")
            return

        # Departments
        departments = [
            Department(department_name="Engineering"),
            Department(department_name="HR"),
            Department(department_name="Finance")
        ]

        db.add_all(departments)
        db.commit()

        # Employees
        employees = []

        for i in range(1,11):

            employee = Employee(
                name=f"Employee{i}",
                email=f"employee{i}@company.com",
                designation="Software Engineer",
                joining_date=date(2024,1,i),
                department_id=(i%3)+1
            )

            employees.append(employee)

        db.add_all(employees)
        db.commit()

        # Projects
        projects = [

            Project(
                project_name="AI Chatbot",
                client_name="Microsoft"
            ),

            Project(
                project_name="RAG Platform",
                client_name="Google"
            ),

            Project(
                project_name="HR Assistant",
                client_name="Amazon"
            ),
            
            Project(
                project_name="Analytics Dashboard",
                client_name="Netflix"
            ),

            Project(
                project_name="Customer Support Bot",
                client_name="Adobe"
            )
        ]

        db.add_all(projects)
        db.commit()

        # Employee Project Mapping

        mappings = [

            EmployeeProject(
                employee_id=1,
                project_id=1,
                role="Backend Developer"
            ),

            EmployeeProject(
                employee_id=2,
                project_id=2,
                role="AI Engineer"
            ),

            EmployeeProject(
                employee_id=3,
                project_id=3,
                role="Tech Lead"
            ),

            EmployeeProject(
                employee_id=4,
                project_id=4,
                role="Java Developer"
            ),

            EmployeeProject(
                employee_id=5,
                project_id=5,
                role="MLOps Engineer"
            )
        ]

        db.add_all(mappings)
        db.commit()

        # Employment History

        history=[]

        for i in range(1,11):

            history.append(

                EmploymentHistory(
                    employee_id=i,
                    previous_company=f"Company{i}",
                    experience_years=i
                )
            )

        db.add_all(history)

        db.commit()

        print("Dummy data inserted")

    finally:

        db.close()