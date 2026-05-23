"""Department → Employee
1 Department : N Employees

Employee → EmploymentHistory
1 Employee : N History records

Employee → Project
N : N relationship

EmployeeProject
Bridge table for many-to-many relation

"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Date,
    ForeignKey
)

from sqlalchemy.orm import relationship

from src.models.database import Base


# ======================
# Department
# ======================

class Department(Base):

    __tablename__ = "department"

    department_id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    department_name = Column(
        String,
        nullable=False
    )

    employees = relationship(
        "Employee",
        back_populates="department"
    )


# ======================
# Employee
# ======================

class Employee(Base):

    __tablename__ = "employee"

    employee_id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    name = Column(
        String,
        nullable=False
    )

    email = Column(
        String,
        unique=True,
        nullable=False
    )

    designation = Column(
        String
    )

    joining_date = Column(
        Date
    )

    department_id = Column(
        Integer,
        ForeignKey(
            "department.department_id"
        )
    )

    department = relationship(
        "Department",
        back_populates="employees"
    )

    projects = relationship(
        "EmployeeProject",
        back_populates="employee"
    )

    employment_history = relationship(
        "EmploymentHistory",
        back_populates="employee"
    )


# ======================
# Project
# ======================

class Project(Base):

    __tablename__ = "project"

    project_id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    project_name = Column(
        String,
        nullable=False
    )

    client_name = Column(
        String
    )

    start_date = Column(
        Date
    )

    end_date = Column(
        Date
    )

    employees = relationship(
        "EmployeeProject",
        back_populates="project"
    )


# ======================
# EmployeeProject
# ======================

class EmployeeProject(Base):

    __tablename__ = "employee_project"

    employee_id = Column(
        Integer,
        ForeignKey("employee.employee_id"),
        primary_key=True
    )

    project_id = Column(
        Integer,
        ForeignKey("project.project_id"),
        primary_key=True
    )

    role = Column(
        String
    )

    employee = relationship(
        "Employee",
        back_populates="projects"
    )

    project = relationship(
        "Project",
        back_populates="employees"
    )


# ======================
# Employment History
# ======================

class EmploymentHistory(Base):

    __tablename__ = "employment_history"

    history_id = Column(
        Integer,
        primary_key=True
    )

    employee_id = Column(
        Integer,
        ForeignKey(
            "employee.employee_id"
        )
    )

    previous_company = Column(
        String
    )

    experience_years = Column(
        Integer
    )

    employee = relationship(
        "Employee",
        back_populates="employment_history"
    )
