from datetime import datetime
from sqlalchemy import (
Column,
Integer,
String,
Date,
DateTime,
ForeignKey
)
from sqlalchemy.orm import relationship
from src.models.database import Base


# ==========================================================
# Department
# ==========================================================

class Department(Base):

    __tablename__ = "departments"

    department_id = Column(Integer, primary_key=True, index=True)

    department_name = Column(String(100), nullable=False)

    location = Column(String(100), nullable=False)

    budget = Column(Integer, nullable=False)

    department_head_id = Column(
        Integer,
        ForeignKey("employees.employee_id"),
        nullable=True
    )

    department_head = relationship(
        "Employee",
        foreign_keys=[department_head_id]
    )

    employees = relationship(
        "Employee",
        back_populates="department",
        foreign_keys="Employee.department_id"
    )


# ==========================================================
# Employee
# ==========================================================

class Employee(Base):

    __tablename__ = "employees"

    employee_id = Column(Integer, primary_key=True, index=True)

    employee_code = Column(String(20), unique=True)

    employee_name = Column(String(100), nullable=False)

    email = Column(String(150), unique=True, nullable=False)

    phone = Column(String(20))

    designation = Column(String(100))

    joining_date = Column(
        DateTime,
        default=datetime.utcnow
    )

    employment_type = Column(String(50))

    status = Column(String(20))

    department_id = Column(
        Integer,
        ForeignKey("departments.department_id")
    )

    project_id = Column(
        Integer,
        ForeignKey("projects.project_id"),
        nullable=True
    )

    manager_id = Column(
        Integer,
        ForeignKey("employees.employee_id"),
        nullable=True
    )

    hr_id = Column(
        Integer,
        ForeignKey("employees.employee_id"),
        nullable=True
    )

    department = relationship(
        "Department",
        back_populates="employees",
        foreign_keys=[department_id]
    )

    project = relationship(
        "Project",
        foreign_keys=[project_id]
    )

    manager = relationship(
        "Employee",
        remote_side=[employee_id],
        foreign_keys=[manager_id]
    )

    hr = relationship(
        "Employee",
        remote_side=[employee_id],
        foreign_keys=[hr_id]
    )


# ==========================================================
# Client
# ==========================================================

class Client(Base):

    __tablename__ = "clients"

    client_id = Column(Integer, primary_key=True)

    client_name = Column(String(150), nullable=False)

    industry = Column(String(100))

    country = Column(String(100))

    contact_person = Column(String(100))

    account_manager_id = Column(
        Integer,
        ForeignKey("employees.employee_id")
    )

    account_manager = relationship(
        "Employee",
        foreign_keys=[account_manager_id]
    )

    projects = relationship(
        "Project",
        back_populates="client"
    )


# ==========================================================
# Project
# ==========================================================

class Project(Base):

    __tablename__ = "projects"

    project_id = Column(Integer, primary_key=True)

    project_name = Column(String(150), nullable=False)

    status = Column(String(50))

    start_date = Column(Date)

    end_date = Column(Date)

    client_id = Column(
        Integer,
        ForeignKey("clients.client_id")
    )

    project_manager_id = Column(
        Integer,
        ForeignKey("employees.employee_id")
    )

    client = relationship(
        "Client",
        back_populates="projects"
    )

    project_manager = relationship(
        "Employee",
        foreign_keys=[project_manager_id]
    )

    employees = relationship(
        "Employee",
        back_populates="project",
        foreign_keys="Employee.project_id",
        overlaps="project_manager"
    )


# ==========================================================
# Payroll
# ==========================================================

class Payroll(Base):

    __tablename__ = "payroll"

    payroll_id = Column(Integer, primary_key=True)

    employee_id = Column(
        Integer,
        ForeignKey("employees.employee_id")
    )

    basic_salary = Column(Integer)

    bonus = Column(Integer)

    variable_pay = Column(Integer)

    effective_date = Column(Date)

    employee = relationship(
        "Employee",
        foreign_keys=[employee_id]
    )

# ==========================================================
# Salary History
# ==========================================================

class SalaryHistory(Base):

    __tablename__ = "salary_history"

    salary_history_id = Column(
        Integer,
        primary_key=True
    )

    employee_id = Column(
        Integer,
        ForeignKey("employees.employee_id")
    )

    old_salary = Column(Integer)

    new_salary = Column(Integer)

    increment_percent = Column(Integer)

    increment_date = Column(Date)

    employee = relationship(
        "Employee",
        foreign_keys=[employee_id]
    )


# ==========================================================
# Assets
# ==========================================================

class Asset(Base):

    __tablename__ = "assets"

    asset_id = Column(Integer, primary_key=True)

    asset_code = Column(String(50))

    asset_name = Column(String(100))

    asset_type = Column(String(50))

    assigned_employee_id = Column(
        Integer,
        ForeignKey("employees.employee_id")
    )

    purchase_date = Column(Date)

    status = Column(String(50))

    employee = relationship(
        "Employee",
        foreign_keys=[assigned_employee_id]
    )


# ==========================================================
# Leave Records
# ==========================================================

class LeaveRecord(Base):

    __tablename__ = "leave_records"

    leave_id = Column(Integer, primary_key=True)

    employee_id = Column(
        Integer,
        ForeignKey("employees.employee_id")
    )

    leave_type = Column(String(50))

    start_date = Column(Date)

    end_date = Column(Date)

    status = Column(String(30))

    employee = relationship(
        "Employee",
        foreign_keys=[employee_id]
    )


# ==========================================================
# Audit Logs
# ==========================================================

class AuditLog(Base):

    __tablename__ = "audit_logs"

    audit_id = Column(Integer, primary_key=True)

    user_employee_id = Column(
        Integer,
        ForeignKey("employees.employee_id")
    )

    action = Column(String(100))

    entity_type = Column(String(100))

    entity_id = Column(String(100))

    description = Column(String(500))

    event_timestamp = Column(DateTime)

    employee = relationship(
        "Employee",
        foreign_keys=[user_employee_id]
    )

