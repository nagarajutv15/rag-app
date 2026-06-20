from sqlalchemy import Date, String, Column, Integer, DateTime, ForeignKey
from datetime import datetime
from sqlalchemy.orm import relationship
from src.models.database import Base

#----------------------------------------------------------------------------------------------------------------------#

class Department(Base):

    __tablename__ = "departments"

    department_id = Column(Integer, primary_key=True, index=True)

    department_name = Column(String(100), nullable=False)

    location = Column(String(100), nullable=False)

    budget = Column(Integer, nullable=False)

    department_head_id = Column(
        Integer,
        ForeignKey("employees.employee_id"),
    )

    employees = relationship(
        "Employee",
        back_populates="department",
        foreign_keys="Employee.department_id"
    )

    
#----------------------------------------------------------------------------------------------------------------------#


class Employee(Base):

    __tablename__ = "employees"

    employee_id = Column(Integer, primary_key=True, index=True)

    employee_code = Column(String(20), unique=True)

    employee_name = Column(String(100), nullable=False)

    email = Column(String(150), unique=True, nullable=False)

    phone = Column(String(20))

    designation = Column(String(100))

    joining_date = Column(DateTime, default=datetime.utcnow)

    employment_type = Column(String(50))

    status = Column(String(20))

    department_id = Column(
        Integer,
        ForeignKey("departments.department_id"),
    )

    manager_id = Column(
        Integer,
        ForeignKey("employees.employee_id")
    )

    hr_id = Column(
        Integer,
        ForeignKey("employees.employee_id")
    )

    department = relationship(
        "Department",
        back_populates="employees",
        foreign_keys=["department_id"]
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


#----------------------------------------------------------------------------------------------------------------------#

class Client(Base):

    __tablename__ = "clients"

    client_id = Column(Integer, primary_key=True)

    client_name = Column(String(150))

    industry = Column(String(100))

    country = Column(String(100))

    contact_person = Column(String(100))

    account_manager_id = Column(
        Integer,
        ForeignKey("employees.employee_id")
    )

    account_manager = relationship(
        "Employee"
    )

    projects = relationship(
        "Project",
        back_populates="client"
    )

#----------------------------------------------------------------------------------------------------------------------#

class Project(Base):
    __tablename__ = "projects"

    project_id = Column(Integer, primary_key=True)

    project_name = Column(String(150))

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
        "Employee"
    )

#----------------------------------------------------------------------------------------------------------------------#

class ProjectMember(Base):
    __tablename__ = "project_members"

    id = Column(Integer, primary_key=True)

    project_id = Column(
        Integer,
        ForeignKey("projects.project_id")
    )

    employee_id = Column(
        Integer,
        ForeignKey("employees.employee_id")
    )

    project_role = Column(String(100))

    allocation_percent = Column(Integer)

    employee = relationship("Employee")

    project = relationship("Project")


#----------------------------------------------------------------------------------------------------------------------#


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

    employee = relationship("Employee")

#----------------------------------------------------------------------------------------------------------------------#


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

    employee = relationship("Employee")


#----------------------------------------------------------------------------------------------------------------------#

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

    employee = relationship("Employee")

#----------------------------------------------------------------------------------------------------------------------#


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

    employee = relationship("Employee")


#----------------------------------------------------------------------------------------------------------------------#


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

    employee = relationship("Employee")