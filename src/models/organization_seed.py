from datetime import date, datetime

from src.models.orgaization import (
    Department,
    Employee,
    Client,
    Project,
    Payroll,
    SalaryHistory,
    Asset,
    LeaveRecord,
    AuditLog
)


def seed_organization_data(db):

    # Seed only once
    existing_employee = db.query(Employee).first()

    if existing_employee:
        return

    print("Seeding organization data...")

    # -----------------------------------------------------
    # Departments
    # -----------------------------------------------------

    engineering = Department(
        department_name="Engineering",
        location="Bangalore",
        budget=5000000
    )

    hr_department = Department(
        department_name="HR",
        location="Bangalore",
        budget=1000000
    )

    finance = Department(
        department_name="Finance",
        location="Mumbai",
        budget=2000000
    )

    sales = Department(
        department_name="Sales",
        location="Hyderabad",
        budget=3000000
    )

    admin = Department(
        department_name="Admin",
        location="Bangalore",
        budget=500000
    )

    db.add_all([
        engineering,
        hr_department,
        finance,
        sales,
        admin
    ])

    db.flush()

    # -----------------------------------------------------
    # Employees
    # -----------------------------------------------------

    hr_manager = Employee(
        employee_code="EMP001",
        employee_name="Emma Johnson",
        email="emma@company.com",
        phone="9000000001",
        designation="HR Manager",
        department=hr_department,
        employment_type="Permanent",
        status="ACTIVE"
    )

    engineering_director = Employee(
        employee_code="EMP002",
        employee_name="Sarah Wilson",
        email="sarah@company.com",
        phone="9000000002",
        designation="Engineering Director",
        department=engineering,
        employment_type="Permanent",
        status="ACTIVE"
    )

    db.add_all([
        hr_manager,
        engineering_director
    ])

    db.flush()

    engineering_manager = Employee(
        employee_code="EMP003",
        employee_name="John Smith",
        email="john@company.com",
        phone="9000000003",
        designation="Engineering Manager",
        department=engineering,
        manager=engineering_director,
        hr=hr_manager,
        employment_type="Permanent",
        status="ACTIVE"
    )

    nagaraju = Employee(
        employee_code="EMP004",
        employee_name="Nagaraju",
        email="nagaraju@company.com",
        phone="9000000004",
        designation="Senior Developer",
        department=engineering,
        manager=engineering_manager,
        hr=hr_manager,
        employment_type="Permanent",
        status="ACTIVE"
    )

    priya = Employee(
        employee_code="EMP005",
        employee_name="Priya Sharma",
        email="priya@company.com",
        phone="9000000005",
        designation="Software Developer",
        department=engineering,
        manager=engineering_manager,
        hr=hr_manager,
        employment_type="Permanent",
        status="ACTIVE"
    )

    db.add_all([
        engineering_manager,
        nagaraju,
        priya
    ])

    db.flush()

    # Department Heads

    engineering.department_head_id = engineering_director.employee_id
    hr_department.department_head_id = hr_manager.employee_id

    # -----------------------------------------------------
    # Clients
    # -----------------------------------------------------

    google = Client(
        client_name="Google",
        industry="Technology",
        country="USA",
        contact_person="Mark Brown",
        account_manager=engineering_manager
    )

    microsoft = Client(
        client_name="Microsoft",
        industry="Technology",
        country="USA",
        contact_person="Steve Clark",
        account_manager=engineering_manager
    )

    amazon = Client(
        client_name="Amazon",
        industry="Retail",
        country="USA",
        contact_person="David Lee",
        account_manager=engineering_manager
    )

    db.add_all([
        google,
        microsoft,
        amazon
    ])

    db.flush()

    # -----------------------------------------------------
    # Projects
    # -----------------------------------------------------

    rag_project = Project(
        project_name="Agentic RAG Platform",
        status="ACTIVE",
        start_date=date(2025, 1, 1),
        client=google,
        project_manager=engineering_manager
    )

    employee_portal = Project(
        project_name="Employee Portal",
        status="ACTIVE",
        start_date=date(2025, 3, 1),
        client=microsoft,
        project_manager=engineering_manager
    )

    db.add_all([
        rag_project,
        employee_portal
    ])

    db.flush()

    # -----------------------------------------------------
    # Payroll
    # -----------------------------------------------------

    db.add_all([
        Payroll(
            employee=nagaraju,
            basic_salary=1800000,
            bonus=200000,
            variable_pay=100000,
            effective_date=date(2025, 1, 1)
        ),
        Payroll(
            employee=priya,
            basic_salary=1200000,
            bonus=100000,
            variable_pay=50000,
            effective_date=date(2025, 1, 1)
        )
    ])

    # -----------------------------------------------------
    # Salary History
    # -----------------------------------------------------

    db.add_all([
        SalaryHistory(
            employee=nagaraju,
            old_salary=1500000,
            new_salary=1800000,
            increment_percent=20,
            increment_date=date(2025, 4, 1)
        ),
        SalaryHistory(
            employee=priya,
            old_salary=1000000,
            new_salary=1200000,
            increment_percent=20,
            increment_date=date(2025, 4, 1)
        )
    ])

    # -----------------------------------------------------
    # Assets
    # -----------------------------------------------------

    db.add_all([
        Asset(
            asset_code="LT1001",
            asset_name="Dell Latitude",
            asset_type="Laptop",
            employee=nagaraju,
            purchase_date=date(2025, 1, 1),
            status="ASSIGNED"
        ),
        Asset(
            asset_code="LT1002",
            asset_name="MacBook Pro",
            asset_type="Laptop",
            employee=priya,
            purchase_date=date(2025, 1, 1),
            status="ASSIGNED"
        )
    ])

    # -----------------------------------------------------
    # Leave Records
    # -----------------------------------------------------

    db.add_all([
        LeaveRecord(
            employee=nagaraju,
            leave_type="Casual Leave",
            start_date=date(2025, 6, 10),
            end_date=date(2025, 6, 11),
            status="APPROVED"
        ),
        LeaveRecord(
            employee=priya,
            leave_type="Sick Leave",
            start_date=date(2025, 5, 15),
            end_date=date(2025, 5, 16),
            status="APPROVED"
        )
    ])

    # -----------------------------------------------------
    # Audit Logs
    # -----------------------------------------------------

    db.add_all([
        AuditLog(
            employee=nagaraju,
            action="CREATE_PROJECT",
            entity_type="PROJECT",
            entity_id="1",
            description="Created Agentic RAG Platform",
            event_timestamp=datetime.utcnow()
        ),
        AuditLog(
            employee=engineering_manager,
            action="UPLOAD_DOCUMENT",
            entity_type="DOCUMENT",
            entity_id="101",
            description="Uploaded Leave Policy",
            event_timestamp=datetime.utcnow()
        )
    ])

    db.commit()

    print("Organization seed data inserted successfully.")