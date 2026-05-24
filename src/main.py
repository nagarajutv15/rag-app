from fastapi import FastAPI, Depends
from sqlalchemy import text
from src.api.routes import router
from src.models.database import engine, get_db
from src.models.init_db import initialize_database
from src.models.schema import Employee, Project
from src.models.seed import seed_data

app = FastAPI()
app.include_router(router)

@app.on_event("startup")
async def startup():

    initialize_database()

    print(" Database initialized")

@app.get("/employees")
async def get_employees(db= Depends(get_db)):
   
    employees = db.query(Employee).filter(Employee.employee_id > 0).all()
    
    return {
       
       "data": [ 
            {
                "employee_id": emp.employee_id,
                "first_name": emp.name,
                "department":emp.department.department_name if emp.department else None,
                "projects":[
                    {
                        "project_name": p.project.project_name
                        
                    }
                    for p in emp.projects
                ],
                "employee_history":[
                    {
                        "history_id": his.history_id,
                        "privious_company": his.previous_company
                    }
                    for his in emp.employment_history
                ]
                
            }
            for emp in employees
       ]
       
    }
