# src/agents/tools/schema.py

DATABASE_SCHEMA = """
Table: employees

Columns:
- employee_id
- employee_code
- employee_name
- email
- designation
- department_id
- manager_id
- hr_id
- status

Table: departments

Columns:
- department_id
- department_name
- location
- budget

Table: assets

Columns:
- asset_id
- asset_code
- asset_name
- asset_type
- assigned_employee_id
- status

Table: projects

Columns:
- project_id
- project_name
- status
- client_id
- project_manager_id

Table: clients

Columns:
- client_id
- client_name
- industry
"""


#-------------------------------------------------------------------------------------------------------#


SQL_GENERATION_PROMPT = """
You are an expert SQL developer.

Generate SQLite SQL only.

Rules:

- Generate ONLY SELECT queries.
- Never generate INSERT.
- Never generate UPDATE.
- Never generate DELETE.
- Never generate DROP.
- Never generate ALTER.
- Never generate CREATE.

Schema:

{schema}

Question:

{question}

Return only SQL.
"""