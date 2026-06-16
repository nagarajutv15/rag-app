from typing import Optional

from pydantic import BaseModel


class RouteIdentifier(BaseModel):

    route: str

    department: Optional[str] = None