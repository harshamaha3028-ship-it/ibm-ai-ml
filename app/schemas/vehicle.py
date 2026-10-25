from pydantic import BaseModel, Field, field_validator
from typing import Optional, Union

class VehicleInput(BaseModel):
    cylinders: int = Field(..., ge=3, le=12, description="Number of engine cylinders (e.g. 4, 6, 8)", examples=[4])
    displacement: float = Field(..., gt=0.0, le=1000.0, description="Engine displacement in cubic inches", examples=[140.0])
    horsepower: float = Field(..., gt=0.0, le=1000.0, description="Engine horsepower (HP)", examples=[90.0])
    weight: float = Field(..., gt=0.0, le=15000.0, description="Vehicle curb weight in lbs", examples=[2400.0])
    acceleration: float = Field(..., gt=0.0, le=60.0, description="0-60 mph acceleration time in seconds", examples=[15.5])
    model_year: int = Field(..., ge=70, le=99, description="Two-digit model year (70 to 82 standard)", examples=[78])
    origin: Union[int, str] = Field(..., description="Origin: 1/USA, 2/Europe, 3/Japan or string", examples=["USA"])
    car_name: Optional[str] = Field(default="Custom Vehicle", description="Display name for vehicle")

    @field_validator('origin')
    @classmethod
    def validate_origin(cls, v):
        if isinstance(v, int):
            if v not in [1, 2, 3]:
                raise ValueError("Origin code must be 1 (USA), 2 (Europe), or 3 (Japan).")
            return v
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ['1', 'usa', 'united states']:
                return 1
            elif clean in ['2', 'europe', 'european']:
                return 2
            elif clean in ['3', 'japan', 'japanese', 'asia']:
                return 3
            else:
                raise ValueError(f"Origin '{v}' is invalid. Allowed: USA (1), Europe (2), Japan (3).")
        raise ValueError("Invalid origin value format.")
