from pydantic import BaseModel

class PredictRequest(BaseModel):
    # the features your model needs
    # for now just a few key ones
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    AMT_CREDIT: float
    AMT_INCOME_TOTAL: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    
class PredictResponse(BaseModel):
    default_probability: float
    risk_category: str  # "LOW", "MEDIUM", "HIGH"