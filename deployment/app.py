from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LendGuard Home Loan Risk API")

class LoanApplication(BaseModel):
    income: float
    loan_amount: float

@app.get("/")
def home():
    return {"status": "healthy", "message": "LendGuard API is running successfully!"}

@app.post("/predict")
def predict(data: LoanApplication):
    # Temporarily bypass the pickle file to avoid the version crash
    return {
        "status": "Success",
        "message": "API received data! (Model bypass active due to Python version difference)",
        "received_data": {
            "income": data.income,
            "loan_amount": data.loan_amount
        }
    }
