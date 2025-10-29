import pickle
from fastapi import FastAPI, Request
from pydantic import BaseModel

# Load model and DictVectorizer
with open('pipeline_v2.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI()


@app.post("/predict")
async def predict(request: Request):
    # Parse incoming JSON dynamically
    customer = await request.json()

    # Transform input and make prediction
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    churn = y_pred >= 0.5

    # Prepare response
    result = {
        "churn_probability": float(y_pred[0]),
        "churn": bool(churn[0])
    }
    return result

# Run with:
# uvicorn filename:app --reload
