from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import test_e
import argparse

app = FastAPI()

# Function to get arguments
def get_args(weight_path, data_path):
    args = argparse.Namespace()
    args.resume = model_last.pth
    args.data_dir = input
    args.dataset = 'spinal'
    args.phase = 'test'
    # Add other arguments with default values here as in your main.py
    return args

# Pydantic model to define the data model for request body
class Item(BaseModel):
    weight_path: str
    data_path: str

@app.post("/predict/")
async def predict(item: Item):
    args = get_args(item.weight_path, item.data_path)
    # Initialize Network
    is_object = test_e.Network(args)
    # Evaluate Network
    is_object.eval(args, save=False)
    # You may want to retrieve and return some results here, 
    # Currently, your eval method does not seem to return anything

    time.sleep(2)
    print("wait 2 sec")

    is_object = test_e.Network1(args)
    is_object.test(args, save=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
