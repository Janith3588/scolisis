import shutil
from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse ,FileResponse
import torch
import test_e
import time
from main import parse_args, test_e

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
        # Save the uploaded file
    save_path = "input.jpg"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

        # Get the arguments for testing
    args = parse_args()
    args.resume = "model_last.pth" 
    args.data_dir = "input"  
    args.phase = "test"

    # Initialize Network
    is_object = test_e.Network(args)
    # Evaluate Network
    is_object.eval(args, save=False)

    is_object = test_e.Network1(args)
    is_object.test(args, save=False)

    # Return the output image
    output_image_path = f"ori_image_regress_0.jpg"  
    return FileResponse(output_image_path)

def is_valid_image(file_path):
    
    try:
        # Check if the file can be opened as an image
        img = File.open(file_path)
        img.verify()  # Verifying the image file
        return True
    except (IOError, SyntaxError) as e:
        # Invalid image file
        return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
