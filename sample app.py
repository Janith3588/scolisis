Certainly, I can help you create the app.py file for the API using FastAPI. Below is the code for app.py file. Here I have assumed that you have all your models and other required python files in the current directory. This code can be modified according to the actual structure of your files.

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import os

app = FastAPI()

@app.post("/run_test_phase")
async def run_test_phase(weight_path:str, data_path:str):
    try:
        # Running the command to run test phase with given weightPath and dataPath
        os.system(f'python main.py --resume {weight_path} --data_dir {data_path} --dataset spinal --phase test')
        return {"status": "Test phase ran successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error occurred while running the test phase.")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"message": f"{exc.detail}"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

After creating the above app.py, you can run it using this command:
```shell
uvicorn app:app --reload
```

This will run your FastAPI application. You can use Postman or any other API client to make a POST request to `http://127.0.0.1:8000/run_test_phase` with `weight_path` and `data_path` as form-data or json and it will run the command for the test phase.