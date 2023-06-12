from fastapi import FastAPI, UploadFile, File
import subprocess

app = FastAPI()

@app.post("/process_image")
async def process_image(image: UploadFile = File(...)):
    # Save the uploaded image to a file
    image_path = "input_image.jpg"
    with open(image_path, "wb") as f:
        f.write(await image.read())

    # Run the test phase using the uploaded image
    command = f"python main.py --resume weights_spinal\model_last.pth --data_dir {image_path} --dataset spinal --phase test"
    subprocess.run(command, shell=True)

    # Read the output image
    output_image_path = "output_image.jpg"
    with open(output_image_path, "rb") as f:
        output_image = f.read()

    # Read the Cobb angle details from the terminal output
    cobb_angle_output = "Sample Cobb Angle: 60 degrees"  # Replace this with actual output

    # Return the output image and Cobb angle details
    return {
        "output_image": output_image,
        "cobb_angle": cobb_angle_output
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
