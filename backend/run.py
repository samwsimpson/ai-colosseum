import uvicorn

if __name__ == "__main__":
    # Correctly pass the application as an import string
    uvicorn.run("api:app", host="0.0.0.0", port=8000)