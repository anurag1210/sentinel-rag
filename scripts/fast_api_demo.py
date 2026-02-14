from fastapi import FastAPI

app = FastAPI()

@app.get("/add")

def add(a: int, b: int):
    return {"echo": f"Adding {a} and {b}", "result": a + b}