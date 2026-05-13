from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config import API_TITLE, API_VERSION, HOST, PORT
from model_loader import model_loader
from routes import health_router, predict_router

app = FastAPI(title=API_TITLE, version=API_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(health_router)
app.include_router(predict_router)

@app.get("/")
def root():
    return {"name": API_TITLE, "version": API_VERSION, "status": "running"}

@app.on_event("startup")
def startup():
    print("Starting API...")
    model_loader.load()
    print(f"API running at http://{HOST}:{PORT}")

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)