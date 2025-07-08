import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from agent_core import agent_executor, AgentState

app = FastAPI(title="KYC Document Extraction Agent", version="1.0.0")

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "KYC Agent API is running."}

@app.post("/v1/extract", tags=["KYC Extraction"])
async def extract_kyc_data(
    image: UploadFile = File(..., description="The image file of the KYC document."),
    external_transaction_id: Optional[str] = Form(None, description="Optional external ID for tracking.")
):
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Image file is empty.")

        initial_state = {
            "image_bytes": image_bytes,
            "source_filename": image.filename,
            "external_transaction_id": external_transaction_id,
            "status": "PROCESSING",
        }

        print(f"Invoking agent for file: {image.filename}...")
        final_state = agent_executor.invoke(initial_state)
        print("Agent finished processing.")

        status = final_state.get("status", "ERROR_UNKNOWN")
        db_id = final_state.get("db_document_id")

        if status == "SUCCESS":
            return JSONResponse(
                status_code=200,
                content={"status": "SUCCESS", "data": final_state.get("structured_data"), "database_record_id": db_id},
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"status": status, "message": final_state.get("error_message"), "database_record_id": db_id},
            )
    except Exception as e:
        print(f"An unexpected error occurred in the API layer: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)