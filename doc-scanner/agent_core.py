import os
import io
from datetime import datetime
from typing import TypedDict, Dict, Any, Optional

from dotenv import load_dotenv
import pymongo
from rapidocr_onnxruntime import RapidOCR
import cv2
import numpy as np

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

load_dotenv()

try:
    print("Initializing clients...")
    ocr_engine = RapidOCR()
    
    mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
    db = mongo_client[os.getenv("MONGO_DB_NAME")]
    collection = db[os.getenv("MONGO_COLLECTION_NAME")]
    mongo_client.admin.command('ping')
    print("MongoDB Atlas connection successful.")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"))
    print("Clients initialized successfully.")

except Exception as e:
    print(f"FATAL: Failed to initialize clients. Error: {e}")

class AgentState(TypedDict):
    image_bytes: bytes
    source_filename: str
    external_transaction_id: Optional[str]
    raw_ocr_text: Optional[str]
    structured_data: Optional[Dict[str, Any]]
    status: str
    error_message: Optional[str]
    db_document_id: Optional[str]

def ocr_node(state: AgentState) -> Dict[str, Any]:
    print("---NODE: Performing OCR---")
    try:
        np_arr = np.frombuffer(state['image_bytes'], np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image.")
        
        result, _ = ocr_engine(img)
        raw_text = "\n".join([line[1] for line in result]) if result else ""
        
        print(f"OCR successful. Extracted text snippet: {raw_text[:100]}...")
        return {"raw_ocr_text": raw_text, "status": "PROCESSING"}
    except Exception as e:
        print(f"ERROR in OCR node: {e}")
        return {"raw_ocr_text": "", "status": "ERROR_OCR", "error_message": f"Failed during OCR: {str(e)}"}

def extraction_node(state: AgentState) -> Dict[str, Any]:
    print("---NODE: Extracting with Gemini---")
    try:
        raw_text = state['raw_ocr_text']
        if not raw_text or len(raw_text.strip()) < 10:
            raise ValueError("Insufficient text from OCR for extraction.")

        prompt_template = """You are an expert AI assistant for extracting information from Indian KYC documents.
        Analyze the OCR text and extract: name, dob (in YYYY-MM-DD format), address, and document_type ('Aadhaar Card', 'PAN Card', etc.).
        OCR TEXT: --- {ocr_text} ---
        Respond ONLY with a valid JSON object with keys "document_type", "name", "dob", "address". If a field is not found, its value should be null."""
        
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        json_parser = JsonOutputParser()
        chain = prompt | llm | json_parser
        structured_response = chain.invoke({"ocr_text": raw_text})
        
        print(f"Gemini extraction successful: {structured_response}")
        return {"structured_data": structured_response}
    except Exception as e:
        print(f"ERROR in extraction node: {e}")
        return {"structured_data": {}, "status": "ERROR_EXTRACTION", "error_message": f"Failed during Gemini extraction: {str(e)}"}

def persistence_node(state: AgentState) -> Dict[str, Any]:
    print("---NODE: Saving to Database---")
    try:
        final_status = "SUCCESS" if state['status'] == "PROCESSING" else state['status']
        document_to_save = {
            "external_transaction_id": state.get("external_transaction_id"),
            "status": final_status,
            "source_filename": state.get("source_filename"),
            "raw_ocr_text": state.get("raw_ocr_text"),
            "error_message": state.get("error_message"),
            "created_at": datetime.utcnow()
        }
        if final_status == "SUCCESS":
            document_to_save["extracted_data"] = state.get("structured_data")

        result = collection.insert_one(document_to_save)
        db_id = str(result.inserted_id)
        print(f"Successfully saved record to MongoDB Atlas with ID: {db_id}")
        return {"db_document_id": db_id, "status": final_status}
    except Exception as e:
        print(f"CRITICAL ERROR in persistence node: {e}")
        return {"db_document_id": None, "status": "ERROR_DATABASE", "error_message": f"Critical: Failed to save to MongoDB. Reason: {str(e)}"}

def decide_after_ocr(state: AgentState) -> str:
    print("---EDGE: Deciding after OCR---")
    return "extract_details" if state["status"] != "ERROR_OCR" else "save_to_db"

print("Assembling the agent graph...")
workflow = StateGraph(AgentState)
workflow.add_node("perform_ocr", ocr_node)
workflow.add_node("extract_details", extraction_node)
workflow.add_node("save_to_db", persistence_node)
workflow.set_entry_point("perform_ocr")
workflow.add_conditional_edges("perform_ocr", decide_after_ocr, {"extract_details": "extract_details", "save_to_db": "save_to_db"})
workflow.add_edge("extract_details", "save_to_db")
workflow.add_edge("save_to_db", END)
agent_executor = workflow.compile()
print("Agent graph compiled successfully.")