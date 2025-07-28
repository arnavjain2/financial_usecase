from dotenv import load_dotenv
import google.generativeai as genai
import os
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
print('**')
model = genai.GenerativeModel("gemini-2.5-pro")
print("****")
prompt = "can u code"
response = model.generate_content(prompt)
print("****")