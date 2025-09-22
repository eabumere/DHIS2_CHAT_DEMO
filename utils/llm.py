# utils/llm.py

from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables only once
load_dotenv()

def get_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_4O"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        model="gpt-4",
        temperature=0
    )
