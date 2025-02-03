import os
import shutil
import json
import hashlib
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import ollama
from typing import List, Generator
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd
import docx

# Class model for the request body
class Data(BaseModel):
    """
    Model to define the request body for asking questions.

    Attributes:
        question (str): The question to be answered.
        file_names (List[str]): List of file names to search for answers.
    """
    question: str 
    file_names: List[str] 


# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins (development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to create necessary directories
def create_directory(directory_path):
    """
    Creates a directory if it doesn't exist.

    Args:
        directory_path (str): Path of the directory to be created.
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory created at: {directory_path}")
    except OSError as e:
        print(f"Error creating directory: {e}")

# Save uploaded files locally
@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file and saves it locally.

    Args:
        file (UploadFile): The file to be uploaded.

    Returns:
        dict: Contains the filename and file path.
    """
    try:
        upload_directory = 'uploads'
        create_directory(upload_directory)
        file_path = os.path.join(upload_directory, file.filename)
        
        # Save the uploaded file to the specified directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"filename": file.filename, "file_path": file_path}
    except Exception as e:
        return {"error": str(e)}

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF file.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        text = "\n".join(page.page_content for page in pages)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_path):
    """
    Extracts text from a DOCX file.

    Args:
        docx_path (str): Path to the DOCX file.

    Returns:
        str: Extracted text from the DOCX file.
    """
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

# Function to extract text from a TXT file
def extract_text_from_txt(txt_path):
    """
    Extracts text from a TXT file.

    Args:
        txt_path (str): Path to the TXT file.

    Returns:
        str: Extracted text from the TXT file.
    """
    try:
        loader = TextLoader(txt_path)
        documents = loader.load()
        text = "\n".join(doc.page_content for doc in documents)
        return text
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        return ""

# Function to extract text from a CSV file
def extract_text_from_csv(csv_path):
    """
    Extracts text from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        str: Extracted text from the CSV file.
    """
    try:
        loader = CSVLoader(file_path=csv_path)
        data = loader.load()
        text = "\n".join([str(record) for record in data])
        return text
    except Exception as e:
        print(f"Error extracting text from CSV: {e}")
        return ""

# Function to extract text from an XLSX file
def extract_text_from_xlsx(xlsx_path):
    """
    Extracts text from an XLSX file.

    Args:
        xlsx_path (str): Path to the XLSX file.

    Returns:
        str: Extracted text from the XLSX file.
    """
    try:
        df = pd.read_excel(xlsx_path)
        text = df.to_string(index=False)
        return text
    except Exception as e:
        print(f"Error extracting text from XLSX: {e}")
        return ""

# Function to get file extension
def get_file_extension(file_path):
    """
    Gets the extension of a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: File extension in lowercase.
    """
    try:
        return os.path.splitext(file_path)[1].lower()
    except Exception as e:
        print(f"Error getting file extension: {e}")
        return ""

# Function to extract text based on file type
def extract_text(file_path):
    """
    Extracts text from a file based on its extension.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Extracted text from the file.
    """
    try:
        ext = get_file_extension(file_path)
        if ext == '.pdf':
            return extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return extract_text_from_docx(file_path)
        elif ext == '.txt':
            return extract_text_from_txt(file_path)
        elif ext == '.csv':
            return extract_text_from_csv(file_path)
        elif ext == '.xlsx':
            return extract_text_from_xlsx(file_path)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""
