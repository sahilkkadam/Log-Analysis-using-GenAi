## Log-Analysis-using-GenAI

Developed an AI system that uses Generative AI to detect anomalies, extract insights, and summarize log data. Implemented data preprocessing, log parsing, and AI-driven pattern recognition to enhance monitoring and troubleshooting efficiency. Utilized technologies like Python, NLP, LLM (LLaMA), ChromaDb, Streamlit, LangChain(Framework) for scalable log processing.

**Key Features:**

* **Chat:** Ask questions and get answers directly from your documents.
* **Local LLM Support:** Leverage the capabilities of local LLMs for offline document interaction.
* **ChromaDB Support:** Store and manage document metadata efficiently with ChromaDB.
* **LangChain Integration:** Streamline information extraction from documents through LangChain.

## Installation

**Prerequisites:**

* Python >=3.9 (https://www.python.org/downloads/)
* pip (usually comes pre-installed with Python)
* Ollama (https://ollama.com/)

**Installation Steps:**

1. **Clone the repository:**

   ```
   git clone https://github.com/gagangupta12/Log-Analysis-using-GenAI
   ```
2. **Navigate to the project directory:**

   ```
   cd Log-Analysis-using-GenAI
   ```

3. **Open project directory in VSCode**

   ```
   code .
   ```
or any other code editor

4. **Install dependencies from requirements.txt**
   ```
   pip install -r requirements.txt
   ```

5. **Pull the required models from Ollama**
   
   - Download & install [Ollama](https://ollama.com/) if not installed
   - Open terminal & run these command to pull the required models into local machine
     
     For `llama3`
     ```
     ollama pull llama3:latest
     ```    
     For `nomic-embed-text`
     ```
     ollama pull nomic-embed-text:latest
     ``` 
   - Insure both models are downloaded
     ```
     ollama ls
     ```
   - Check if ollama is running
     ```
     ollama serve
     ```
     go to localhost:11434 and you should see
     ollama is running

     ![image](https://github.com/user-attachments/assets/842761f0-8641-4ada-9aa0-32baac84500f)


6. **Start chromadb server**

   - Go to backend directory
   ```
   cd backend
   ```
   
   - Create db folder for storing Chromadb files
   ```
   mkdir db
   ```
   
   - Start Chromadb server:
   ```
   chroma run --path db --port 8001
   ```

   ![image](https://github.com/user-attachments/assets/59da7a0b-aedb-4a54-9bc7-92032758e653)

7. **Backend**
   - Open new terminal and go into backend folder & Run backend server:
   ```
   python backend.py
   ```
   - The command is used to starts a FastAPI application using the Uvicorn ASGI server, binding it to localhost (127.0.0.1) on port 8000.(Optional)
   ```
   uvicorn backend:app --host 127.0.0.1 --port 8000
   ```

8. **Frontend**

   - Open new terminal and go to frontend folder
   ```
   cd frontend
   ```
   - Run frontend.py
   ```
   streamlit run frontend.py
   ```
   ![Frontend img](https://github.com/user-attachments/assets/5f3949c1-f663-4a99-8c69-df734cf19228)

   - User Interface:
   
   ![Screenshot 2025-02-08 144642](https://github.com/user-attachments/assets/aaf7aa3c-401f-4779-87fa-826908f24224)



   


