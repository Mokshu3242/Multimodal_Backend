# Standard library imports
import asyncio
import base64
import datetime
import hashlib
import io
import json
import logging
import os
import random
import re
import smtplib
import stat
import string
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
import wave
from collections import defaultdict
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
from pathlib import Path
from typing import Annotated, Dict, List, Optional
from uuid import uuid4

# Third-party library imports
import bcrypt
import docx2txt
import jwt
import msoffcrypto
import olefile
import pandas as pd
import plotly.express as px
import pymongo
import pypdfium2
import requests
import seaborn as sns
import speech_recognition as sr
import uvicorn
import yaml
from docx import Document
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from matplotlib import pyplot as plt
from PIL import Image
from pydantic import BaseModel
from pypdfium2 import PdfDocument
from slowapi import Limiter
from slowapi.util import get_remote_address
from tabulate import tabulate
from youtube_transcript_api import YouTubeTranscriptApi

# --------------------------------------------------------------------
# Environment and Configuration
load_dotenv()

# Cloudflare configuration
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
AUTH_TOKEN = os.getenv("CLOUDFLARE_AUTH_TOKEN")
API_ID = os.getenv("CLOUDFLARE_AI_API")

# ElevenLabs configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ElevenLabs_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Database configuration
MONGO_CONN = os.environ.get("MONGO_DB_CONNECT")
mongo_client = pymongo.MongoClient(MONGO_CONN)
mydb = mongo_client["multigpt"]
users_db = mydb["users"]
history_db = mydb["history"]
documents_db = mydb["document_chunks"]

# Security configuration
JWT_KEY = os.getenv("JWT_SECRET", "RagModelProjt")

# Model configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
models = config["llm_model"]
image_model = config["Image_model"]
whisper_model = config["whisper_model"]

# --------------------------------------------------------------------
# Application Setup
app = FastAPI()

# Configures CORS (Cross-Origin Resource Sharing) for the FastAPI app
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://192.168.0.145:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://192.168.0.212:5173",
    "http://192.168.0.235",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Initializes an in-memory chat history
chat_histories = {}


# --------------------------------------------------------------------
# Classes
class DocumentData(BaseModel):
    file_name: str
    content: str


class ChatInput(BaseModel):
    chat_id: str
    input_text: str
    transcribe: Optional[str] = None
    document: Optional[DocumentData] = None


class UserRegister(BaseModel):
    name: str
    email: str
    password: str
    profilePic: Optional[str] = None


class ChatPDFInput(BaseModel):
    chat_id: str
    input_text: str
    doc_path: str


class UserLogin(BaseModel):
    email: str
    password: str


class UserProfile(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    profilePic: Optional[str] = None


class OTPRequest(BaseModel):
    email: str


class OTPVerification(BaseModel):
    email: str
    otp: str


# ------------------------------------------------------------------
# Helper function for making safe Cloudflare API requests with error handling
def safe_cloudflare_request(url, headers, payload):
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        json_data = response.json()

        if "result" in json_data and "response" in json_data["result"]:
            return json_data["result"]["response"]
        else:
            logger.error("Unexpected API response format: %s", json_data)
            return "Error: Unexpected API response format."

    except requests.exceptions.RequestException as e:
        logger.error("API request failed: %s", e)
        if e.response is not None:
            logger.error("Status code: %s", e.response.status_code)
            logger.error("Response text: %s", e.response.text)

        return "Error: Failed to process request."


# Helper function to verify and decode a JWT token
def verify_jwt_token(token: str | None):
    if token is None:
        return None
    try:
        claims = jwt.decode(
            token,
            algorithms=["HS256"],
            key=JWT_KEY,
        )
        return claims
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        return None
    except jwt.InvalidTokenError:
        logger.error("Invalid token")
        return None


# Helper function to hash passwords
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed_password.decode("utf-8")


# Helper function to check if a user already exists
def user_exists(email: str) -> bool:
    return users_db.find_one({"email": email}) is not None


# Helper function to verify password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


# Helper function to generate JWT token
def generate_jwt_token(user_id: str) -> str:
    payload = {
        "userId": user_id,
        "exp": datetime.utcnow() + timedelta(hours=7),  # Token expires in 7 hour
    }
    token = jwt.encode(payload, JWT_KEY, algorithm="HS256")
    return token


# ------------------------------------------------------------------
def transcribe_audio_with_whisper(audio_bytes: bytes) -> str:
    try:
        # Encode the audio bytes as base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        response = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{whisper_model}",
            headers={
                "Authorization": f"Bearer {AUTH_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "audio": audio_base64,
                "task": "transcribe",  # or "translate" for English translation
            },
        )
        response.raise_for_status()
        return response.json()["result"]["text"]
    except Exception as e:
        logger.error(f"Whisper transcription failed: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        return None


# ------------------------------------------------------------------
# Extracts text from a PDF file using pypdfium2
def extract_text_from_pdf(pdf_bytes):
    if not pdf_bytes:
        return "Error: Empty PDF file."
    try:
        pdf_file = pypdfium2.PdfDocument(pdf_bytes)
        text = "\n".join(
            pdf_file.get_page(page_number).get_textpage().get_text_range()
            for page_number in range(len(pdf_file))
        )
        return text
    except Exception as e:
        logger.error("Error extracting text from PDF: %s", e)
        return f"Error processing PDF: {e}"


# Extracts text from multiple PDFs
def get_pdf_texts(pdfs_bytes_list):
    return [extract_text_from_pdf(pdf_bytes) for pdf_bytes in pdfs_bytes_list]


# Splits text into manageable chunks for processing
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=50, separators=["\n", "\n\n"]
    )
    return splitter.split_text(text)


# Converts extracted text into structured document chunks
def get_document_chunks(text_list):
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            documents.append(Document(page_content=chunk))
    return documents


# Extracts text from an Excel file using pandas
def extract_text_from_excel(excel_path):
    try:
        df = pd.read_excel(excel_path, engine="openpyxl")
        return df.to_string()
    except Exception as e:
        logger.error("Error extracting text from Excel file: %s", e)
        return ""


# Decrypts a password-protected DOCX file
def decrypt_docx(file_path, password, output_path):
    try:
        with open(file_path, "rb") as f:
            file = msoffcrypto.OfficeFile(f)
            file.load_key(password=password)
            with open(output_path, "wb") as decrypted_file:
                file.decrypt(decrypted_file)
        logger.info(f"Decrypted file saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to decrypt file: {e}")
        return None


# Extracts text from a DOCX file, handling password protection and read-only mode
def extract_text_from_docx(doc_path, password=None):
    try:
        if not Path(doc_path).exists():
            logger.error(f"File not found: {doc_path}")
            return ""

        if is_docx_protected(doc_path):
            logger.warning(f"File is edit-protected: {doc_path}")

            if is_read_only(doc_path):
                remove_read_only(doc_path)

            if password:
                decrypted_path = f"{doc_path}_decrypted.docx"
                decrypt_docx(doc_path, password, decrypted_path)
                doc_path = decrypted_path
            else:
                logger.error("Password is required for protected file.")
                return ""

        text = docx2txt.process(doc_path)
        logger.info(f"Successfully extracted text from DOCX file: {doc_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from Word document: {e}")
        return ""


def add_to_db(doc_path, chat_id, user_id):
    """Stores document chunks in MongoDB only."""
    try:
        with open(doc_path, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        logger.error("Error reading file: %s", e)
        return f"Error: {e}"

    file_extension = Path(doc_path).suffix.lower()
    file_name = os.path.basename(doc_path)

    # Extract text based on file type
    if file_extension == ".pdf":
        texts = get_pdf_texts([file_bytes])
    elif file_extension in {".doc", ".docx"}:
        texts = [extract_text_from_docx(doc_path)]
    elif file_extension in {".xls", ".xlsx"}:
        texts = [extract_text_from_excel(doc_path)]
    elif file_extension == ".txt":
        with open(doc_path, "r", encoding="utf-8") as f:
            texts = [f.read()]
    else:
        logger.error("Unsupported file type: %s", file_extension)
        return f"Error: Unsupported file type {file_extension}"

    documents = get_document_chunks(texts)

    if not documents:
        logger.error("No valid document chunks found.")
        return "Error: No valid data extracted from file."

    # Store each chunk in MongoDB
    chunk_ids = []
    for doc in documents:
        chunk_id = str(uuid4())
        documents_db.insert_one(
            {
                "_id": chunk_id,
                "chat_id": chat_id,
                "user_id": user_id,
                "file_name": file_name,
                "file_path": doc_path,
                "content": doc.page_content,
                "created_at": datetime.now(),
                "expires_at": datetime.now()
                + timedelta(days=2),  # Auto-expire in 2 days
            }
        )
        chunk_ids.append(chunk_id)

    logger.info(
        "Successfully added %s document chunks from file: %s (Path: %s) to MongoDB.",
        len(documents),
        file_name,
        doc_path,
    )

    return texts


# Checks if a DOCX file is password-protected or read-only
def is_docx_protected(file_path):
    try:
        with olefile.OleFileIO(file_path) as ole:
            if ole.exists("EncryptionInfo"):
                return True
            if ole.exists("WordDocument"):
                stream = ole.openstream("WordDocument")
                data = stream.read()
                if data[0x0B] & 0x01:
                    return True
    except Exception as e:
        logger.error(f"Error checking file protection: {e}")
    return False


# Checks if a file is read-only
def is_read_only(file_path):
    try:
        file_stat = os.stat(file_path)
        return not bool(file_stat.st_mode & stat.S_IWUSR)
    except Exception as e:
        logger.error(f"Error checking read-only status for file {file_path}: {e}")
        return False


# Removes the read-only flag from a file
def remove_read_only(file_path):
    try:
        os.chmod(file_path, stat.S_IWRITE)
        logger.info(f"Removed read-only flag for file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to remove read-only flag: {e}")


# Extracts text from a file based on its type
def extract_text_from_file(file_path):
    try:
        file_extension = Path(file_path).suffix.lower()

        if file_extension == ".pdf":
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            return extract_text_from_pdf(file_bytes)
        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif file_extension in {".xls", ".xlsx"}:
            return extract_text_from_excel(file_path)
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from file {file_path}: {e}")
        return ""


# ------------------------------------------------------------------
# Converts an image to an integer array (byte representation)
def image_to_int_array(image, format="JPEG"):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return list(buffer.getvalue())


# Handles image processing, ensuring correct format and saving the image
def handle_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))

        if img.mode == "RGBA":
            img = img.convert("RGB")

        img.save("output.jpg", format="JPEG")

        return "Image processed successfully."
    except Exception as e:
        logger.error(f"Error in handle_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Converts structured text (key-value pairs, lists, JSON, CSV) into a Markdown table
def ensure_table_format(response_text: str) -> str:
    if "|" in response_text and "-" in response_text:
        return response_text

    try:
        lines = response_text.split("\n")
        data = []

        if any(":" in line for line in lines):
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    data.append([key.strip(), value.strip()])
            if data:
                return tabulate(data, headers=["Key", "Value"], tablefmt="pipe")

        if len(lines) > 1 and all(":" not in line for line in lines):
            data = [[item.strip()] for item in lines if item.strip()]
            if data:
                return tabulate(data, headers=["Item"], tablefmt="pipe")

        try:
            parsed_data = json.loads(response_text)
            if isinstance(parsed_data, dict):
                table_data = [[k, v] for k, v in parsed_data.items()]
                return tabulate(table_data, headers=["Key", "Value"], tablefmt="pipe")
            elif isinstance(parsed_data, list) and all(
                isinstance(item, dict) for item in parsed_data
            ):
                headers = parsed_data[0].keys()
                table_data = [list(item.values()) for item in parsed_data]
                return tabulate(table_data, headers=headers, tablefmt="pipe")
        except json.JSONDecodeError:
            pass

        if "," in response_text and "\n" in response_text:
            rows = [line.split(",") for line in lines if line.strip()]
            if len(rows) > 1:
                headers, table_data = rows[0], rows[1:]
                return tabulate(table_data, headers=headers, tablefmt="pipe")

        return response_text

    except Exception as e:
        logger.error(f"Error formatting table: {e}")
        return response_text


# Extracts a Markdown table from a given text if present
def extract_markdown_table(text: str) -> str:
    table_pattern = r"(\|.*\|\n\|.*\|\n(\|.*\|\n)+)"
    match = re.search(table_pattern, text)
    return match.group(0).strip() if match else text


# ------------------------------------------------------------------
# Visualization
def generate_visualization(
    data: Dict | List,
    chart_type: str,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> str:
    try:
        plt.figure(figsize=(10, 6))
        sns.set_theme()
        plt.title(title, pad=20)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if chart_type == "bar":
            if isinstance(data, dict):
                plt.bar(data.keys(), data.values(), color="#4C72B0")
            else:  # list
                plt.bar(range(len(data)), data, color="#4C72B0")
            plt.xticks(rotation=45)

        elif chart_type == "line":
            if isinstance(data, dict):
                plt.plot(
                    list(data.keys()),
                    list(data.values()),
                    marker="o",
                    color="#55A868",
                    linewidth=2.5,
                )
            else:
                plt.plot(data, marker="o", color="#55A868", linewidth=2.5)

        elif chart_type == "pie":
            if isinstance(data, dict):
                plt.pie(
                    data.values(),
                    labels=data.keys(),
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=plt.cm.Pastel1.colors,
                )
            else:
                plt.pie(
                    data, autopct="%1.1f%%", startangle=90, colors=plt.cm.Pastel1.colors
                )
            plt.axis("equal")

        elif chart_type == "scatter":
            if isinstance(data, dict) and "x" in data and "y" in data:
                plt.scatter(data["x"], data["y"], color="#C44E52", s=100)
            elif isinstance(data, list) and all(
                isinstance(x, (list, tuple)) for x in data
            ):
                x, y = zip(*data)
                plt.scatter(x, y, color="#C44E52", s=100)

        elif chart_type == "histogram":
            values = list(data.values()) if isinstance(data, dict) else data
            plt.hist(
                values, bins=min(10, len(values)), color="#8172B2", edgecolor="black"
            )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return image_base64

    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        return None


# ------------------------------------------------------------------
# OTP
class OTPService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.email_address = os.getenv("EMAIL_ADDRESS")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        self.otp_expiry_minutes = 5
        self.max_attempts = 3
        self.otp_collection = mydb["otps"]

        if not all(
            [self.smtp_server, self.smtp_port, self.email_address, self.email_password]
        ):
            logger.error("Missing email configuration in environment variables")
            raise ValueError("Email service configuration is incomplete")

    def generate_otp(self, length=6) -> str:
        return "".join(
            random.SystemRandom().choice(string.digits) for _ in range(length)
        )

    def _hash_otp(self, otp: str) -> str:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(otp.encode("utf-8"), salt).decode("utf-8")

    async def send_otp_email(self, email: str, otp: str) -> bool:
        try:
            message = MIMEMultipart()
            message["From"] = self.email_address
            message["To"] = email
            message["Subject"] = "Your Verification Code for MultiGPT"

            html = f"""<html><body>
                <h2>Your Verification Code</h2>
                <p>Your OTP code is: <strong>{otp}</strong></p>
                <p>Valid for {self.otp_expiry_minutes} minutes.</p>
                </body></html>"""

            message.attach(MIMEText(html, "html"))

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_smtp_message, message)
            return True

        except Exception as e:
            logger.error(f"Failed to send OTP email to {email}: {str(e)}")
            return False

    def _send_smtp_message(self, message: MIMEMultipart):
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(self.email_address, self.email_password)
            server.send_message(message)

    async def store_otp(self, email: str, otp: str):
        try:
            expiry_time = datetime.utcnow() + timedelta(minutes=self.otp_expiry_minutes)
            hashed_otp = self._hash_otp(otp)

            # Remove await from these synchronous operations
            self.otp_collection.update_many(
                {"email": email, "is_used": False}, {"$set": {"is_used": True}}
            )

            self.otp_collection.insert_one(
                {
                    "email": email,
                    "otp_hash": hashed_otp,
                    "created_at": datetime.utcnow(),
                    "expires_at": expiry_time,
                    "is_used": False,
                    "attempts": 0,
                }
            )

        except Exception as e:
            logger.error(f"Failed to store OTP for {email}: {str(e)}")
            raise

    async def verify_otp(self, email: str, otp: str) -> bool:
        try:
            record = self.otp_collection.find_one(
                {
                    "email": email,
                    "is_used": False,
                    "expires_at": {"$gt": datetime.utcnow()},
                }
            )

            if not record:
                logger.warning(f"No active OTP found for {email}")
                self._increment_attempts(email)
                return False

            if not bcrypt.checkpw(
                otp.encode("utf-8"), record["otp_hash"].encode("utf-8")
            ):
                logger.warning(f"Invalid OTP attempt for {email}")
                self._increment_attempts(email)
                return False

            self.otp_collection.update_one(
                {"_id": record["_id"]}, {"$set": {"is_used": True}}
            )
            return True

        except Exception as e:
            logger.error(f"OTP verification failed for {email}: {str(e)}")
            return False

    def _increment_attempts(self, email: str):
        self.otp_collection.update_many(
            {"email": email, "is_used": False}, {"$inc": {"attempts": 1}}
        )

    async def cleanup_expired_otps(self):
        try:
            result = self.otp_collection.delete_many(
                {"expires_at": {"$lt": datetime.utcnow()}}
            )
            logger.info(f"Cleaned up {result.deleted_count} expired OTPs")
        except Exception as e:
            logger.error(f"Failed to clean up expired OTPs: {str(e)}")


# ------------------------------------------------------------------
# Extract Youtube transcript
def extract_youtube_transcript(youtube_video_url):
    try:
        video_id = youtube_video_url.split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        transcript_text = ""
        for i in transcript:
            transcript_text += "" + i["text"]

        return transcript_text

    except Exception as e:
        logger.error(f"Error extracting YouTube transcript: {e}")
        return None


# ------------------------------------------------------------------
# Routes
otp_service = OTPService()


@app.get("/")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/v1/otp/request")
@limiter.limit("1/minute")
async def request_otp(request: Request, otp_request: OTPRequest):
    if not re.match(r"[^@]+@[^@]+\.[^@]+", otp_request.email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    otp = otp_service.generate_otp()

    try:
        await otp_service.store_otp(otp_request.email, otp)
        if not await otp_service.send_otp_email(otp_request.email, otp):
            raise HTTPException(status_code=500, detail="Failed to send OTP")

        await otp_service.cleanup_expired_otps()
        return {
            "message": "OTP sent",
            "expires_in": f"{otp_service.otp_expiry_minutes} minutes",
        }
    except Exception as e:
        logger.error(f"OTP request failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process OTP request")


@app.post("/api/v1/otp/verify")
async def verify_otp(otp_verification: OTPVerification):
    try:
        if not await otp_service.verify_otp(
            otp_verification.email, otp_verification.otp
        ):
            raise HTTPException(status_code=400, detail="Invalid OTP")
        return {"message": "OTP verified", "email": otp_verification.email}
    except Exception as e:
        logger.error(f"OTP verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify OTP")


@app.post("/api/v1/user/auth/login", status_code=status.HTTP_200_OK)
async def login_user(user: UserLogin):
    user_data = users_db.find_one({"email": user.email})
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not verify_password(user.password, user_data["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token = generate_jwt_token(user_data["_id"])

    response = JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Welcome", "isAuthenticated": True},
    )
    response.set_cookie(
        key="authToken",
        value=token,
        path="/",
    )
    return response


@app.post("/api/v1/user/auth/register", status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: dict = Body(...),
):
    """Modified registration endpoint with hashed OTP verification"""
    try:
        otp = user_data.pop("otp", None)
        if not otp:
            raise HTTPException(status_code=400, detail="OTP is required")

        if not await otp_service.verify_otp(user_data["email"], otp):
            raise HTTPException(status_code=400, detail="Invalid OTP")

        if user_exists(user_data["email"]):
            raise HTTPException(status_code=400, detail="Email already exists")

        if len(user_data["password"]) < 8:
            raise HTTPException(status_code=400, detail="Password too short")

        user_data["_id"] = str(uuid4())
        user_data["password"] = hash_password(user_data["password"])
        user_data["created_at"] = datetime.utcnow()
        user_data["verified"] = True

        users_db.insert_one(user_data)

        otp_service.otp_collection.delete_many({"email": user_data["email"]})

        return {"message": "Registration successful"}

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.get("/api/v1/user/auth/profile", status_code=status.HTTP_200_OK)
async def get_profile(request: Request):
    token = verify_jwt_token(request.cookies.get("authToken"))
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token.",
        )

    user_data = users_db.find_one({"_id": token["userId"]})
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    return {
        "success": True,
        "user": {
            "name": user_data.get("name"),
            "email": user_data.get("email"),
            "profilePic": user_data.get("profilePic"),
            "language": user_data.get("language"),
        },
    }


@app.post("/api/v1/user/auth/update-language", status_code=status.HTTP_200_OK)
async def update_language(request: Request, language: str = Body(..., embed=True)):
    token = verify_jwt_token(request.cookies.get("authToken"))
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token.",
        )

    # Map short language codes to their full names
    language_map = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

    if language not in language_map:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid language.",
        )

    # Get the full language name
    language_full = language_map[language]

    result = users_db.update_one(
        {"_id": token["userId"]},
        {"$set": {"language": language_full}},  # Save the full language name
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    return {
        "success": True,
        "message": "Language updated successfully.",
    }


@app.put("/api/v1/user/auth/update", status_code=status.HTTP_200_OK)
async def update_profile(request: Request, updated_data: UserProfile):
    token = verify_jwt_token(request.cookies.get("authToken"))
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token.",
        )

    user_data = users_db.find_one({"_id": token["userId"]})
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    update_fields = {}
    if updated_data.name:
        update_fields["name"] = updated_data.name
    if updated_data.email:
        update_fields["email"] = updated_data.email
    if updated_data.profilePic:
        update_fields["profilePic"] = updated_data.profilePic

    if update_fields:
        users_db.update_one({"_id": token["userId"]}, {"$set": update_fields})

    updated_user = users_db.find_one({"_id": token["userId"]})
    return {
        "success": True,
        "user": {
            "name": updated_user.get("name"),
            "email": updated_user.get("email"),
            "profilePic": updated_user.get("profilePic"),
        },
    }


@app.delete("/api/v1/user/auth/delete-account", status_code=status.HTTP_200_OK)
async def delete_user_account(
    request: Request,
    password: str = Body(..., embed=True),
    otp: str = Body(..., embed=True),
):
    try:
        token = verify_jwt_token(request.cookies.get("authToken"))
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication token",
            )

        user_id = token["userId"]

        user_data = users_db.find_one({"_id": user_id})
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        if not verify_password(password, user_data["password"]):
            users_db.update_one(
                {"_id": user_id}, {"$inc": {"failed_login_attempts": 1}}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Incorrect password"
            )

        if not await otp_service.verify_otp(user_data["email"], otp):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired OTP"
            )

        with mongo_client.start_session() as session:
            with session.start_transaction():
                users_db.delete_one({"_id": user_id}, session=session)

                history_db.delete_many({"user_id": user_id}, session=session)

                files_db = mydb["files"]
                user_files = list(files_db.find({"user_id": user_id}, session=session))

                for file_meta in user_files:
                    try:
                        if os.path.exists(file_meta["file_path"]):
                            os.remove(file_meta["file_path"])
                    except Exception as file_error:
                        logger.error(
                            f"Failed to delete file {file_meta['file_path']}: {file_error}"
                        )
                        continue

                otp_service.otp_collection.delete_many({"email": user_data["email"]})

                session.commit_transaction()

        response = JSONResponse(
            content={
                "success": True,
                "message": "Account and all associated data deleted successfully",
                "deleted_at": datetime.utcnow().isoformat(),
            }
        )
        response.delete_cookie("authToken")

        return response

    except pymongo.errors.PyMongoError as db_error:
        logger.error(f"Database error during deletion: {db_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database operation failed during account deletion",
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting account",
        )


@app.get("/chats")
async def chats(request: Request):
    token = verify_jwt_token(request.cookies.get("authToken"))
    if len(token) == 0:
        return JSONResponse(content={"error": "Invalid Token"}, status_code=401)

    chats = []

    ittr_chat_history = list(
        history_db.aggregate(
            [
                {"$match": {"user_id": token["userId"]}},
                {"$sort": {"timestamp": -1}},
                {
                    "$group": {
                        "_id": "$chat_id",
                        "chat_id": {"$first": "$chat_id"},
                        "user_id": {"$first": "$user_id"},
                        "timestamp": {"$first": "$timestamp"},
                    }
                },
                {"$sort": {"timestamp": -1}},
            ]
        )
    )

    if ittr_chat_history != None:
        for chat in ittr_chat_history:
            if chat["chat_id"]:
                chats.append(chat["chat_id"])

    return JSONResponse(content={"chat_history": chats})


@app.get("/chat/{chat_id}")
async def chats(request: Request, chat_id: str):
    token = verify_jwt_token(request.cookies.get("authToken"))
    if len(token) == 0:
        return JSONResponse(content={"error": "Invalid Token"}, status_code=401)

    chat_history = []

    ittr_chat_history = history_db.find(
        {"chat_id": chat_id, "user_id": token["userId"]}
    ).sort("timestamp", pymongo.ASCENDING)

    if ittr_chat_history != None:
        for chat in ittr_chat_history:
            if chat["role"] == "assistant":
                chat_history.append(
                    {
                        "sender": "bot",
                        "content": chat["content"],
                        "img": chat.get("img", None),
                    }
                )
            else:
                chat_history.append(
                    {
                        "sender": "user",
                        "content": chat["content"],
                        "img": chat.get("img", None),
                    }
                )

    return JSONResponse(content={"chat_history": chat_history})


@app.post("/chat")
async def chat(input: ChatInput, request: Request):
    print("Received input payload:", input.dict())
    start = datetime.now()

    chat_id = input.chat_id
    input_text = input.input_text
    transcribe = input.transcribe
    document = input.document

    # Verify authentication
    token = verify_jwt_token(request.cookies.get("authToken"))
    if not token:
        raise HTTPException(status_code=401, detail="Invalid Token")

    user_data = users_db.find_one({"_id": token["userId"]})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found.")

    user_language = user_data.get("language")

    # Check for visualization keywords in user input
    visualization_keywords = [
        "chart",
        "graph",
        "visualize",
        "plot",
        "bar",
        "pie",
        "line",
        "histogram",
        "scatter",
    ]
    visualization_requested = any(
        keyword in input_text.lower() for keyword in visualization_keywords
    )

    # Handle YouTube URLs
    youtube_transcript = None
    youtube_url_pattern = (
        r"(https?:\/\/)?"
        r"((www\.|m\.)?"
        r"(youtube\.com|youtu\.be|youtube-nocookie\.com)\/)"
        r"(watch\?v=|embed\/|v\/|shorts\/|live\/|playlist\?list=|)"
        r'([^"&?\/\s]{11})'
        r'([^"\s]*)'
    )
    youtube_match = re.search(youtube_url_pattern, input_text)

    if youtube_match:
        youtube_url = youtube_match.group(0)
        youtube_transcript = extract_youtube_transcript(youtube_url)

        if youtube_transcript:
            if len(youtube_transcript) > 4000:
                youtube_transcript = youtube_transcript[:4000] + "... [truncated]"
            input_text = json.dumps(
                {
                    "url": youtube_url,
                    "transcript": youtube_transcript,
                    "question": input_text.replace(youtube_url, "").strip(),
                    "code": "ajdksadkashdahsdkaskdhaskdhk",
                }
            )
            # input_text = f"Here's a YouTube video transcript: {youtube_transcript}\n\nUser's question: {input_text.replace(youtube_url, '').strip()}"
        else:
            input_text = f"Couldn't extract transcript from YouTube video. {input_text}"

    # Store user message
    history_db.insert_one(
        {
            "chat_id": chat_id,
            "user_id": token["userId"],
            "content": input_text,
            "role": "user",
            "timestamp": datetime.now(),
            "file_data": "",
        }
    )

    # Enhanced system prompt with strict visualization controls
    system_content = f"""
    You are MultiGPT, a friendly AI assistant. Respond in {user_language} language unless specified otherwise and even if the user speak in any language use {user_language} language.
    
    Language:
    - English (en)
    - Hindi (hi)
    - Marathi (mr)
    - Use the language specified by the user or default to English if not specified.

    STRICT VISUALIZATION RULES:
    1. NEVER generate visualizations unless the user explicitly requests with clear phrases like:
       - "Show me a chart/graph of..."
       - "Visualize this data as..."
       - "Create a [bar/line/pie] chart for..."
    2. If visualization is not explicitly requested, provide ONLY text responses
    3. When explaining data, use text only unless visualization is requested
    
    YOUR CAPABILITIES:
    - Text processing in English, Hindi, and Marathi
    - Audio/image/document processing
    - Data visualization (ONLY when explicitly requested)
    - Code generation and debugging
    - Youtube video Processing
    
    RESPONSE FORMATTING:
    - For normal responses: Just provide the text answer.
    - For visualization requests ONLY and check if the user explicitly asked for it:
      [RESPONSE]
      <textual explanation>
      
      [VISUALIZATION]
      type: <chart_type>
      data: ```json
      <data_in_json_format>
      ```
      title: <chart_title>
      x_label: <x_axis_label>
      y_label: <y_axis_label>

    Example:
    [RESPONSE]
    Here's the sales data for Q1-Q3 showing steady growth:
    
    [VISUALIZATION]
    type: line
    data: ```json
    {{"Q1": 1000, "Q2": 1500, "Q3": 2000}}
    ```
    title: Quarterly Sales Growth
    x_label: Quarter
    y_label: Sales ($)

    """

    # Add document/transcript context if provided
    if document:
        system_content += f"\n\nDOCUMENT CONTEXT:\nFilename: {document.file_name}\nContent: {document.content}"
    if transcribe and transcribe.strip():
        system_content += f"\n\nTRANSCRIPT CONTEXT:\n{transcribe}"

    # Build chat history
    ittr_chat_history = history_db.find(
        {"chat_id": chat_id, "user_id": token["userId"]}
    ).sort("timestamp", pymongo.ASCENDING)

    chat_history = [{"role": "system", "content": system_content}]

    for chat in ittr_chat_history:
        if chat["file_data"] != "":
            chat["content"] = (
                chat["content"] + "\nFile Data:\n" + str(chat["file_data"])
            )
        if chat.get("img_data", None) is not None:
            chat["content"] += (
                f"\nVisualization data: {chat['img_data']}"
                if chat.get("img_data")
                else chat["content"]
            )
        chat_history.append({"role": chat["role"], "content": chat["content"]})

    print("Final chat history being sent to AI:", chat_history)

    # Call Cloudflare AI
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{models}"
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    payload = {"messages": chat_history, "max_tokens": 2000}

    ai_response = safe_cloudflare_request(url, headers, payload)
    print("AI response:", ai_response)

    # Process response
    response_text = ai_response
    visualization = None
    vis_data = None

    if "[VISUALIZATION]" in ai_response:
        if visualization_requested:
            try:
                parts = ai_response.split("[RESPONSE]")[-1].split("[VISUALIZATION]")
                if len(parts) >= 2:
                    text_part, vis_part = parts[0], parts[1]
                    response_text = text_part.strip()

                    vis_params = {
                        "type": re.search(r"type:\s*(.+)", vis_part).group(1).strip(),
                        "data": json.loads(
                            re.search(
                                r"data: ```json\n(.*?)\n```", vis_part, re.DOTALL
                            ).group(1)
                        ),
                        "title": re.search(r"title:\s*(.+)", vis_part).group(1).strip(),
                        "x_label": re.search(r"x_label:\s*(.+)", vis_part)
                        .group(1)
                        .strip(),
                        "y_label": re.search(r"y_label:\s*(.+)", vis_part)
                        .group(1)
                        .strip(),
                    }

                    visualization = generate_visualization(
                        data=vis_params["data"],
                        chart_type=vis_params["type"],
                        title=vis_params["title"],
                        x_label=vis_params["x_label"],
                        y_label=vis_params["y_label"],
                    )
                    vis_data = vis_params["data"]
            except Exception as e:
                logger.error(f"Visualization processing error: {e}")
                response_text = ai_response.split("[VISUALIZATION]")[0]
        else:
            # Remove visualization if not requested
            response_text = ai_response.split("[VISUALIZATION]")[0]
            logger.warning("Removed unsolicited visualization from response")

    # Format tables if requested
    if "table" in input_text.lower():
        response_text = ensure_table_format(response_text)
        response_text = extract_markdown_table(response_text)

    # Store assistant response
    history_db.insert_one(
        {
            "chat_id": chat_id,
            "user_id": token["userId"],
            "content": response_text,
            "img": visualization,
            "img_data": vis_data,
            "role": "assistant",
            "timestamp": datetime.now(),
            "file_data": "",
        }
    )

    end = datetime.now()
    total_time = end - start
    print(f"Total Duration: {total_time}")

    return JSONResponse(
        content={
            "chat_id": chat_id,
            "response": response_text,
            "img": visualization,
        }
    )


@app.post("/chat_doc")
async def chat_doc(
    input: ChatPDFInput,
    request: Request,
):
    start = datetime.now()
    try:
        chat_id = input.chat_id
        input_text = input.input_text
        doc_path = input.doc_path
        file_data = []

        # Authentication
        token = verify_jwt_token(request.cookies.get("authToken"))
        if len(token) == 0:
            return JSONResponse(content={"error": "Invalid Token"}, status_code=401)

        user_id = token["userId"]

        # Verify user exists
        user_data = users_db.find_one({"_id": user_id})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found.")

        user_language = user_data.get("language", "en")

        # Process document if provided
        if doc_path:
            file_data = add_to_db(doc_path, chat_id, user_id)

        # Store user message in history
        history_db.insert_one(
            {
                "chat_id": chat_id,
                "user_id": user_id,
                "content": input_text,
                "role": "user",
                "timestamp": datetime.now(),
                "file_data": file_data,
            }
        )

        # Build chat history
        chat_history = []
        ittr_chat_history = history_db.find(
            {"chat_id": chat_id, "user_id": user_id}
        ).sort("timestamp", pymongo.ASCENDING)

        for chat in ittr_chat_history:
            if chat["file_data"] != "":
                chat["content"] = (
                    chat["content"] + "\nFile Data:\n" + str(chat["file_data"])
                )
            chat_history.append({"role": chat["role"], "content": chat["content"]})

        # Get relevant document chunks from MongoDB
        vector_data = []
        if doc_path:
            document_chunks = (
                documents_db.find({"file_path": doc_path, "user_id": user_id})
                .sort("created_at", pymongo.ASCENDING)
                .limit(5)
            )  # Limit to 5 most relevant chunks
            vector_data = [chunk["content"] for chunk in document_chunks]

        system_content = f"""
        You are a friendly assistant that can generate both text responses and visualizations & your name is multigpt.
        The user's preferred language is {user_language}. Respond in {user_language} unless specified otherwise and even if the user speak in any language use {user_language} language.
    
        Language:
        - English (en)
        - Hindi (hi)
        - Marathi (mr)
        - Use the language specified by the user or default to English if not specified.


        when users ask about your functionality tell them audio, image, document and text processing. also tell them you can chat in primarily 3 languages english, marathi and hindi. also you can visualize stastical data.
        Don't generate visualizations unless user explicitly asks you to generate. STRICTLY If user asks to explain visualization data, provide a textual explanation don't generate visualization.

        When visualization is requested or appropriate, structure your response like this:
        
        [RESPONSE]
        <textual explanation of the data>
        
        [VISUALIZATION]
        type: <chart_type>
        data: ```json
        <data_in_json_format>
        ```
        title: <chart_title>
        x_label: <x_axis_label>
        y_label: <y_axis_label>

        Supported chart types:
        - bar (for comparisons between categories)
        - line (for trends over time)
        - pie (for showing proportions)
        - scatter (for relationships between variables)
        - histogram (for distribution of data)

        Rules for visualizations:
        1. Always provide a textual summary first
        2. Use the [VISUALIZATION] block only when appropriate
        3. Keep JSON data simple (avoid nested structures)
        4. Specify clear axis labels

        Example:
        [RESPONSE]
        Here's the sales data for Q1-Q3 showing steady growth:
        
        [VISUALIZATION]
        type: line
        data: ```json
        {{"Q1": 1000, "Q2": 1500, "Q3": 2000}}
        ```
        title: Quarterly Sales Growth
        x_label: Quarter
        y_label: Sales ($)
        """

        # Add document context if available
        if vector_data:
            system_content += f"\n\nDOCUMENT CONTEXT:\n{''.join(vector_data)}"

        # Prepare full chat history for AI
        messages = [{"role": "system", "content": system_content}] + chat_history

        # Call Cloudflare AI
        url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{models}"
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        payload = {"messages": messages, "max_tokens": 2000}

        ai_response = safe_cloudflare_request(url, headers, payload)
        print("AI response:", ai_response)

        # Process AI response
        response_text = ai_response
        visualization = None
        vis_data = None

        # Handle visualization if present
        if "[VISUALIZATION]" in ai_response:
            try:
                text_part, vis_part = ai_response.split("[RESPONSE]")[-1].split(
                    "[VISUALIZATION]"
                )
                response_text = text_part.strip()

                # Extract visualization parameters
                vis_type = re.search(r"type:\s*(.+)", vis_part).group(1).strip()
                vis_json = re.search(
                    r"data: ```json\n(.*?)\n```", vis_part, re.DOTALL
                ).group(1)
                vis_title = re.search(r"title:\s*(.+)", vis_part).group(1).strip()
                vis_x_label = re.search(r"x_label:\s*(.+)", vis_part).group(1).strip()
                vis_y_label = re.search(r"y_label:\s*(.+)", vis_part).group(1).strip()

                # Generate visualization
                visualization = generate_visualization(
                    data=json.loads(vis_json),
                    chart_type=vis_type,
                    title=vis_title,
                    x_label=vis_x_label,
                    y_label=vis_y_label,
                )
                vis_data = json.loads(vis_json)

            except Exception as e:
                logger.error(f"Visualization processing error: {e}")
                response_text = ai_response.split("[VISUALIZATION]")[0]

        # Format as table if requested
        if "table" in input_text.lower():
            response_text = ensure_table_format(response_text)
            response_text = extract_markdown_table(response_text)

        # Store assistant response
        history_db.insert_one(
            {
                "chat_id": chat_id,
                "user_id": user_id,
                "content": response_text,
                "img": visualization,
                "img_data": vis_data,
                "role": "assistant",
                "timestamp": datetime.now(),
                "file_data": "",
            }
        )

        # Prepare response
        response_data = {
            "chat_id": chat_id,
            "response": response_text,
            "img": visualization,
        }

        end = datetime.now()
        total_time = end - start
        print(f"Total Duration: {total_time}")

        return JSONResponse(content=response_data)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("Error in /chat_doc: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/handle_image")
async def handle_image(
    request: Request,
    image_path: UploadFile = File(...),
    user_message: str = Form(default=""),
    chat_id: str = Query(..., description="The ID of the chat session"),
):
    start = datetime.now()
    try:
        token = verify_jwt_token(request.cookies.get("authToken"))
        if len(token) == 0:
            return JSONResponse(content={"error": "Invalid Token"}, status_code=401)

        user_data = users_db.find_one({"_id": token["userId"]})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found.")

        user_language = user_data.get("language", "en")
        image_bytes = await image_path.read()
        img = Image.open(io.BytesIO(image_bytes))

        if img.mode == "RGBA":
            img = img.convert("RGB")

        img_array = image_to_int_array(img)

        prompt_text = (
            user_message.strip()
            if user_message.strip()
            else f"""Describe the image in detail in {user_language} and even if the user speak in any language use {user_language} language.
    
            Language:
            - English (en)
            - Hindi (hi)
            - Marathi (mr)
            - Use the language specified by the user or default to English if not specified.
    """
        )

        url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{image_model}"
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        payload = {"prompt": prompt_text, "image": img_array, "max_tokens": 2000}
        response_text = safe_cloudflare_request(url, headers, payload)

        history_db.insert_one(
            {
                "chat_id": chat_id,
                "user_id": token["userId"],
                "content": f"User uploaded an image with the message: {user_message}",
                "role": "user",
                "timestamp": datetime.now(),
                "file_data": "Image uploaded",
            }
        )

        history_db.insert_one(
            {
                "chat_id": chat_id,
                "user_id": token["userId"],
                "content": response_text,
                "role": "assistant",
                "timestamp": datetime.now(),
                "file_data": "",
            }
        )

        end = datetime.now()

        # Calculate total duration
        total_time = end - start
        print(f"Total Duration: {total_time}")

        return JSONResponse(content={"response": response_text})

    except Exception as e:
        logger.error("Error in /handle_image: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/transcribe_audio")
async def transcribe_audio_endpoint(
    request: Request,
    audio_file: UploadFile = File(...),
    user_message: str = Form(default=""),
    chat_id: str = Query(..., description="The ID of the chat session"),
    language: str = Query("en", description="Language for transcription"),
):
    start = datetime.now()
    try:
        token = verify_jwt_token(request.cookies.get("authToken"))
        if len(token) == 0:
            return JSONResponse(content={"error": "Invalid Token"}, status_code=401)

        user_data = users_db.find_one({"_id": token["userId"]})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found.")

        user_language = user_data.get("language", "en")

        # Read the audio file
        audio_bytes = await audio_file.read()

        # Transcribe using Whisper
        transcription = transcribe_audio_with_whisper(audio_bytes)

        if not transcription:
            raise HTTPException(
                status_code=400,
                detail="Audio transcription failed. Please try again with a different audio file.",
            )

        logger.info(f"Generated Transcription: {transcription}")

        # Rest of your existing logic for handling the transcription...
        if user_message.strip():
            prompt_text = (
                f"User asked: {user_message}. Here is the transcript: {transcription}"
            )
        else:
            prompt_text = f"""Just Summarize this transcript in {user_language} and even if the user speak in any language use {user_language} language : {transcription}
                the transcript can be a conversation or a lecture and there will be no mentions of persons name so think accordingly.
                
                Language:
                - English (en)
                - Hindi (hi)
                - Marathi (mr)
                - Use the language specified by the user or default to English if not specified.

                """

        logger.info(f"Prompt Text: {prompt_text}")

        url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{models}"
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        payload = {"prompt": prompt_text, "max_tokens": 2000}

        response_text = safe_cloudflare_request(url, headers, payload)

        # Store in history
        history_db.insert_one(
            {
                "chat_id": chat_id,
                "user_id": token["userId"],
                "content": (user_message + "\n\n")
                + f"User uploaded an audio file: {audio_file.filename}",
                "role": "user",
                "timestamp": datetime.now(),
                "file_data": transcription,
            }
        )

        history_db.insert_one(
            {
                "chat_id": chat_id,
                "user_id": token["userId"],
                "content": response_text,
                "role": "assistant",
                "timestamp": datetime.now(),
                "file_data": "",
            }
        )

        end = datetime.now()
        # Calculate total duration
        total_time = end - start
        print(f"Total Duration: {total_time}")

        return JSONResponse(
            content={"transcription": transcription, "response": response_text}
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /transcribe_audio: {str(e)}")
        return JSONResponse(
            content={"error": "Failed to process audio file. Please try again."},
            status_code=500,
        )


@app.delete("/delete_chat")
async def delete_chat(
    chat_id: str = Query(..., description="The ID of the chat session to delete"),
    request: Request = None,
):
    try:
        token = verify_jwt_token(request.cookies.get("authToken"))
        if len(token) == 0:
            return JSONResponse(content={"error": "Invalid Token"}, status_code=401)

        user_id = token["userId"]

        delete_result = history_db.delete_many({"chat_id": chat_id, "user_id": user_id})

        if delete_result.deleted_count == 0:
            return JSONResponse(
                content={"error": "No chat found with the specified chat_id."},
                status_code=404,
            )

        return JSONResponse(
            content={"message": f"Chat {chat_id} deleted successfully."},
            status_code=200,
        )

    except Exception as e:
        logger.error("Error in /delete_chat: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/upload_doc")
async def upload_doc(
    file: UploadFile = File(...),
    password: str = Form(default=None),
    request: Request = None,
):
    try:
        token = verify_jwt_token(request.cookies.get("authToken"))
        if len(token) == 0:
            raise HTTPException(status_code=401, detail="Invalid or missing token.")

        user_id = token["userId"]

        allowed_extensions = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt"}
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed types: {', '.join(allowed_extensions)}",
            )

        UPLOAD_DIR = Path("uploaded_files") / user_id
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        file_location = UPLOAD_DIR / file.filename
        with file_location.open("wb") as f:
            f.write(await file.read())

        if file_extension == ".docx":
            text = extract_text_from_docx(str(file_location), password)
        else:
            text = extract_text_from_file(str(file_location))

        if not text:
            raise HTTPException(
                status_code=400, detail="Failed to extract text from file."
            )

        # Store in MongoDB
        file_data = add_to_db(str(file_location), str(uuid4()), user_id)

        file_metadata = {
            "user_id": user_id,
            "file_name": file.filename,
            "file_path": str(file_location),
            "file_type": file_extension,
            "uploaded_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=2),  # Auto-expire in 2 days
        }
        files_db = mydb["files"]
        files_db.insert_one(file_metadata)

        return JSONResponse(
            content={
                "message": "File uploaded successfully.",
                "doc_path": str(file_location),
                "file_name": file.filename,
                "file_type": file_extension,
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("Error in /upload_doc: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/fetch_all_documents")
async def fetch_all_documents(request: Request):
    try:
        token = verify_jwt_token(request.cookies.get("authToken"))
        if len(token) == 0:
            raise HTTPException(status_code=401, detail="Invalid or missing token.")

        user_id = token["userId"]

        files_db = mydb["files"]
        user_documents = list(files_db.find({"user_id": user_id}))

        if not user_documents:
            return JSONResponse(
                content={"message": "No documents found for the user."},
                status_code=200,
            )

        documents_response = []

        for doc in user_documents:
            # Get all chunks for this document
            document_chunks = documents_db.find(
                {"file_path": doc["file_path"], "user_id": user_id}
            ).sort("created_at", pymongo.ASCENDING)

            combined_content = "\n\n".join(
                chunk["content"] for chunk in document_chunks
            )

            documents_response.append(
                {
                    "file_name": doc["file_name"],
                    "file_path": doc["file_path"],
                    "file_type": doc["file_type"],
                    "uploaded_at": doc["uploaded_at"].isoformat()
                    if doc.get("uploaded_at")
                    else None,
                    "expires_at": doc.get("expires_at", "").isoformat(),
                    "content": combined_content,
                }
            )

        return JSONResponse(
            content={"documents": documents_response},
            status_code=200,
        )
    except Exception as e:
        logger.error("Error in /fetch_all_documents: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.delete("/delete_document")
async def delete_document(
    file_name: str = Query(..., description="Name of the file to delete"),
    request: Request = None,
):
    try:
        token = verify_jwt_token(request.cookies.get("authToken"))
        if len(token) == 0:
            raise HTTPException(status_code=401, detail="Invalid or missing token.")

        user_id = token["userId"]

        files_db = mydb["files"]
        document_metadata = files_db.find_one(
            {"file_name": file_name, "user_id": user_id}
        )

        if not document_metadata:
            return JSONResponse(
                content={
                    "error": "No document found with the specified file name for the user."
                },
                status_code=404,
            )

        file_path = document_metadata["file_path"]

        # Delete document chunks from MongoDB
        delete_result = documents_db.delete_many(
            {"file_path": file_path, "user_id": user_id}
        )
        logger.info(
            f"Deleted {delete_result.deleted_count} chunks from MongoDB for file: {file_name}"
        )

        # Delete file metadata
        files_db.delete_one({"file_name": file_name, "user_id": user_id})
        logger.info(f"Deleted document metadata from MongoDB for file: {file_name}")

        # Delete physical file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted physical file from server: {file_path}")
        else:
            logger.warning(f"Physical file not found on server: {file_path}")

        return JSONResponse(
            content={"message": f"Document '{file_name}' deleted successfully."},
            status_code=200,
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("Error in /delete_document: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/voice")
async def handle_voice(
    request: Request,
    voice: UploadFile = File(...),
    chat_id: str = Query(..., description="The ID of the chat session"),
):
    start = datetime.now()
    try:
        token = verify_jwt_token(request.cookies.get("authToken"))
        if not token:
            raise HTTPException(status_code=401, detail="Invalid Token")

        user_data = users_db.find_one({"_id": token["userId"]})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found.")

        user_language = user_data.get("language", "en")

        voice_bytes = await voice.read()
        audio_text_res = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{whisper_model}",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={
                "audio": base64.b64encode(voice_bytes).decode("utf-8"),
                "task": "transcribe",
            },
        )
        audio_text = audio_text_res.json()["result"]["text"]
        logger.info("Transcribed audio: %s", audio_text)

        history_db.insert_one(
            {
                "chat_id": chat_id,
                "user_id": token["userId"],
                "content": audio_text,
                "role": "user",
                "timestamp": datetime.now(),
                "file_data": "Voice input",
            }
        )

        chat_history = []
        ittr_chat_history = history_db.find(
            {"chat_id": chat_id, "user_id": token["userId"]}
        ).sort("timestamp", pymongo.ASCENDING)

        for chat in ittr_chat_history:
            chat_history.append({"role": chat["role"], "content": chat["content"]})

        system_content = (
            f"You are a friendly assistant, generate responses based on the user's input and the context of the conversation. your name is multigpt"
            f"""The user's preferred language is {user_language}. Respond in {user_language} unless the user specifies otherwise and even if the user speak in any language use {user_language} language.
            
            Language:
            - English (en)
            - Hindi (hi)
            - Marathi (mr)
            - Use the language specified by the user or default to English if not specified. 
            """
            "Keep responses short and sweet like a human. Based on the topic, always ask a follow-up question at the end. "
            "when users ask about your functionality tell them audio, image, document and text processing. also tell them you can chat in primarily 3 languages english, marathi and hindi. also you can visualize stastical data. also - Code generation and debugging, Youtube video Processing"
        )
        chat_history = [{"role": "system", "content": system_content}] + chat_history

        url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{models}"
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        payload = {"messages": chat_history, "max_tokens": 400}

        ai_response = safe_cloudflare_request(url, headers, payload)
        logger.info("AI response: %s", ai_response)

        history_db.insert_one(
            {
                "chat_id": chat_id,
                "user_id": token["userId"],
                "content": ai_response,
                "role": "assistant",
                "timestamp": datetime.now(),
                "file_data": "",
            }
        )

        response_audio = eleven_client.text_to_speech.convert(
            voice_id=ElevenLabs_VOICE_ID,
            text=ai_response,
            model_id="eleven_multilingual_v2",
        )

        end = datetime.now()
        # Calculate total duration
        total_time = end - start
        print(f"Total Duration: {total_time}")

        return StreamingResponse(
            response_audio,
            media_type="audio/mp3",
        )

    except Exception as e:
        logger.error("Error in /voice: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/audio")
async def handle_audio(
    request: Request,
    audio: UploadFile = File(...),
    chat_id: str = Query(..., description="The ID of the chat session"),
):
    start = datetime.now()
    try:
        token = verify_jwt_token(request.cookies.get("authToken"))
        if not token:
            raise HTTPException(status_code=401, detail="Invalid Token")

        user_data = users_db.find_one({"_id": token["userId"]})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found.")

        user_language = user_data.get("language", "en")

        audio_bytes = await audio.read()
        audio_text_res = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{whisper_model}",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            json={
                "audio": base64.b64encode(audio_bytes).decode("utf-8"),
                "task": "transcribe",
            },
        )
        audio_text = audio_text_res.json()["result"]["text"]
        logger.info("Transcribed audio: %s", audio_text)

        history_db.insert_one(
            {
                "chat_id": chat_id,
                "user_id": token["userId"],
                "content": audio_text,
                "role": "user",
                "timestamp": datetime.now(),
                "file_data": "Audio input",
            }
        )

        chat_history = []
        ittr_chat_history = history_db.find(
            {"chat_id": chat_id, "user_id": token["userId"]}
        ).sort("timestamp", pymongo.ASCENDING)

        for chat in ittr_chat_history:
            chat_history.append({"role": chat["role"], "content": chat["content"]})

        system_content = (
            f"You are a friendly assistant, generate responses based on the user's input and the context of the conversation. your name is multigpt"
            f"""The user's preferred language is {user_language}. Respond in {user_language} unless the user specifies otherwise and even if the user speak in any language use {user_language} language.
    
            Language:
            - English (en)
            - Hindi (hi)
            - Marathi (mr)
            - Use the language specified by the user or default to English if not specified. 
            """
            "Keep responses short and sweet like a human. "
            "when users ask about your functionality tell them audio, image, document and text processing. also tell them you can chat in primarily 3 languages english, marathi and hindi. also you can visualize stastical data. also Code generation and debugging and Youtube video Processing"
        )
        chat_history = [{"role": "system", "content": system_content}] + chat_history

        url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{models}"
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        payload = {"messages": chat_history, "max_tokens": 400}

        ai_response = safe_cloudflare_request(url, headers, payload)
        logger.info("AI response: %s", ai_response)

        history_db.insert_one(
            {
                "chat_id": chat_id,
                "user_id": token["userId"],
                "content": ai_response,
                "role": "assistant",
                "timestamp": datetime.now(),
                "file_data": "",
            }
        )

        end = datetime.now()
        # Calculate total duration
        total_time = end - start
        print(f"Total Duration: {total_time}")

        return JSONResponse(content={"response": ai_response})

    except Exception as e:
        logger.error("Error in /audio: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn fast_api:app --host 0.0.0.0 --port 8000 --reload

