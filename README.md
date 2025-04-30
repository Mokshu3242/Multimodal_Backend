# MultiGPT - AI Agent with Multi-Modal Capabilities 🚀

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green)
![Cloudflare](https://img.shields.io/badge/Cloudflare_AI-Integrated-orange)
![MultiModal](https://img.shields.io/badge/Multi_Modal-Text%2C%20Audio%2C%20Image%2C%20Doc-brightgreen)

An advanced AI agent capable of processing **text, audio, images, and documents** with visualization support. Built with FastAPI, Cloudflare AI, ElevenLabs, and LangChain.

## 🌟 Features

### 1. **Core Capabilities**
- **Conversational AI** with persistent chat history
- **Multi-language support** (English, Hindi, Marathi)
- **JWT Authentication** + Self-Hosted OTP verification
- **Document, Image and Audio Processing**
- **Rate-limited API endpoints**

### 2. **Input Processing**
| Type       | Endpoint          | Technologies Used              |
|------------|-------------------|--------------------------------|
| Text       | `/chat`           | Cloudflare LLM                 |
| Voice      | `/voice`          | ElevenLabs TTS + Whisper       |
| Audio      | `/audio`          | Whisper transcription          |
| Images     | `/handle_image`   | CLIP image analysis            |
| Documents  | `/upload_doc`     | PyPDFium2, docx2txt, msoffcrypto |

### 3. **Advanced Functions**
- **YouTube transcript extraction**
- **Data visualization** (bar/line/pie charts)
- **Auto-expiring file storage** (2-day TTL)

## 🛠️ Tech Stack
- **Frontend**: React JS
- **Backend**: FastAPI
- **AI Services**: 
  - Cloudflare (LLaMA-2, Whisper, CLIP)
  - ElevenLabs (Text-to-Speech)
- **Database**: MongoDB
- **Data Processing**: 
  - LangChain (Document chunking)
  - Pandas/Plotly (Visualizations)

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- MongoDB
- Cloudflare & ElevenLabs accounts

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/multigpt.git
cd Multimodal_Backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```
Also add frontend URL in the CORS

### 2. Environment Variables Setup

To run **MultiGPT** successfully, you need to create a `.env` file in your backend root directory with the following environment variables. Below are detailed steps to acquire each one:

---

### 🔹 MONGO_DB_CONNECT  
MongoDB connection URI

🖥️ Local MongoDB:
- Install MongoDB: https://www.mongodb.com/try/download/community
- Start MongoDB service (`mongod`)
- Use this:
  MONGO_DB_CONNECT="mongodb://localhost:27017"

☁️ MongoDB Atlas:
- Sign up at https://cloud.mongodb.com
- Create a cluster → Click "Connect" → "Connect your application"
- Copy and paste the provided connection URI (replace <username> and <password>)

---

### 🔹 CLOUDFLARE_ACCOUNT_ID & CLOUDFLARE_AUTH_TOKEN  
Required for Cloudflare AI APIs

- Visit https://dash.cloudflare.com
- Select your domain → Overview → Copy Account ID
- Go to My Profile → API Tokens
- Click Create Token
- Use the "Edit zone DNS" template or create a custom token with needed scopes
- Copy the generated token as CLOUDFLARE_AUTH_TOKEN

---

### 🔹 ELEVENLABS_API_KEY & ELEVEN_VOICE_ID  
Used for Text-to-Speech via ElevenLabs

- Go to https://www.elevenlabs.io
- Sign in → Go to Profile → API Key
- Click Generate API Key → Copy and paste it as ELEVENLABS_API_KEY
- Navigate to Voice Lab → Select or create a voice
- Copy the Voice ID from the voice details panel

---

### 🔹 JWT_SECRET  
Used to securely sign JWT tokens

Generate a secure key using Python:

    import secrets
    print(secrets.token_hex(32))

Copy the generated value into .env:
    JWT_SECRET="your_generated_secret"

---

### 🔹 SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD  
Used for sending emails (e.g., OTP verification)

✉️ Gmail SMTP Setup:
    SMTP_SERVER="smtp.gmail.com"
    SMTP_PORT=587
    EMAIL_ADDRESS="your@gmail.com"

🔐 Get Gmail App Password:
- Go to Google Account Security
- Enable 2-Step Verification
- Go to App Passwords → Select app: Mail, device: Other
- Generate → Copy the password → Add it to .env as:

    EMAIL_PASSWORD="your_app_password"

---

### 🔹 FRONTEND_URL  
CORS configuration for your React frontend

🧪 For local testing:
    FRONTEND_URL="http://localhost:3000"

🌐 For production:
    FRONTEND_URL="https://your-frontend-domain.com"

---

### 📄 Example .env File

    MONGO_DB_CONNECT="mongodb://localhost:27017"
    CLOUDFLARE_ACCOUNT_ID="your_account_id"
    CLOUDFLARE_AUTH_TOKEN="your_api_token"
    ELEVENLABS_API_KEY="your_elevenlabs_key"
    ELEVEN_VOICE_ID="Voice_ID"
    JWT_SECRET="your_jwt_secret"
    SMTP_SERVER="smtp.gmail.com"
    SMTP_PORT=587
    EMAIL_ADDRESS="your@email.com"
    EMAIL_PASSWORD="your_app_password"
    FRONTEND_URL="http://localhost:3000"

### 3. Running
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Access docs: http://localhost:8000/docs
```

### 4. Working

<table>
  <tr>
    <td align="center">
      <b>Login and Signup</b><br>
      <img src="Images/Login_Signup.webp" alt="Login and Signup" width="200">
    </td>
    <td align="center">
      <b>Chat with Documents</b><br>
      <img src="Images/Chat_Document.webp" alt="Chat with Documents" width="200">
    </td>
    <td align="center">
      <b>AI Image Processing</b><br>
      <img src="Images/image_processing.webp" alt="AI Image Processing" width="200">
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Speech Interaction</b><br>
      <img src="Images/speech_interaction.webp" alt="Speech Interaction" width="200">
    </td>
    <td align="center">
      <b>Data Visualization</b><br>
      <img src="Images/visualization.webp" alt="Data Visualization" width="200">
    </td>
    <td align="center">
      <b>YouTube Integration</b><br>
      <img src="Images/Youtube.webp" alt="YouTube Integration" width="200">
    </td>
  </tr>
</table>
