# Mengimpor library Google GenAI untuk membuat client API Google.
from google import genai
# Mengimpor fungsi load_dotenv agar file .env bisa dibaca.
from dotenv import load_dotenv
# Mengimpor wrapper model Gemini dari LangChain.
from langchain_google_genai import ChatGoogleGenerativeAI


# Memuat variabel lingkungan dari file .env agar API key tersedia.
load_dotenv()

# Menampilkan indikator awal bahwa proses dimulai.
print("process 1 start")

# Membuat instance client model Gemini.
client = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Mengirimkan prompt sederhana ke Gemini untuk mendapatkan jawaban singkat.
response = client.invoke(
    # Menentukan input prompt yang dikirim ke model.
    input="Halo Gemini! Balas singkat: API berhasil."
)

# Menampilkan indikator proses kedua yang sedang berjalan.
print("process start")


# Menampilkan isi jawaban dari model ke terminal.
print(response.content)