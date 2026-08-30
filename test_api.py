from google import genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

print("process 1 start")

client = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

response = client.invoke(
    input="Halo Gemini! Balas singkat: API berhasil."
)

print("process start")


print(response.content)