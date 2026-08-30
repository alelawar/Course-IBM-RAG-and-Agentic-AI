# Mengimpor decorator @tool dari LangChain agar fungsi bisa dipakai sebagai tool AI.
from langchain_core.tools import tool
# Mengimpor helper create_react_agent untuk membuat agen yang bisa berpikir dan memakai tool.
from langgraph.prebuilt import create_react_agent
# Mengimpor class model Gemini dari LangChain agar bisa mengakses API Gemini.
from langchain_google_genai import ChatGoogleGenerativeAI
# Mengimpor modul regex dan os untuk manipulasi string dan akses sistem.
import re, os
# Mengimpor fungsi load_dotenv agar variabel lingkungan dari file .env bisa dibaca.
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env ke process environment.
load_dotenv()

# Membuat instance model Gemini dengan model tertentu.
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Contoh percobaan lama untuk menguji LLM secara langsung.
# response = llm.invoke("Apa itu tool calling di LangChain?")
# # print(response.text)

# Menandai fungsi add_numbers sebagai tool yang bisa dipanggil AI.
@tool
def add_numbers(inputs: str) -> dict:
    # Docstring menjelaskan fungsi ini untuk dokumentasi dan AI.
    """Adds all numbers found in the input string and returns their sum."""
    # Mencari semua angka di teks lalu mengubahnya ke integer.
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    # Mengembalikan hasil penjumlahan dalam format dictionary.
    return {"result": sum(numbers)}

# Menandai fungsi pengurang sebagai tool AI.
@tool
def new_subtract_numbers(inputs: str) -> dict:
    # Docstring menjelaskan cara kerja fungsi pengurangan sequential.
    """Extracts numbers from the input string and subtracts them sequentially, starting from the first number."""
    # Menghapus koma lalu memisahkan string menjadi list kata, lalu ambil yang hanya angka.
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    # Jika tidak ada angka, kembalikan 0 agar aman.
    if not numbers:
        return {"result": 0}
    # Menetapkan hasil awal dari angka pertama.
    result = numbers[0]
    # Mengurangi angka berikutnya satu per satu secara berurutan.
    for num in numbers[1:]:
        result -= num
    # Mengembalikan hasil akhir.
    return {"result": result}

# Menandai fungsi perkalian sebagai tool AI.
@tool
def multiply_numbers(inputs: str) -> dict:
    # Docstring menjelaskan fungsi perkalian angka yang ditemukan di input.
    """Extracts numbers from the input string and calculates their product."""
    # Menghapus koma lalu memisahkan teks menjadi kata, lalu ambil kata yang berupa angka.
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    # Jika tidak ada angka, hasil default adalah 1.
    if not numbers:
        return {"result": 1}
    # Menetapkan hasil awal ke 1 agar perkalian benar.
    result = 1
    # Mengalikan semua angka satu per satu.
    for num in numbers:
        result *= num
    # Mengembalikan hasil perkalian.
    return {"result": result}

# Menandai fungsi pembagian sebagai tool AI.
@tool
def divide_numbers(inputs: str) -> dict:
    # Docstring menjelaskan cara kerja fungsi pembagian berurutan.
    """Extracts numbers from the input string and divides the first number by each subsequent number in sequence."""
    # Menghapus koma lalu memisahkan string menjadi kata, lalu ambil angka saja.
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    # Jika tidak ada angka, hasil default 0.
    if not numbers:
        return {"result": 0}
    # Menentukan angka awal yang akan dibagi.
    result = numbers[0]
    # Membagi hasil dengan setiap angka berikutnya secara berurutan.
    for num in numbers[1:]:
        result /= num
    # Mengembalikan hasil pembagian.
    return {"result": result}

# Menyimpan semua tool yang tersedia untuk agen matematika.
tools = [add_numbers, new_subtract_numbers, multiply_numbers, divide_numbers]

# Membuat agen reAct yang bisa memanggil tool dan menjawab pertanyaan matematika.
math_agent = create_react_agent(
    # Menghubungkan model Gemini ke agen.
    model=llm,
    # Menentukan tool yang dipakai agen.
    tools=tools,
    # Memberi instruksi ke agen agar bertindak seperti asisten matematika yang helpful.
    prompt="You are a helpful mathematical assistant that can perform various operations. Use the tools precisely and explain your reasoning clearly."
)

# Mengirimkan pertanyaan ke agen untuk diproses dengan tool yang ada.
response = math_agent.invoke({
    # Format pesan menggunakan role human dan konten pertanyaan.
    "messages": [("human", "What is 25 divided by 4?")]
})
# Menampilkan jawaban terakhir dari agen ke terminal.
print(response["messages"][-1].content)