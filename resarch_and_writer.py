"""Contoh aplikasi multi-agent sederhana menggunakan CrewAI dan Gemini.

Alur program:
1. Agent researcher mencari dan merangkum informasi tentang sebuah topik.
2. Agent writer mengubah hasil riset menjadi artikel pendek.
3. Crew menjalankan kedua task tersebut secara berurutan.
"""

# os digunakan untuk membaca API key dari environment variable.
import os
# sys digunakan untuk membaca argumen yang diketik saat menjalankan file.
import sys
# load_dotenv membaca isi file .env ke environment variable.
from dotenv import load_dotenv
# Mengimpor komponen utama CrewAI untuk membuat agent, task, dan workflow.
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
# Memuat konfigurasi, termasuk GEMINI_API_KEY, dari file .env.
load_dotenv()

# Membuat konfigurasi model bahasa yang akan dipakai oleh semua agent.
# CrewAI menggunakan LiteLLM di belakang layar, sehingga format modelnya
# adalah "gemini/<nama_model>".
gemini_llm = LLM(
    # Menentukan provider Gemini dan nama model yang digunakan.
    model="gemini/gemini-3.6-flash",
    # Mengambil API key dari environment variable agar tidak ditulis langsung di kode.
    api_key=os.getenv("GEMINI_API_KEY"),
    # Mengatur tingkat kreativitas model; angka lebih tinggi biasanya lebih variatif.
    temperature=0.7,
)
serper_api = os.getenv("SERPER_API_KEY")

# Membuat agent researcher yang bertugas mengumpulkan informasi.
researcher = Agent(
    # Nama peran agent dalam proses kerja.
    role="Senior Researcher",
    # Tujuan utama agent. {topic} akan diisi saat crew dijalankan.
    goal="Menemukan fakta, data, dan poin-poin kunci paling relevan tentang {topic}",
    # Latar belakang membantu model memahami cara berpikir dan gaya kerja agent.
    backstory=(
        "Kamu adalah seorang periset yang teliti dan suka menggali informasi "
        "sampai dapat insight yang tajam dan akurat, bukan cuma info permukaan."
    ),
    # Menghubungkan agent ke konfigurasi model Gemini yang sudah dibuat.
    llm=gemini_llm,
    # Menampilkan proses kerja agent di terminal.
    verbose=True,
    # Agent tidak boleh membagi tugasnya kepada agent lain.
    allow_delegation=False,
    # Tools Tambahan untuk search engine
    tools=[SerperDevTool()]
)

# Membuat agent writer yang bertugas menulis artikel berdasarkan hasil riset.
writer = Agent(
    # Nama peran agent dalam proses kerja.
    role="Konten Writer",
    # Tujuan utama agent writer.
    goal="Menulis artikel pendek yang jelas, menarik, dan mudah dipahami tentang {topic}",
    # Latar belakang memberi arahan tentang gaya tulisan yang diharapkan.
    backstory=(
        "Kamu adalah penulis konten berpengalaman yang jago mengubah "
        "hasil riset mentah menjadi tulisan yang enak dibaca orang awam."
    ),
    # Menggunakan model Gemini yang sama dengan agent researcher.
    llm=gemini_llm,
    # Menampilkan proses kerja agent writer di terminal.
    verbose=True,
    # Agent writer juga tidak mendelegasikan tugas ke agent lain.
    allow_delegation=False
)

# Membuat task pertama: researcher mengumpulkan poin-poin penting.
research_task = Task(
    # Instruksi detail yang akan dikerjakan oleh researcher.
    description=(
        "Riset topik '{topic}'. Kumpulkan 4-6 poin kunci paling penting, "
        "termasuk fakta menarik atau data pendukung kalau ada."
    ),
    # Bentuk hasil yang diharapkan dari task ini.
    expected_output="Daftar poin-poin kunci (bullet points) hasil riset tentang {topic}.",
    # Menentukan agent yang bertanggung jawab atas task ini.
    agent=researcher
)

# Membuat task kedua: writer menyusun artikel dari hasil riset.
writing_task = Task(
    # Instruksi penulisan artikel untuk agent writer.
    description=(
        "Berdasarkan hasil riset dari tugas sebelumnya, tulis artikel pendek "
        "(3-4 paragraf) tentang '{topic}' dengan gaya santai tapi informatif."
    ),
    # Bentuk hasil akhir yang diharapkan dari task penulisan.
    expected_output="Artikel pendek 3-4 paragraf dalam Bahasa Indonesia tentang {topic}.",
    # Menentukan agent yang bertanggung jawab atas task ini.
    agent=writer,
    # Memberikan hasil research_task sebagai konteks untuk task ini.
    context=[research_task],
)

# Membuat crew yang mengatur agent dan task dalam satu workflow.
crew = Crew(
    # Daftar agent yang tersedia di dalam crew.
    agents=[researcher, writer],
    # Daftar task yang harus dikerjakan.
    tasks=[research_task, writing_task],
    # Task dijalankan sesuai urutan: research terlebih dahulu, lalu writing.
    process=Process.sequential,
    # Menampilkan detail proses crew di terminal.
    verbose=True
)


# Blok ini hanya dijalankan saat file dieksekusi langsung,
# bukan saat file diimpor sebagai module.
if __name__ == "__main__":
    # Menggabungkan semua argumen command line menjadi satu topik.
    # Jika tidak ada argumen, gunakan topik default.
    topic = " ".join(sys.argv[1:]) or "manfaat olahraga pagi bagi produktivitas kerja"
    # Menjalankan seluruh crew dan mengirimkan topic ke placeholder {topic}.
    result = crew.kickoff(inputs={"topic": topic})

    # Menampilkan garis pemisah agar output lebih mudah dibaca.
    print("\n" + "=" * 50)
    # Menampilkan judul hasil akhir.
    print("HASIL AKHIR:")
    # Menampilkan garis pemisah penutup judul.
    print("=" * 50)
    # Menampilkan artikel yang dihasilkan oleh writer.
    print(result)