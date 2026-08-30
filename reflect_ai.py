# Mengimpor modul os untuk keperluan operasi sistem seperti membaca environment.
import os
# Mengimpor fungsi load_dotenv supaya konfigurasi dari file .env bisa terbaca.
from dotenv import load_dotenv

# Mengimpor model Gemini dari package LangChain.
from langchain_google_genai import ChatGoogleGenerativeAI
# Mengimpor tipe pesan dasar dan kelas pesan spesifik HumanMessage dan AIMessage.
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# Mengimpor template prompt dan placeholder pesan untuk pembuatan/pembandingan chat.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Mengimpor class MessageGraph dan END untuk membangun alur graf percakapan.
from langgraph.graph import END, MessageGraph
# Mengimpor tipe List dan Sequence untuk annotation pada fungsi.
from typing import List, Sequence

# Memuat variabel lingkungan dari file .env sebelum aplikasi berjalan.
load_dotenv()

# Membuat instance model Gemini dengan versi model tertentu.
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Membuat template prompt untuk menghasilkan draft konten LinkedIn.
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            # Peran sistem memberi instruksi agar AI bertindak seperti assistant LinkedIn yang profesional.
            "system",
            # Instruksi utama untuk menulis konten LinkedIn yang bagus, menarik, dan terstruktur.
            "You are a professional LinkedIn content assistant tasked with crafting engaging, insightful, and well-structured LinkedIn posts."
            # Lanjutkan instruksi agar AI menghasilkan post terbaik sesuai permintaan pengguna.
            " Generate the best LinkedIn post possible for the user's request."
            # Jika ada feedback, AI harus memperbaiki versi sebelumnya dengan lebih jelas dan lebih menarik.
            " If the user provides feedback or critique, respond with a refined version of your previous attempts, improving clarity, tone, or engagement as needed.",
        ),
        # Placeholder untuk menerima riwayat percakapan yang masuk.
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Menggabungkan prompt generation dengan LLM agar siap dipakai.
generate_chain = generation_prompt | llm

# Membuat template prompt khusus untuk merefleksikan / mengkritik draft LinkedIn.
reflection_prompt = ChatPromptTemplate.from_messages([
    (
        # Sistem memberi peran AI sebagai content strategist dan expert thought leadership.
        "system",
        # Instruksi untuk mengevaluasi kualitas draft LinkedIn dari berbagai aspek. 
        """You are a professional LinkedIn content strategist and thought leadership expert. Your task is to critically evaluate the given LinkedIn post and provide a comprehensive critique. Follow these guidelines:

        1. Assess the post's overall quality, professionalism, and alignment with LinkedIn best practices.
        2. Evaluate the structure, tone, clarity, and readability of the post.
        3. Analyze the post's potential for engagement (likes, comments, shares) and its effectiveness in building professional credibility.
        4. Consider the post's relevance to the author's industry, audience, or current trends.
        5. Examine the use of formatting (e.g., line breaks, bullet points), hashtags, mentions, and media (if any).
        6. Evaluate the effectiveness of any call-to-action or takeaway.

        Provide a detailed critique that includes:
        - A brief explanation of the post's strengths and weaknesses.
        - Specific areas that could be improved.
        - Actionable suggestions for enhancing clarity, engagement, and professionalism.

        Your critique will be used to improve the post in the next revision step, so ensure your feedback is thoughtful, constructive, and practical.
        """
    ),
    # Placeholder untuk menampung pesan yang dibahas saat refleksi.
    MessagesPlaceholder(variable_name="messages")
])

# Menggabungkan prompt refleksi dengan model LLM.
reflect_chain = reflection_prompt | llm

# Membuat graf pesan untuk alur generate -> reflect -> generate.
graph = MessageGraph()

# Fungsi yang menghasilkan draft tulisan LinkedIn dari state saat ini.
def generation_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    # Jalankan chain generate dengan state berisi riwayat pesan.
    generated_post = generate_chain.invoke({"messages": state})
    # Kembalikan hasil sebagai AIMessage agar bisa masuk ke state berikutnya.
    return [AIMessage(content=generated_post.content)]

# Fungsi yang mengevaluasi draft saat ini dan menghasilkan kritik.
def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    # Tukar role: AIMessage jadi HumanMessage dan sebaliknya,
    # supaya giliran terakhir yang dikirim ke Gemini selalu dari "user"
    # Ini penting agar model menerima pola user/assistant yang konsisten.
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # Mengubah tipe pesan agar kritik dibentuk dalam format yang masuk akal untuk model.
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    # Jalankan chain refleksi untuk memberikan kritik terhadap draft yang ada.
    res = reflect_chain.invoke({"messages": translated})
    # Kembalikan hasil kritik sebagai HumanMessage agar masuk ke state berikutnya sebagai user feedback.
    return [HumanMessage(content=res.content)]

# Menambahkan node generate ke graf.alur.
graph.add_node("generate", generation_node)
# Menambahkan node reflect ke graf.alur.
graph.add_node("reflect", reflection_node)
# Menyambungkan reflect kembali ke generate agar iterasi bisa terjadi.
graph.add_edge("reflect", "generate")
# Menetapkan node generate sebagai titik masuk pertama.
graph.set_entry_point("generate")

# Fungsi keputusan: apakah proses harus berhenti atau melanjutkan refleksi.
def should_continue(state: List[BaseMessage]):
    # Jika panjang state melebihi 6, kita berhenti agar tidak berulang terus.
    if len(state) > 6:
        return END
    # Jika masih dalam batas, lanjut ke node reflect.
    return "reflect"

# Menambahkan logika kondisi antar node generate dan reflect.
graph.add_conditional_edges("generate", should_continue)

# Mengompilasi graf menjadi workflow yang bisa dipanggil.
workflow = graph.compile()

# Input awal user: meminta AI menulis draft LinkedIn tentang melamar developer di IBM.
inputs = HumanMessage(content="Write a linkedin post on getting a software developer job at IBM under 160 characters")

# Menjalankan workflow dari input awal.
response = workflow.invoke(inputs)

# Menampilkan draft pertama sebelum kritik.
print("--- Draft pertama (sebelum kritik) ---")
# Menampilkan isi draft pertama dari response index 1.
print(response[1].content)

# Menampilkan kritik pertama yang dihasilkan oleh refleksi.
print("\n--- Kritik pertama ---")
# Menampilkan kritik di index 2.
print(response[2].content)

# Menampilkan hasil akhir setelah beberapa iterasi.
print("\n--- Hasil akhir setelah beberapa iterasi ---")
# Menampilkan hasil akhir dari response paling terakhir.
print(response[-1].content)

