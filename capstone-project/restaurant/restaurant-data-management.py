"""
restaurant_data_management.py
------------------------------
Adaptasi dari lab-instructions.md (Module 1, Lesson 3) — versi ini
menggunakan Gemini API (google-genai) alih-alih ibm-watsonx-ai, mengikuti
pola yang sudah terbukti jalan di script referensi (California-Culinary-Map
batch processor).

CATATAN PENTING SOAL KUOTA:
Kamu bilang jatah maksimal 15 request ke Gemini. Supaya aman:
  - Tidak ada pemanggilan LLM yang dilakukan beruntun tanpa jeda
    (lihat SLEEP_BETWEEN_CALLS).
  - Ada REQUEST_COUNTER global + MAX_LLM_REQUESTS sebagai pengaman keras:
    kalau kepakai lebih dari batas, program akan berhenti dengan pesan
    jelas, bukan diam-diam boros kuota.
  - Unit test (Exercise 3) di-MOCK total: tidak memanggil API asli sama
    sekali, supaya kamu bisa run test berkali-kali tanpa takut kuota habis.
    Untuk screenshot `new_data_entry_process()`, jalankan manual lewat
    `manage_restaurants(FILEPATH, BACKUP_PATH)` (baris paling bawah, saat
    ini di-comment) dengan SATU input paragraf baru.
"""

import os
import io
import json
import shutil
import time
import unittest
from unittest.mock import patch

from dotenv import load_dotenv
import google.genai as genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# ---------------------------------------------------------------------------
# Konfigurasi & klien Gemini
# ---------------------------------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY tidak ditemukan! Pastikan file .env sudah berisi "
        "GOOGLE_API_KEY=your_key_here"
    )

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.6-flash"

FILEPATH = "structured_restaurant_data.json"
BACKUP_PATH = "structured_restaurant_data.json.bak"

EXAMPLE_RESTAURANT_PARAGRAPH = (
    "Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, "
    "**casual taqueria** specializing in **Baja-style seafood**. With a "
    "**4.2/5** rating, it captures the salt-air energy of the coast through "
    "its signature beer-battered snapper tacos and zesty octopus ceviche, "
    "making it a premier spot for open-air dining near the pier. "
    "Price range: $"
)

# --- Pengaman kuota (versi konservatif, sesuai permintaan) -----------------
MAX_LLM_REQUESTS = 15          # batas keras total request per sesi
SLEEP_BETWEEN_CALLS = 8        # jeda antar panggilan LLM yang berurutan (detik)
RETRY_SLEEP_RATE_LIMIT = 45    # jeda kalau kena 429 / RESOURCE_EXHAUSTED
RETRY_SLEEP_GENERIC = 10       # jeda kalau error lain
MAX_RETRIES = 2                # lebih sedikit dari contoh (yang pakai 3)

REQUEST_COUNTER = {"count": 0}


class QuotaExceededError(RuntimeError):
    """Dilempar saat REQUEST_COUNTER melewati MAX_LLM_REQUESTS."""
    pass


# ---------------------------------------------------------------------------
# Skema data
# ---------------------------------------------------------------------------

class Restaurant(BaseModel):
    """Skema pydantic untuk satu record restoran."""
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper file I/O (persis dari lab-instructions.md)
# ---------------------------------------------------------------------------

def load_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_data(data, file_path, backup_path):
    # Buat backup dulu sebelum menimpa file
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def show_restaurant_card(res, index):
    """Menampilkan data restoran dalam format vertikal yang rapi."""
    print(f"\n{'=' * 15} RESTAURANT #{index} {'=' * 15}")
    name = res.get("name", res.get("restaurant_name", "Unnamed Restaurant"))
    print(f"NAME        : {name}")

    for key, value in res.items():
        if key.lower() not in ["name", "restaurant_name"]:
            label = key.replace("_", " ").upper()
            print(f"{label:<12}: {value}")
    print("=" * 45)


# ---------------------------------------------------------------------------
# Exercise 1: Integrasi LLM (Gemini) untuk memproses paragraf baru
# ---------------------------------------------------------------------------

def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    """
    Membangun system message + user prompt untuk mengekstrak SATU paragraf
    deskripsi restoran menjadi JSON sesuai skema Restaurant.
    """
    system_msg = (
        "You are a data extraction assistant. Extract restaurant information "
        "from the given paragraph into a single strict JSON object matching "
        "the provided schema. Convert price range dollar signs (e.g. $, $$, "
        "$$$) into an integer (1, 2, 3). If a field is not mentioned, omit it "
        "or use null where the schema allows it."
    )
    user_prompt = f"""
    Task: Extract structured attributes from the restaurant description below
    and return ONLY a JSON object matching the schema (no extra commentary).

    Restaurant description:
    {restaurant_paragraph}
    """
    return system_msg, user_prompt


def llm_model(system_msg, prompt_txt, params=None):
    """
    Wrapper tipis di atas Gemini API, meniru pola safe_llm_call() pada
    script referensi, tapi dengan setting lebih konservatif (retry lebih
    sedikit, jeda lebih panjang) supaya hemat kuota.

    params (opsional): dict, bisa berisi 'response_schema' (pydantic model)
    untuk memaksa structured output.
    """
    if REQUEST_COUNTER["count"] >= MAX_LLM_REQUESTS:
        raise QuotaExceededError(
            f"Batas {MAX_LLM_REQUESTS} request LLM sudah tercapai. "
            "Berhenti untuk menghindari pemakaian kuota berlebih."
        )

    params = params or {}
    response_schema = params.get("response_schema")

    config_kwargs = {"system_instruction": system_msg}
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            REQUEST_COUNTER["count"] += 1
            print(
                f"[LLM] Request #{REQUEST_COUNTER['count']}/{MAX_LLM_REQUESTS} "
                f"(percobaan {attempt + 1}/{MAX_RETRIES})..."
            )
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_txt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return response.text
        except Exception as e:
            last_error = e
            error_msg = str(e)
            print(f"[LLM] Terjadi kesalahan: {error_msg}")

            if "RESOURCE_EXHAUSTED" in error_msg and "PerDay" in error_msg:
                print(">>> KUOTA HARIAN HABIS. Menghentikan proses.")
                raise QuotaExceededError("Kuota harian Gemini habis.") from e
            elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                time.sleep(RETRY_SLEEP_RATE_LIMIT)
            else:
                time.sleep(RETRY_SLEEP_GENERIC)
        finally:
            # Jeda konservatif antar panggilan berurutan, biar tidak
            # "rapid-fire" walau sukses di percobaan pertama.
            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"[LLM] Gagal setelah {MAX_RETRIES} percobaan: {last_error}")
    return None


def JSON_auto_repair_prompts(response, error_message):
    """
    Membangun system message + user prompt untuk meminta model memperbaiki
    JSON yang gagal divalidasi, berdasarkan pesan error dari Pydantic.
    """
    system_msg = (
        "You are a strict JSON repair assistant. You will be given a broken "
        "or invalid JSON string and the validation error it produced. Return "
        "ONLY the corrected JSON object, with no explanation, no markdown "
        "fences, and no extra text."
    )
    user_prompt = f"""
    The following JSON failed validation.

    Original JSON:
    {response}

    Validation error:
    {error_message}

    Please return a corrected JSON object that fixes this error while
    preserving all the original information as much as possible.
    """
    return system_msg, user_prompt


def new_data_entry_process(paragraph, itemId):
    """
    Menggabungkan seluruh pipeline untuk mengubah satu paragraf restoran
    mentah menjadi satu record JSON yang valid sesuai skema Restaurant.

    Alur:
      1. Bangun prompt ekstraksi awal, panggil llm_model() dengan
         response_schema=Restaurant supaya Gemini dipaksa balikin JSON
         sesuai skema (structured output).
      2. Coba validasi hasilnya dengan Pydantic.
      3. Kalau gagal validasi, coba SATU KALI perbaikan lewat
         JSON_auto_repair_prompts() + panggilan LLM tambahan.
      4. Kalau masih gagal, kembalikan None (tidak memaksa data rusak
         masuk ke database).
    """
    system_msg, user_prompt = restaurant_data_structure_prompt_generation(paragraph)
    raw_output = llm_model(system_msg, user_prompt, params={"response_schema": Restaurant})

    if raw_output is None:
        print("❌ Gagal mendapatkan respons dari LLM.")
        return None

    try:
        restaurant = Restaurant.model_validate_json(raw_output)
    except ValidationError as e:
        print(f"⚠️ Validasi gagal, mencoba perbaikan JSON sekali: {e}")
        repair_system_msg, repair_prompt = JSON_auto_repair_prompts(raw_output, str(e))
        repaired_output = llm_model(
            repair_system_msg, repair_prompt, params={"response_schema": Restaurant}
        )
        if repaired_output is None:
            print("❌ Perbaikan JSON gagal (tidak ada respons).")
            return None
        try:
            restaurant = Restaurant.model_validate_json(repaired_output)
        except ValidationError as e2:
            print(f"❌ Perbaikan JSON tetap gagal divalidasi: {e2}")
            return None

    entry = restaurant.model_dump()
    entry["itemId"] = itemId
    return entry


# ---------------------------------------------------------------------------
# Exercise 2: UI utama
# ---------------------------------------------------------------------------

def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        if choice == "1":
            print("\n--- Current Listings ---")
            for i, res in enumerate(data):
                name = res.get("name", res.get("restaurant_name", "N/A"))
                print(f"{i}: {name}")

        elif choice == "2":
            idx_input = input("Enter record index to view: ")
            try:
                idx = int(idx_input)
                if 0 <= idx < len(data):
                    show_restaurant_card(data[idx], idx)
                else:
                    print("invalid index.")
            except ValueError:
                print("invalid index.")
            continue

        elif choice in ["3", "4", "5"]:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != "yes":
                print("Operation cancelled.")
                continue

            if choice == "3":  # ADD NEW DATA
                itemId = 1000000 + len(data) + 1  # id untuk data baru

                new_paragraph = input("Enter the new restaurant description:\n")
                new_entry = new_data_entry_process(new_paragraph, itemId)

                if new_entry is not None:
                    data.append(new_entry)
                    save_data(data, file_path, backup_path)
                    print("✅ Restaurant added.")
                else:
                    print("❌ Gagal memproses data baru, tidak disimpan.")

            elif choice == "4":  # EDIT DATA
                idx_input = input("Enter record index to edit: ")
                try:
                    idx = int(idx_input)
                except ValueError:
                    idx = -1

                if 0 <= idx < len(data):
                    record = data[idx]
                    for key in list(record.keys()):
                        current_value = record[key]
                        new_value = input(
                            f"{key} [{current_value}] (Enter untuk skip): "
                        )
                        if new_value.strip() != "":
                            record[key] = new_value
                    data[idx] = record
                    save_data(data, file_path, backup_path)
                    print("✅ Record updated.")
                else:
                    print("invalid index.")

            elif choice == "5":  # DELETE DATA
                idx_input = input("Enter record index to delete: ")
                try:
                    idx = int(idx_input)
                except (ValueError, TypeError):
                    idx = -1

                if 0 <= idx < len(data):
                    data.pop(idx)
                    save_data(data, file_path, backup_path)
                    print("✅ Record deleted.")
                else:
                    print("invalid index.")

        elif choice == "6":  # EXIT
            break
        else:
            print("Invalid input.")


# ---------------------------------------------------------------------------
# Exercise 3: Unit test — SEMUA panggilan LLM DI-MOCK (tidak makan kuota)
# ---------------------------------------------------------------------------

def _fake_new_data_entry_process(paragraph, itemId):
    """
    Pengganti new_data_entry_process() untuk unit test: tidak memanggil
    Gemini sama sekali, cukup mengembalikan record dummy yang valid.
    """
    return {
        "name": "Mocked Restaurant",
        "location": "Mock City",
        "type": "Mock Type",
        "food_style": "Mock Cuisine",
        "rating": 4.5,
        "price_range": 2,
        "signatures": ["Mock Dish"],
        "vibe": "cozy",
        "environment": "indoor",
        "shortcomings": [],
        "itemId": itemId,
    }


class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        """Buat database sementara yang bersih untuk testing."""
        self.test_file = "structured_restaurant_data_unit_test.json"
        self.test_file_backup = "structured_restaurant_data_unit_test.json.bak"
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, "w") as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Bersihkan file test setelahnya."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch("__main__.new_data_entry_process", side_effect=_fake_new_data_entry_process)
    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input, mock_entry_process):
        """
        Skenario: Tambah restoran baru lalu hapus lagi.
        LLM di-mock total, tidak ada request asli ke Gemini di test ini.
        """
        mock_restaurant = (
            "The Copper Sprout is a high-concept, Modern Appalachian "
            "farm-to-table destination that blends an industrial-chic "
            "aesthetic with rustic forest charm."
        )
        mock_input.side_effect = ["3", "yes", mock_restaurant, "6"]

        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass

        with open(self.test_file, "r") as f:
            data = json.load(f)

        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ["5", "yes", "1", "6"]

        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass

        with open(self.test_file, "r") as f:
            data = json.load(f)

        self.assertEqual(len(data), 1)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Skenario: Coba hapus tapi jawab 'no' di security warning.
        Tidak perlu mock LLM karena LLM tidak akan pernah dipanggil.
        """
        mock_input.side_effect = ["5", "no", "6"]

        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, "r") as f:
            data = json.load(f)

        self.assertEqual(len(data), 1)  # Data harus tidak berubah
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())


if __name__ == "__main__":
    # --- Mode 1: Unit test (default, aman, TIDAK memakai kuota Gemini) ----
    # unittest.main()

    # --- Mode 2: UI asli, benar-benar memanggil Gemini API ----------------
    # Aktifkan baris ini (dan comment unittest.main() di atas) saat kamu
    # mau benar-benar pakai app-nya / ambil screenshot new_data_entry_process.
    manage_restaurants(FILEPATH, BACKUP_PATH)