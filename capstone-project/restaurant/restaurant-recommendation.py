import os
import time
import json
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY tidak ditemukan! Pastikan file .env sudah berisi GOOGLE_API_KEY=your_key_here")

client = genai.Client(api_key=api_key)


MODEL_NAME = "gemini-3.6-flash"

OUTPUT_FILE = "structured_restaurant_data.json"
INPUT_FILE = "California-Culinary-Map.txt"
BATCH_SIZE = 5         
SLEEP_BETWEEN_CALLS = 5  

class Restaurant(BaseModel):
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


class RestaurantBatch(BaseModel):
    restaurants: List[Restaurant]


def safe_llm_call(system_msg, prompt_txt, response_schema=None, retries=3):
    """
    response_schema: kalau diisi, Gemini akan dipaksa balikin JSON sesuai schema
    (menggunakan structured output bawaan API), jadi validasi Pydantic manual
    hampir selalu langsung berhasil -> tidak perlu repair call lagi.
    """
    config_kwargs = {"system_instruction": system_msg}
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    for i in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_txt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return response.text
        except Exception as e:
            error_msg = str(e)
            print(f"Terjadi kesalahan, mencoba lagi ({i + 1}/{retries}): {error_msg}")

            if "RESOURCE_EXHAUSTED" in error_msg and "PerDay" in error_msg:
                # Ini kuota HARIAN. Nunggu detik/menit nggak akan membantu.
                # Lebih baik hentikan proses sekarang, simpan progres, lanjut besok.
                print("\n>>> KUOTA HARIAN HABIS. Menghentikan proses agar progres tidak hilang.")
                print(">>> Jalankan lagi script ini nanti (misal besok) -- restoran yang")
                print(">>> sudah berhasil diproses TIDAK akan diulang.\n")
                raise SystemExit(1)
            elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                time.sleep(30)
            else:
                time.sleep(5)

    return None


def load_existing_results():
    """Checkpointing: load hasil yang sudah pernah berhasil diproses."""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        done_names = {item["name"] for item in data}
        return data, done_names
    return [], set()


def save_results(results):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def build_batch_prompt(paragraphs):
    joined = "\n\n---\n\n".join(paragraphs)
    system_msg = (
        "You are a data extraction assistant. Extract restaurant information "
        "into a strict JSON array, one object per restaurant, in the same order "
        "as given. Convert price range dollar signs (e.g. $, $$, $$$) into an "
        "integer (1, 2, 3)."
    )
    user_prompt = f"""
    Task: Extract structured attributes for EACH restaurant description below.
    Return a JSON object with a single key "restaurants" containing a list,
    one entry per restaurant, in the same order they appear.

    Restaurant descriptions (separated by ---):
    {joined}
    """
    return system_msg, user_prompt


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = f.read()
    restaurant_list = data.split("\n\n")[1:]

    results, done_names = load_existing_results()
    print(f"Sudah ada {len(results)} restoran tervalidasi sebelumnya (akan di-skip).")

    processed_indices = {item.get("sourceIndex") for item in results}
    remaining = [
        (idx, para) for idx, para in enumerate(restaurant_list)
        if idx not in processed_indices
    ]

    if not remaining:
        print("Semua restoran sudah diproses sebelumnya. Selesai.")
        return

    print(f"Memproses {len(remaining)} restoran tersisa dalam batch of {BATCH_SIZE}...")

    for batch in chunk_list(remaining, BATCH_SIZE):
        indices = [idx for idx, _ in batch]
        paragraphs = [para for _, para in batch]

        system_msg, user_prompt = build_batch_prompt(paragraphs)

        raw_output = safe_llm_call(system_msg, user_prompt, response_schema=RestaurantBatch)
        if raw_output is None:
            print(f"Gagal memproses batch index {indices}, dilewati.")
            continue

        try:
            parsed = RestaurantBatch.model_validate_json(raw_output)
        except ValidationError as e:
            print(f"Batch {indices} gagal validasi setelah structured output: {e}")
            continue

        for idx, restaurant in zip(indices, parsed.restaurants):
            entry = restaurant.model_dump()
            entry["sourceIndex"] = idx
            entry["itemId"] = 1000001 + idx
            results.append(entry)

        save_results(results)
        print(f"Batch {indices} selesai & disimpan. Total sejauh ini: {len(results)}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nSelesai! Total {len(results)} restoran tersimpan di '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()