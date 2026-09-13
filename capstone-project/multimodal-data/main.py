import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
import zipfile
import ast
import requests
from tenacity import retry, stop_after_attempt, wait_exponential



from dotenv import load_dotenv
import google.genai as genai
from google.genai import types

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY tidak ditemukan! Pastikan file .env sudah berisi GOOGLE_API_KEY=your_key_here")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.6-flash"
INPUT_FILE_RECIPES = "Recipes.json"
INPUT_FILE_USER_REVIEWS = "Synthetic-User-Reviews.json"
INPUT_ZIP_RECIPE_IMAGE = "synthetic_recipe_images.zip"
# url = https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip


with open(INPUT_FILE_RECIPES, "r", encoding="utf-8") as f:
    recipe_data = json.load(f)

for key, value in recipe_data[0].items():
    print(f"{key} ({type(value).__name__}): {value}")

with zipfile.ZipFile("INPUT_ZIP_RECIPE_IMAGE", 'r') as zip_ref:
    zip_ref.extractall()

def vision_llm(system_msg, prompt_txt, image_path):
    # Step 2.1
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Step 2.2
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/png"
    )

    # Step 2.3
    config = types.GenerateContentConfig(
        system_instruction=system_msg,
        max_output_tokens=300
    )

    # Step 2.4
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[image_part, prompt_txt],
        config=config
    )

    return response.text

def image_caption_prompt_template(food_name):
    # System message: menentukan role Gemini
    image_caption_system_msg = (
        "You are an expert food and culinary description assistant. "
        "Describe food images accurately and objectively."
    )

    # User prompt: instruksi untuk gambar tertentu
    image_caption_prompt_txt = (
        f"Describe this image of {food_name} briefly and factually. "
        "Mention visible ingredients, presentation, colors, and textures. "
        "Avoid making assumptions about taste or ingredients that cannot be seen."
    )

    return image_caption_system_msg, image_caption_prompt_txt

# for key, value in recipe_data[0].items():
#     print(f"{key} ({type(value).__name__}): {value}")

# food_name = recipe_data[0]["name"]

# system_msg, prompt_txt = image_caption_prompt_template(food_name)

# for i in range(5):

#     # Menampilkan progress setiap 20 recipe
#     if (i + 1) % 20 == 0:
#         print(f"{i + 1} out of {len(recipe_data)} is done")

#     food_name = recipe_data[i]["name"]

#     system_msg, prompt_txt = image_caption_prompt_template(food_name)

#       # Tentukan path gambar berdasarkan nomor recipe
#     image_path = f"./synthetic_recipe_images/recipe{i + 1}.png"

#     # Kirim gambar + prompt ke Gemini
#     response = vision_llm(
#         system_msg,
#         prompt_txt,
#         image_path
#     )

#     # Simpan hasil caption ke recipe
#     recipe_data[i]["image_description"] = response

# print("ALL DONE!")

# filename = 'augmented_food_recipe.json'
# with open(filename, 'w', encoding='utf-8') as f:
#     json.dump(recipe_data, f, indent=4)

with open(INPUT_FILE_USER_REVIEWS, "r", encoding="utf-8") as f:
    user_review_data = json.load(f)

for key, value in user_review_data[0].items():
    print(f"{key} ({type(value).__name__}): {value}")

# review_images = ast.literal_eval(user_review_data[0]["images"])
# image_data = requests.get(review_images[0])

# with open("review_image_placeholder.jpg", "wb") as img_file:
#     img_file.write(image_data.content)

# image = Image.open("review_image_placeholder.jpg")

# plt.imshow(image)
# plt.axis("off")
# plt.show()

def review_context_image_caption_prompt_template(reviews):

     # System message: menentukan role Gemini
    review_context_image_caption_system_msg = (
        "You are a culinary expert analyzing a customer's food photo together "
        "with their written review. Combine what is visible in the image with "
        "the context from the review to produce a short, accurate description "
        "of the dish and the dining experience it depicts. Do not invent "
        "details that are not supported by either the image or the review."
    )
    review_context_image_caption_prompt_txt = (
        f"Here is a customer review of their visit:\n\n\"{reviews}\"\n\n"
        f"Looking at the attached photo and considering the review above, "
        f"write a 2-3 sentence description of the food/scene shown in the "
        f"image, incorporating relevant details mentioned in the review."
    )

    return (
        review_context_image_caption_system_msg,
        review_context_image_caption_prompt_txt
    )

# reviews = user_review_data[0]["text"]

# system_msg, prompt_txt = review_context_image_caption_prompt_template(
#     reviews
# )

# response = vision_llm(
#     system_msg,
#     prompt_txt,
#     "review_image_placeholder.jpg"
# )

# print(response)

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def get_data_with_retry(url):
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response

for i in range(min(5, len(user_review_data))):
    # Convert string images menjadi Python list
    review_images = ast.literal_eval(user_review_data[i]["images"])

    review_image_captions = []

    if len(review_images) > 0:

        for img_url in review_images:

            try:
                # Download image
                image_data = get_data_with_retry(img_url)
                print("Success!")

            except Exception as e:
                print(f"All retries failed at url {img_url}:", e)
                continue

            # Simpan image sementara
            image = image_data.content

            with open("review_image_placeholder.jpg", "wb") as img_file:
                img_file.write(image)

            # Ambil review text
            reviews = user_review_data[i]["text"]

            # Buat prompt
            system_msg, prompt_txt = review_context_image_caption_prompt_template(
                reviews
            )

            # Kirim image + review ke vision model
            response = vision_llm(
                system_msg,
                prompt_txt,
                "review_image_placeholder.jpg"
            )

            # Simpan caption
            review_image_captions.append(response)

    # Tambahkan caption ke data review
    user_review_data[i]["image_captions"] = review_image_captions

print("ALL DONE!")

filename = "augmented_user_review.json"

with open(filename, "w", encoding="utf-8") as f:
    json.dump(user_review_data, f, indent=4, ensure_ascii=False)

print(f"Saved to {filename}")