from fastmcp import FastMCP
from pathlib import Path
import json

mcp = FastMCP("Connoisseur-Server")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CULINARY_MAP_PATH = DATA_DIR / "raw" / "California-Culinary-Map.txt"
RESTAURANT_DATA_PATH = DATA_DIR / "processed" / "structured_restaurant_data.json"
REVIEW_DATA_PATH = DATA_DIR / "processed" / "augmented_user_review.json"

def load_restaurant_data()->list[dict]:
    """Load the structured restaurant data produced in Module 1."""
    with open(RESTAURANT_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_review_data()->list[dict]:
    """Load the augmented user reviews produced in Module 1."""
    with open(REVIEW_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@mcp.resource("culinary-map://california")  
def get_culinary_map() -> str:
    """The full raw California Culinary Map text from Module 1.
    Contains detailed descriptions of 100+ restaurants across California
    including their vibes, cuisines, ratings, and price ranges."""
    return CULINARY_MAP_PATH.read_text(encoding="utf-8")

@mcp.tool()
def get_restaurant_info(restaurant_name: str) -> str:
    """Search for a restaurant by name and return its structured details
    including cuisine, rating, price range, and signature dish."""
    restaurants = load_review_data()
    query = restaurant_name.lower().strip()

    matches = []
    for restaurant in restaurants:
        name = restaurant['name'].lower()
        if query in name or name in query:
            matches.append(restaurant)

    if not matches:
        return json.dumps(
            {
                "status": "not_found",
                "message": f"No restaurant found matching '{restaurant_name}'.",
                "suggestion": "Try a partial name like 'Iron' or 'Sakura'.",
            },
            indent=2,
        )

    return json.dumps(
        {"status": "found", "count": len(matches), "results": matches},
        indent=2,
    )

@mcp.tool()
def recommend_by_vibe(vibe: str)->str:
    """Find restaurants that match a given vibe or atmosphere keyword.
    Searches both structured vibe tags and raw text descriptions.
    Examples of vibe keywords: "moody", "sun-drenched", "romantic"""
    restaurants = load_review_data()
    vibe_lower = vibe.lower().strip()

    structured_matches = []
    for restaurant in restaurants:

        vibes_list = []
        for v in restaurant.get("vibes", []):
            vibes_list.append(v.lower())

        description = restaurant.get("description", "").lower()

        if any(vibe_lower in v for v in vibes_list) or vibe_lower in description:
            structured_matches.append(
                {
                    "name": restaurant["name"],
                    "neighborhood": restaurant["neighborhood"],
                    "cuisine": restaurant["cuisine"],
                    "rating": restaurant["rating"],
                    "vibes": restaurant["vibes"],
                    "price_range": restaurant["price_range"],
                }
             )

    raw_text = CULINARY_MAP_PATH.read_text(encoding="utf-8")
    paragraphs = raw_text.split("\n\n")
    text_excerpts = []
    for para in paragraphs:
        if vibe_lower in para.lower() and para.strip():
            text_excerpts.append(para.strip()[:300])


    return json.dumps(
        {
            "vibe_searched": vibe,
            "structured_matches": structured_matches,
            "raw_text_excerpts": text_excerpts[:5],
        },
         indent=2
    )

# TOOL 3 — Get Review (Returns Review Data for Lab 2 Demonstration)
@mcp.tool()
def get_review(restaurant_name: str) -> str:
    """Retrieve the full review for a restaurant."""
    reviews = load_review_data()
    query = restaurant_name.lower().strip()

    # Find the matching review
    matching_review = None
    for review in reviews:
        if query in review["name"].lower():
            matching_review = review
            break
    
    # Return a not found message if no review matches the query
    if not matching_review:
        return json.dumps(
            {
                "status": "not_found" , #TODO
                "message": f"We Cannot find review for {query}" , #TODO
            },
            indent=2,
        )

    return json.dumps(
        {
            "status": "found",
            "restaurant": matching_review["name"],
            "reviewer": matching_review["reviewer"],
            "rating": matching_review["rating"],
            "review_text": matching_review["review_text"],
            "image_description": matching_review.get("image_description", "N/A"),
            "visit_date": matching_review.get("visit_date", "N/A"),
        },
        indent=2,
    )

if __name__ == "__main__":
    mcp.run(show_banner=False)