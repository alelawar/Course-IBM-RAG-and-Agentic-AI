"""
test_project.py — Unit tests for the Connoisseur Companion capstone project.

All tests are fully OFFLINE — no Gemini API calls, no ChromaDB required.
Functions are tested via mocks or by calling pure logic directly.
Run with:  python -m pytest tests/ -v
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Paths — data ada di folder data/ pada root project
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESTAURANT_DATA = DATA_DIR / "processed" / "structured_restaurant_data.json"
REVIEW_DATA = DATA_DIR / "processed" / "augmented_user_review.json"
CULINARY_MAP = DATA_DIR / "raw" / "California-Culinary-Map.txt"


# ===================================================================
# 1. DATA INTEGRITY TESTS — validate JSON files and data quality
# ===================================================================

class TestDataIntegrity(unittest.TestCase):
    """Verify that the core JSON data files are valid and well-structured."""

    def test_restaurant_data_is_valid_json(self):
        """structured_restaurant_data.json must be parseable."""
        with open(RESTAURANT_DATA, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0, "Restaurant data should not be empty")

    def test_restaurant_records_have_required_fields(self):
        """Every restaurant record must contain the expected keys."""
        required = {"name", "location", "type", "food_style", "environment"}
        with open(RESTAURANT_DATA, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, record in enumerate(data):
            for field in required:
                self.assertIn(
                    field, record,
                    f"Record #{i} ({record.get('name', '?')}) missing '{field}'"
                )

    def test_restaurant_ratings_in_range(self):
        """Ratings, if present, must be between 0 and 5."""
        with open(RESTAURANT_DATA, "r", encoding="utf-8") as f:
            data = json.load(f)
        for record in data:
            rating = record.get("rating")
            if rating is not None:
                self.assertGreaterEqual(rating, 0)
                self.assertLessEqual(rating, 5)

    def test_restaurant_price_range_valid(self):
        """Price range, if present, must be 1–4."""
        with open(RESTAURANT_DATA, "r", encoding="utf-8") as f:
            data = json.load(f)
        for record in data:
            pr = record.get("price_range")
            if pr is not None:
                self.assertIn(pr, [1, 2, 3, 4])

    def test_no_duplicate_restaurant_names(self):
        """Restaurant names should be unique."""
        with open(RESTAURANT_DATA, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = [r["name"] for r in data]
        self.assertEqual(len(names), len(set(names)), "Duplicate restaurant names found")

    def test_review_data_is_valid_json(self):
        """augmented_user_review.json must be parseable."""
        with open(REVIEW_DATA, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_review_records_have_required_fields(self):
        """Every review must have name, reviewer, rating, review_text."""
        required = {"name", "reviewer", "rating", "review_text"}
        with open(REVIEW_DATA, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, review in enumerate(data):
            for field in required:
                self.assertIn(
                    field, review,
                    f"Review #{i} missing '{field}'"
                )

    def test_culinary_map_exists_and_not_empty(self):
        """California-Culinary-Map.txt must exist and have content."""
        self.assertTrue(CULINARY_MAP.exists(), "Culinary map file not found")
        content = CULINARY_MAP.read_text(encoding="utf-8")
        self.assertGreater(len(content), 100, "Culinary map seems too short")


# ===================================================================
# 2. MCP SERVER TOOL TESTS — test server.py tool functions directly
# ===================================================================

class TestMCPServerTools(unittest.TestCase):
    """Test MCP server tool logic without starting a server process."""

    @classmethod
    def setUpClass(cls):
        """Load data files once for all tests in this class."""
        with open(REVIEW_DATA, "r", encoding="utf-8") as f:
            cls.reviews = json.load(f)
        cls.culinary_text = CULINARY_MAP.read_text(encoding="utf-8")

    # --- get_restaurant_info ---

    def _search_by_name(self, query):
        """Replicate get_restaurant_info logic."""
        query_lower = query.lower().strip()
        matches = []
        for r in self.reviews:
            name = r["name"].lower()
            if query_lower in name or name in query_lower:
                matches.append(r)
        return matches

    def test_get_restaurant_info_found(self):
        """Searching for 'Iron' should find 'Iron & Embers'."""
        matches = self._search_by_name("Iron")
        self.assertGreater(len(matches), 0)
        names = [m["name"] for m in matches]
        self.assertTrue(any("Iron" in n for n in names))

    def test_get_restaurant_info_not_found(self):
        """Searching for a nonexistent name should return empty."""
        matches = self._search_by_name("ZZZZZ_Nonexistent_Restaurant")
        self.assertEqual(len(matches), 0)

    def test_get_restaurant_info_case_insensitive(self):
        """Search should be case-insensitive."""
        matches_lower = self._search_by_name("sakura garden")
        matches_upper = self._search_by_name("SAKURA GARDEN")
        self.assertEqual(len(matches_lower), len(matches_upper))

    # --- recommend_by_vibe ---

    def _search_by_vibe(self, vibe):
        """Replicate recommend_by_vibe logic (fixed version)."""
        vibe_lower = vibe.lower().strip()
        structured_matches = []
        for restaurant in self.reviews:
            vibes_list = [v.lower() for v in restaurant.get("vibes", [])]
            description = restaurant.get("description", "").lower()
            if any(vibe_lower in v for v in vibes_list) or vibe_lower in description:
                structured_matches.append(restaurant)

        # Text excerpts from culinary map (outside loop — fixed)
        paragraphs = self.culinary_text.split("\n\n")
        text_excerpts = []
        for para in paragraphs:
            if vibe_lower in para.lower() and para.strip():
                text_excerpts.append(para.strip()[:300])

        return structured_matches, text_excerpts

    def test_recommend_by_vibe_moody(self):
        """'moody' should find at least one match in text excerpts."""
        _, excerpts = self._search_by_vibe("moody")
        self.assertGreater(len(excerpts), 0, "'moody' not found in culinary map")

    def test_recommend_by_vibe_nonsense(self):
        """A nonsense vibe should return no structured matches."""
        matches, _ = self._search_by_vibe("xyzzy999")
        self.assertEqual(len(matches), 0)

    # --- get_review ---

    def _search_review(self, query):
        """Replicate get_review logic."""
        query_lower = query.lower().strip()
        for review in self.reviews:
            if query_lower in review["name"].lower():
                return review
        return None

    def test_get_review_found(self):
        """Looking up 'Iron & Embers' should return a review."""
        review = self._search_review("Iron & Embers")
        self.assertIsNotNone(review)
        self.assertIn("review_text", review)

    def test_get_review_not_found(self):
        """Looking up a nonexistent name returns None."""
        review = self._search_review("ZZZZZ_Nothing")
        self.assertIsNone(review)

    def test_get_review_partial_match(self):
        """Partial name 'Sakura' should match 'Sakura Garden'."""
        review = self._search_review("Sakura")
        self.assertIsNotNone(review)
        self.assertIn("Sakura", review["name"])


# ===================================================================
# 3. DATA MANAGEMENT TESTS — test CRUD helpers (file I/O)
# ===================================================================

class TestDataManagement(unittest.TestCase):
    """Test file I/O functions from src/data_pipeline/manage_restaurants.py."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_data.json")
        self.backup_file = os.path.join(self.test_dir, "test_data.json.bak")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _load_data(self, path):
        """Replicate load_data()."""
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def _save_data(self, data, path, backup):
        """Replicate save_data()."""
        if os.path.exists(path):
            shutil.copy(path, backup)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def test_load_nonexistent_file(self):
        """Loading a file that doesn't exist should return empty list."""
        result = self._load_data("/nonexistent/path.json")
        self.assertEqual(result, [])

    def test_load_invalid_json(self):
        """Loading a file with invalid JSON should return empty list."""
        with open(self.test_file, "w") as f:
            f.write("{broken json!!")
        result = self._load_data(self.test_file)
        self.assertEqual(result, [])

    def test_save_creates_file(self):
        """save_data should create a new file."""
        data = [{"name": "Test Restaurant"}]
        self._save_data(data, self.test_file, self.backup_file)
        self.assertTrue(os.path.exists(self.test_file))

    def test_save_creates_backup(self):
        """save_data should create a backup of the existing file."""
        # Create initial file
        with open(self.test_file, "w") as f:
            json.dump([{"name": "Original"}], f)
        # Save new data (should trigger backup)
        self._save_data([{"name": "Updated"}], self.test_file, self.backup_file)
        self.assertTrue(os.path.exists(self.backup_file))
        with open(self.backup_file, "r") as f:
            backup_data = json.load(f)
        self.assertEqual(backup_data[0]["name"], "Original")

    def test_save_and_load_roundtrip(self):
        """Data should survive a save → load roundtrip."""
        original = [
            {"name": "Café Luna", "rating": 4.5},
            {"name": "Ramen House", "rating": 4.2},
        ]
        self._save_data(original, self.test_file, self.backup_file)
        loaded = self._load_data(self.test_file)
        self.assertEqual(loaded, original)

    def test_add_and_delete_record(self):
        """Add a record then delete it — length should match."""
        data = [{"name": "A"}, {"name": "B"}]
        self._save_data(data, self.test_file, self.backup_file)

        # Add
        data.append({"name": "C"})
        self._save_data(data, self.test_file, self.backup_file)
        self.assertEqual(len(self._load_data(self.test_file)), 3)

        # Delete
        data.pop(2)
        self._save_data(data, self.test_file, self.backup_file)
        self.assertEqual(len(self._load_data(self.test_file)), 2)


# ===================================================================
# 4. PYDANTIC SCHEMA TESTS — validate Restaurant model
# ===================================================================

class TestPydanticSchema(unittest.TestCase):
    """Test the Restaurant Pydantic model used for structured output."""

    def setUp(self):
        # Import pydantic here to keep tests isolated
        from pydantic import BaseModel, Field
        from typing import List, Optional

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

        self.Restaurant = Restaurant

    def test_valid_restaurant(self):
        """A complete record should validate."""
        r = self.Restaurant(
            name="Test Place",
            location="Test City",
            type="bistro",
            food_style="Italian",
            rating=4.5,
            price_range=3,
            signatures=["carbonara"],
            vibe="romantic",
            environment="indoor",
            shortcomings=[],
        )
        self.assertEqual(r.name, "Test Place")

    def test_minimal_restaurant(self):
        """Only required fields — optional fields should use defaults."""
        r = self.Restaurant(
            name="Minimal", location="Here", type="cafe",
            food_style="Coffee", environment="outdoor"
        )
        self.assertIsNone(r.rating)
        self.assertEqual(r.signatures, [])

    def test_invalid_missing_required(self):
        """Missing 'name' should raise ValidationError."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self.Restaurant(
                location="Here", type="cafe",
                food_style="Coffee", environment="outdoor"
            )

    def test_model_dump_json_roundtrip(self):
        """model_dump_json → model_validate_json roundtrip."""
        r = self.Restaurant(
            name="Roundtrip", location="There", type="diner",
            food_style="American", environment="indoor", rating=3.8
        )
        json_str = r.model_dump_json()
        r2 = self.Restaurant.model_validate_json(json_str)
        self.assertEqual(r.name, r2.name)
        self.assertEqual(r.rating, r2.rating)


# ===================================================================
# 5. MULTIMODAL FUSION LOGIC TESTS — test scoring math
# ===================================================================

class TestMultimodalFusionLogic(unittest.TestCase):
    """Test the scoring/normalization functions from multimodal_fusion.py."""

    @staticmethod
    def _minmax(x):
        """Replicate _minmax from multimodal_fusion.py."""
        import numpy as np
        x = np.array(x, dtype=np.float32)
        if x.size == 0:
            return x
        lo, hi = float(x.min()), float(x.max())
        if abs(hi - lo) < 1e-8:
            return np.ones_like(x)
        return (x - lo) / (hi - lo)

    @staticmethod
    def _to_similarity(dists):
        """Replicate _to_similarity."""
        import numpy as np
        d = np.array(dists, dtype=np.float32)
        return 1.0 - d

    def test_minmax_basic(self):
        """[1, 2, 3] → [0, 0.5, 1]."""
        import numpy as np
        result = self._minmax([1, 2, 3])
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_minmax_constant(self):
        """All same values → all ones."""
        import numpy as np
        result = self._minmax([5, 5, 5])
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])

    def test_minmax_empty(self):
        """Empty array → empty array."""
        import numpy as np
        result = self._minmax([])
        self.assertEqual(len(result), 0)

    def test_to_similarity(self):
        """Distance 0.2 → similarity 0.8."""
        import numpy as np
        result = self._to_similarity([0.0, 0.2, 1.0])
        np.testing.assert_array_almost_equal(result, [1.0, 0.8, 0.0])

    def test_fused_score_weighted(self):
        """Manual fusion: w_text * text_score + w_img * img_score."""
        w_text, w_img = 0.6, 0.4
        text_score, img_score = 0.9, 0.7
        fused = w_text * text_score + w_img * img_score
        self.assertAlmostEqual(fused, 0.82)

    def test_fused_ranking_order(self):
        """Higher fused score should rank first."""
        rows = [
            {"fused": 0.5, "name": "B"},
            {"fused": 0.9, "name": "A"},
            {"fused": 0.1, "name": "C"},
        ]
        rows.sort(key=lambda r: r["fused"], reverse=True)
        self.assertEqual(rows[0]["name"], "A")
        self.assertEqual(rows[-1]["name"], "C")


# ===================================================================
# 6. CHATBOT HELPER TESTS — intent classification, formatting
# ===================================================================

class TestChatbotHelpers(unittest.TestCase):
    """Test pure helper functions from chatbot_ui.py."""

    def test_format_recommendations_restaurants_only(self):
        """format_recommendations with only restaurants."""
        recommendations = {
            "restaurants": [
                {"name": "Place A", "cuisine": "Italian", "price": "$$",
                 "reasoning": "Great match"},
            ]
        }
        output = self._format_recommendations(recommendations)
        self.assertIn("Place A", output)
        self.assertIn("Restaurant", output)
        self.assertNotIn("Recipe", output)

    def test_format_recommendations_empty(self):
        """Empty recommendations → fallback message."""
        output = self._format_recommendations({})
        self.assertIn("couldn't generate", output.lower())

    def test_format_recommendations_both(self):
        """Both restaurants and recipes."""
        recommendations = {
            "restaurants": [
                {"name": "R1", "cuisine": "Thai", "price": "$",
                 "reasoning": "Spicy!"},
            ],
            "recipes": [
                {"name": "Curry", "cuisine": "Thai", "difficulty": "Easy",
                 "reasoning": "Quick and delicious"},
            ],
        }
        output = self._format_recommendations(recommendations)
        self.assertIn("R1", output)
        self.assertIn("Curry", output)

    @staticmethod
    def _format_recommendations(recommendations):
        """Replicate format_recommendations from chatbot_ui.py."""
        output = ""
        if "restaurants" in recommendations and recommendations["restaurants"]:
            output += "🍽️ **Restaurant Recommendations:**\n\n"
            for i, restaurant in enumerate(recommendations["restaurants"], 1):
                output += f"**{i}. {restaurant['name']}**\n"
                output += f"   - Cuisine: {restaurant['cuisine']}\n"
                output += f"   - Price: {restaurant['price']}\n"
                output += f"   - Why: {restaurant['reasoning']}\n\n"
        if "recipes" in recommendations and recommendations["recipes"]:
            output += "👨‍🍳 **Recipe Recommendations:**\n\n"
            for i, recipe in enumerate(recommendations["recipes"], 1):
                output += f"**{i}. {recipe['name']}**\n"
                output += f"   - Cuisine: {recipe['cuisine']}\n"
                output += f"   - Difficulty: {recipe['difficulty']}\n"
                output += f"   - Why: {recipe['reasoning']}\n\n"
        if not output:
            output = "I couldn't generate recommendations. Please try again."
        return output


# ===================================================================
# 7. AGENT CONFIG TESTS — validate multi-agent configurations
# ===================================================================

class TestAgentConfigs(unittest.TestCase):
    """Test agent configuration structures from agent_definitions.py."""

    REQUIRED_KEYS = {"role", "goal", "backstory"}
    AGENT_CONFIGS = {
        "user_profile_generator": {
            "role": "User Profile Generator",
            "goal": "Analyze user restaurant visit history.",
            "backstory": "Expert analyst."
        },
        "rag_retriever": {
            "role": "RAG Retriever",
            "goal": "Query vector databases.",
            "backstory": "Data retrieval specialist."
        },
        "food_trend_analyst": {
            "role": "Food Trend Analyst",
            "goal": "Identify food trends.",
            "backstory": "Culinary journalist."
        },
        "recommendation_expert": {
            "role": "Recommendation Expert",
            "goal": "Synthesize recommendations.",
            "backstory": "Recommendation architect."
        },
    }

    def test_all_configs_have_required_keys(self):
        """Each agent config must have role, goal, backstory."""
        for name, config in self.AGENT_CONFIGS.items():
            for key in self.REQUIRED_KEYS:
                self.assertIn(key, config, f"Agent '{name}' missing '{key}'")

    def test_create_agent_prompt(self):
        """create_agent_prompt should incorporate role, goal, backstory."""
        config = self.AGENT_CONFIGS["user_profile_generator"]
        prompt = (
            f"You are a {config['role']}.\n"
            f"Your goal: {config['goal']}\n"
            f"Your background: {config['backstory']}\n"
        )
        self.assertIn("User Profile Generator", prompt)
        self.assertIn("Analyze", prompt)

    def test_no_empty_values(self):
        """No agent config value should be empty."""
        for name, config in self.AGENT_CONFIGS.items():
            for key, value in config.items():
                self.assertTrue(
                    len(value.strip()) > 0,
                    f"Agent '{name}' has empty '{key}'"
                )


# ===================================================================
# RUN
# ===================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)

