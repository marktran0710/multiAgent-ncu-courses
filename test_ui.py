import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "static" / "index.html"
BENCHMARK_HTML = ROOT / "static" / "benchmark.html"


class StaticHtmlParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.classes = set()
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if "class" in attrs:
            self.classes.update(attrs["class"].split())

    def handle_data(self, data):
        cleaned = data.strip()
        if cleaned:
            self.text.append(cleaned)


class TestInternationalAdvisorUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX_HTML.read_text(encoding="utf-8")
        cls.parser = StaticHtmlParser()
        cls.parser.feed(cls.html)
        cls.visible_text = " ".join(cls.parser.text)

    def test_international_advising_branding_is_present(self):
        self.assertIn("International Course Advisor", self.visible_text)
        self.assertIn("National Central University", self.visible_text)
        self.assertIn("Academic Advising Desk", self.visible_text)
        self.assertIn("Student Profile", self.visible_text)
        self.assertIn("Advising Trail", self.visible_text)

    def test_core_interactive_controls_exist(self):
        expected_ids = {
            "chatContainer",
            "quickPrompts",
            "messageInput",
            "sendButton",
            "clearChatButton",
            "benchmarkPageButton",
            "adminModeButton",
            "adminOverlay",
            "adminLoginForm",
            "adminDashboard",
            "adminCoursesList",
            "adminLogsList",
            "profileState",
            "profileMeter",
            "profileMeterLabel",
            "activityList",
            "toast",
        }
        self.assertTrue(expected_ids.issubset(self.parser.ids))

    def test_professional_context_strip_exists(self):
        for class_name in ["context-strip", "context-item", "context-label", "context-value"]:
            self.assertIn(class_name, self.parser.classes)

        self.assertIn("Pathway", self.visible_text)
        self.assertIn("Undergraduate to PhD", self.visible_text)
        self.assertIn("English / Chinese", self.visible_text)
        self.assertIn("Fit and eligibility", self.visible_text)

    def test_prompt_groups_cover_international_student_needs(self):
        self.assertIn("const promptGroups", self.html)
        for group in ["Goals", "Background", "Language", "Schedule"]:
            self.assertRegex(self.html, rf"\b{group}\s*:")

        expected_prompts = [
            "I want to study machine learning",
            "I am a first-year undergraduate student",
            "Recommend an English-taught AI course",
            "I prefer Tuesday or Thursday classes",
        ]
        for prompt in expected_prompts:
            self.assertIn(prompt, self.html)

    def test_message_rendering_avoids_raw_html_injection(self):
        self.assertIn("bubble.textContent = text;", self.html)
        self.assertNotIn("messageDiv.innerHTML", self.html)

    def test_profile_meter_updates_from_profile_signals(self):
        self.assertIn("function updateProfileMeter(profile)", self.html)
        self.assertIn("profileMeter.style.width", self.html)
        self.assertIn("profileState.textContent", self.html)
        self.assertRegex(self.html, r"completed_courses\s*&&\s*profile\.completed_courses\.length")

    def test_composer_has_character_limit_and_loading_state(self):
        self.assertIn("const maxChars = 500;", self.html)
        self.assertIn("function setSending(value)", self.html)
        self.assertIn("sendButton.disabled", self.html)
        self.assertIn("characterCount.textContent", self.html)

    def test_realtime_chat_uses_websocket_with_http_fallback(self):
        self.assertIn("new WebSocket", self.html)
        self.assertIn("/ws/chat", self.html)
        self.assertIn("function handleRealtimeEvent(event)", self.html)
        self.assertIn("HTTP fallback", self.html)

    def test_management_is_embedded_in_main_ui(self):
        self.assertIn("Management", self.visible_text)
        self.assertIn("Review advising activity", self.visible_text)
        self.assertIn("Enter Management", self.visible_text)
        self.assertIn("Dashboard > Environment > ADMIN_PASSWORD", self.visible_text)
        self.assertIn("function openAdminMode()", self.html)
        self.assertIn("function loginAdmin(event)", self.html)
        self.assertIn("function bypassAdminLogin(options = {})", self.html)
        self.assertIn("bypassAdminLogin({ silent: true })", self.html)
        self.assertIn("/admin/status", self.html)
        self.assertIn("/admin/dev-login", self.html)
        self.assertIn("/admin/courses", self.html)
        self.assertIn("/admin/logs", self.html)
        self.assertIn("/admin/benchmark", self.html)
        self.assertIn("Run Retrieval Benchmark", self.visible_text)
        self.assertIn("function renderAdminBenchmark()", self.html)
        self.assertIn("Past BM25 + Vector FusionAgent", self.html)
        self.assertIn("Special Case Comparison", self.html)
        self.assertIn("Current Query RAG", self.html)

    def test_admin_logs_render_algorithm_rankings(self):
        self.assertIn("function renderRankingMethods(log)", self.html)
        self.assertIn("retrieval_methods", self.html)
        self.assertIn("query_rag", self.html)
        self.assertIn("Ranking results from all algorithms", self.html)
        self.assertIn("final_query_rag_fusion", self.html)
        self.assertIn("standard_score", self.html)
        self.assertIn("Raw scores use different units", self.html)

    def test_public_benchmark_page_compares_old_and_current_versions(self):
        html = BENCHMARK_HTML.read_text(encoding="utf-8")
        parser = StaticHtmlParser()
        parser.feed(html)
        visible_text = " ".join(parser.text)

        self.assertIn("Benchmark: Old FusionAgent vs Current Query RAG", visible_text)
        self.assertIn("BM25Agent + VectorAgent using FusionAgent", visible_text)
        self.assertIn("percentage_point_change", html)
        self.assertIn("relative_percent_change", html)
        self.assertIn("profile_benchmark", html)
        self.assertIn("Profile-Aware Cases", visible_text)
        self.assertIn("/benchmark/data", html)
        self.assertIn("Special Cases", visible_text)
        self.assertIn("Same-rank cases with identical top results", visible_text)
        self.assertIn("formatChange", html)
        self.assertIn("top 3 changed", html)

    def test_no_old_generic_branding_remains(self):
        self.assertNotIn("<h1>Course Finder</h1>", self.html)
        self.assertNotIn("Recommendation Chat", self.html)


if __name__ == "__main__":
    unittest.main()
