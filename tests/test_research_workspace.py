from __future__ import annotations

import json
from pathlib import Path
import unittest

from research_workspace.workbench import (
    audit_dossier,
    build_boolean_query,
    build_crossref_url,
    deduplicate_sources,
    evidence_family_summary,
    prisma_flow_errors,
    research_fingerprint,
    should_stop_searching,
)


ROOT = Path(__file__).resolve().parents[1]


class ResearchWorkspaceTests(unittest.TestCase):
    def test_boolean_query_uses_or_within_and_between_facets(self) -> None:
        query = build_boolean_query(
            {
                "对象": ["automatic speech recognition", "ASR"],
                "问题": ["accent robustness", "speaker generalization"],
            }
        )
        self.assertEqual(
            query,
            '("automatic speech recognition" OR ASR) AND ("accent robustness" OR "speaker generalization")',
        )

    def test_crossref_url_is_bounded_and_encoded(self) -> None:
        url = build_crossref_url("speech recognition", rows=25, from_year=2020)
        self.assertTrue(url.startswith("https://api.crossref.org/works?"))
        self.assertIn("rows=25", url)
        self.assertIn("query.bibliographic=speech+recognition", url)
        self.assertIn("from-pub-date%3A2020-01-01", url)

    def test_invalid_crossref_rows_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_crossref_url("ASR", rows=1001)

    def test_deduplication_prefers_doi(self) -> None:
        sources = [
            {"title": "First title", "doi": "10.1/ABC"},
            {"title": "Renamed title", "DOI": "https://doi.org/10.1/abc"},
        ]
        self.assertEqual(len(deduplicate_sources(sources)), 1)

    def test_source_families_prevent_copy_count_inflation(self) -> None:
        summary = evidence_family_summary(
            [
                {"id": "S1", "source_family": "study-a", "direct_evidence": True},
                {"id": "S2", "source_family": "study-a", "direct_evidence": False},
                {"id": "S3", "source_family": "study-b", "direct_evidence": True},
            ]
        )
        self.assertEqual(summary["report_count"], 3)
        self.assertEqual(summary["family_count"], 2)
        self.assertEqual(summary["direct_family_count"], 2)

    def test_stop_rule_exposes_boundary(self) -> None:
        result = should_stop_searching([5, 2, 0, 0, 0], window=3)
        self.assertTrue(result["stop"])
        self.assertIn("不是完整性证明", result["boundary"])

    def test_prisma_flow_arithmetic(self) -> None:
        flow = {
            "identified": 10,
            "duplicates_removed": 2,
            "screened": 8,
            "excluded_screening": 3,
            "full_text_assessed": 5,
            "excluded_full_text": 2,
            "included": 3,
        }
        self.assertEqual(prisma_flow_errors(flow), [])
        flow["included"] = 4
        self.assertTrue(prisma_flow_errors(flow))

    def test_example_dossier_is_valid_and_stable(self) -> None:
        path = ROOT / "research_workspace" / "example_dossier.json"
        dossier = json.loads(path.read_text(encoding="utf-8"))
        report = audit_dossier(dossier)
        self.assertEqual(report["errors"], [])
        self.assertEqual(research_fingerprint(dossier), research_fingerprint(dossier))
        self.assertEqual(len(research_fingerprint(dossier)), 64)

    def test_ai_summary_is_navigation_not_evidence(self) -> None:
        dossier = json.loads(
            (ROOT / "research_workspace" / "example_dossier.json").read_text(encoding="utf-8")
        )
        dossier["sources"].append(
            {
                "id": "S4",
                "title": "AI generated summary",
                "url": "https://example.invalid/summary",
                "source_type": "ai_summary",
                "source_family": "summary",
                "direct_evidence": False,
            }
        )
        report = audit_dossier(dossier)
        self.assertTrue(any("not direct evidence" in warning for warning in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
