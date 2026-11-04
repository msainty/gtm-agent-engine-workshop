import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from gtm_agent import data_service


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        records = patch.object(data_service, "PROSPECTS", deepcopy(data_service.PROSPECTS))
        profiles = patch.object(data_service, "_PROFILES", {})
        records.start()
        profiles.start()
        self.addCleanup(records.stop)
        self.addCleanup(profiles.stop)

    def test_update_persists_and_refreshes_profile_before_scoring(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
            from gtm_agent import gtm_agent

            cached = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})
            self.assertNotIn("Kafka", cached["prospect_profile"]["tech_stack"])

            result = gtm_agent.update_prospect_info.invoke({
                "prospect_id": "LEAD-71001", "technology": "Kafka",
            })
            self.assertTrue(result["updated"])
            self.assertIn("Kafka", data_service.get_prospect_record("LEAD-71001")["tech_stack"])
            refreshed = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})
            self.assertIn("Kafka", refreshed["prospect_profile"]["tech_stack"])

            def score(messages):
                offering_text, profile_text = messages[1]["content"].split("\n\nProspect profile:\n")
                offering = json.loads(offering_text.removeprefix("Offering:\n"))
                profile = json.loads(profile_text)
                missing = set(offering["required_tech_stack"]) - set(profile["tech_stack"])
                self.assertNotIn("Kafka", missing)
                return gtm_agent.ProspectScore(
                    score=100,
                    justification="Missing technologies: " + ", ".join(sorted(missing)),
                    rubric_breakdown=gtm_agent.RubricBreakdown(
                        revenue_fit=100, tech_stack_match=100, segment_fit=100,
                    ),
                )

            with patch.object(gtm_agent, "_scoring_llm") as model:
                model.invoke.side_effect = score
                scored = gtm_agent.score_prospect.invoke({
                    "prospect_profile": cached["prospect_profile"],
                    "offering": data_service.get_offering("OFFER-10005"),
                })
            self.assertNotIn("Kafka", scored["justification"])
            model.invoke.assert_called_once()

    def test_duplicate_technology_is_not_added_twice(self):
        data_service.update_prospect_info("LEAD-71001", "Kafka")
        result = data_service.update_prospect_info("LEAD-71001", "Kafka")
        self.assertTrue(result["updated"])
        self.assertEqual(data_service.fetch_tech_stack("LEAD-71001").count("Kafka"), 1)

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(
            data_service.update_prospect_info("unknown", "Kafka"),
            {"updated": False, "found": False},
        )

    def test_failed_write_does_not_report_success_or_change_saved_stack(self):
        class ReadOnlyRecord(dict):
            def __setitem__(self, key, value):
                raise OSError("Storage unavailable")

        record = ReadOnlyRecord(data_service.PROSPECTS["LEAD-71001"])
        data_service.PROSPECTS["LEAD-71001"] = record
        with self.assertRaisesRegex(OSError, "Storage unavailable"):
            data_service.update_prospect_info("LEAD-71001", "Kafka")
        self.assertNotIn("Kafka", record["tech_stack"])


if __name__ == "__main__":
    unittest.main()
