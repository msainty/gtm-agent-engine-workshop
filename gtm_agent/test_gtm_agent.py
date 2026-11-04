"""Tests for prospect data exposed by the agent tools."""

import copy
import importlib
import json
import unittest
from unittest.mock import Mock, patch


with patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}), \
        patch("langchain_openai.ChatOpenAI"), \
        patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "disqualified": False,
            "annual_revenue": 50000000,
            "engagement_history": ["Requested a demo"],
            "account_details": {"segment": "Enterprise"},
            "tech_stack": ["AWS"],
            "billing_qualification": {
                "card_on_file": "test-card-value",
                "tax_id": "test-tax-value",
            },
        }
        self.offering = {
            "required_tech_stack": ["AWS"],
            "min_annual_revenue": 1000000,
            "description": "Cloud services",
        }
        self.enterContext(patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}))
        self.enterContext(patch.dict(agent.data_service.PROSPECTS, {"TEST": self.record}))
        self.enterContext(patch.object(agent.data_service, "_PROFILES", {}))
        self.scoring = self.enterContext(patch.object(agent, "_scoring_llm"))
        self.score = {"score": 80, "justification": "Good fit"}
        self.scoring.invoke.return_value = Mock(model_dump=lambda: self.score)

    def assert_no_billing(self, value):
        serialized = json.dumps(value)
        for sensitive in ("billing_qualification", "card_on_file", "tax_id",
                          "test-card-value", "test-tax-value"):
            self.assertNotIn(sensitive, serialized)

    def test_contact_profile_and_scoring_flow(self):
        for has_billing in (True, False):
            with self.subTest(has_billing=has_billing):
                if not has_billing:
                    self.record.pop("billing_qualification")
                agent.data_service._PROFILES.clear()
                contact = agent.get_prospect.invoke({"prospect_id": "TEST"})
                self.assertEqual(contact, {"found": True, "prospect": {
                    "prospect_id": "TEST", "name": "Test Prospect",
                    "email": "prospect@example.com", "disqualified": False,
                }})
                self.assert_no_billing(contact)
                profile = agent.build_prospect_profile.invoke({"prospect_id": "TEST"})
                expected = {"prospect_id": "TEST", **self.record}
                expected.pop("billing_qualification", None)
                self.assertEqual(profile, {"found": True, "prospect_profile": expected})
                self.assert_no_billing(profile)
                self.assert_no_billing(agent.data_service._PROFILES)
                self.assertEqual(agent.build_prospect_profile.invoke(
                    {"prospect_id": "TEST"}), profile)
                arguments = {"prospect_profile": profile["prospect_profile"],
                             "offering": self.offering}
                self.assert_no_billing(arguments)
                self.assertEqual(agent.score_prospect.invoke(arguments), self.score)
                messages = self.scoring.invoke.call_args.args[0]
                self.assert_no_billing(messages)
                scored = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
                self.assertEqual(scored, expected)

    def test_cached_profile_is_filtered(self):
        agent.data_service._PROFILES["TEST"] = {"prospect_id": "TEST", **self.record}
        result = agent.build_prospect_profile.invoke({"prospect_id": "TEST"})
        self.assert_no_billing(result)
        self.assertEqual(result["prospect_profile"]["annual_revenue"], 50000000)

    def test_scoring_filters_supplied_profile_without_mutating_it(self):
        for prospect_id in ("TEST", None):
            with self.subTest(prospect_id=prospect_id):
                supplied = copy.deepcopy(self.record)
                if prospect_id:
                    supplied["prospect_id"] = prospect_id
                original = copy.deepcopy(supplied)
                result = agent.score_prospect.invoke({
                    "prospect_profile": supplied, "offering": self.offering,
                })
                self.assertEqual(result, self.score)
                self.assertEqual(supplied, original)
                self.assert_no_billing(self.scoring.invoke.call_args.args[0])

    def test_missing_prospect(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "MISSING"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "MISSING"}),
                         {"prospect_profile": None, "found": False})


if __name__ == "__main__":
    unittest.main()
