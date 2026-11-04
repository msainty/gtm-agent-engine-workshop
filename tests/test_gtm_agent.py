import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.prospect_id = next(iter(agent.data_service.PROSPECTS))
        self.offering_id = next(iter(agent.data_service.OFFERINGS))
        self.rep = agent.REP_IDS[0]
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": self.rep["rep_id"]}})
        profiles = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        profiles.start()
        self.addCleanup(profiles.stop)

    def test_email_resolves_contact_and_signed_in_sender(self):
        result = agent.send_prospect_email.func(self.prospect_id, "Hello", "Body", self.runtime)
        prospect = agent.data_service.get_prospect_record(self.prospect_id)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], prospect["email"])
        self.assertEqual(result["from"], self.rep["email"])
        self.assertEqual(result["from_name"], self.rep["name"])
        self.assertEqual(result["subject"], "Hello")
        self.assertEqual(result["body"], "Body")

    def test_email_unknown_prospect_does_not_send(self):
        with patch.object(agent.uuid, "uuid4") as message_id:
            result = agent.send_prospect_email.func("missing", "Hello", "Body", self.runtime)
        self.assertEqual(result["status"], "failed")
        self.assertIn("error", result)
        self.assertNotIn("message_id", result)
        message_id.assert_not_called()

    def test_score_loads_records_from_ids(self):
        expected = {"score": 70, "justification": "Fit"}
        with patch.object(agent, "_scoring_llm") as llm:
            llm.invoke.return_value.model_dump.return_value = expected
            result = agent.score_prospect.invoke({"prospect_id": self.prospect_id, "offering_id": self.offering_id})
        self.assertEqual(result, expected)
        content = llm.invoke.call_args.args[0][1]["content"]
        offering_text, profile_text = content.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
        self.assertEqual(json.loads(offering_text), agent.data_service.get_offering(self.offering_id))
        profile = json.loads(profile_text)
        self.assertEqual(profile["prospect_id"], self.prospect_id)
        self.assertEqual(profile["tech_stack"], agent.data_service.fetch_tech_stack(self.prospect_id))
        self.assertEqual(profile["account_details"], agent.data_service.fetch_account_details(self.prospect_id))

    def test_invalid_scoring_ids_do_not_call_model(self):
        with patch.object(agent, "_scoring_llm") as llm:
            result = agent.score_prospect.invoke({"prospect_id": self.prospect_id, "offering_id": "missing"})
            self.assertEqual(result, {"score": None, "error": "Cannot score without a valid offering."})
            result = agent.score_prospect.invoke({"prospect_id": "missing", "offering_id": self.offering_id})
            self.assertEqual(result, {"score": None, "error": "Cannot score without a valid prospect."})
        llm.invoke.assert_not_called()

    def test_explicit_lookups_remain_available(self):
        self.assertTrue(agent.get_prospect.invoke({"prospect_id": self.prospect_id})["found"])
        self.assertTrue(agent.lookup_offering.invoke({"offering_id": self.offering_id})["found"])
        self.assertTrue(agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})["found"])

    def test_workflows_complete_with_one_tool_turn(self):
        cases = [
            ("send_prospect_email", {"prospect_id": self.prospect_id, "subject": "Hello", "body": "Body"}),
            ("score_prospect", {"prospect_id": self.prospect_id, "offering_id": self.offering_id}),
        ]
        for name, args in cases:
            with self.subTest(tool=name):
                model = ScriptedModel(messages=iter([
                    AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}]),
                    AIMessage(content="Done"),
                ]))
                graph = agent.create_deep_agent(
                    model=model,
                    tools=[agent.send_prospect_email, agent.score_prospect, agent.get_current_rep,
                           agent.get_prospect, agent.lookup_offering, agent.build_prospect_profile],
                    system_prompt=agent.SYSTEM_PROMPT,
                )
                with patch.object(agent, "gtm_agent", graph), patch.object(agent, "_scoring_llm") as llm:
                    llm.invoke.return_value.model_dump.return_value = {"score": 70}
                    result = agent.run_agent("Complete this request", user_id=self.rep["rep_id"])
                self.assertEqual(result["reply"], "Done")
                identity = result["messages"][0]
                self.assertEqual(identity.type, "system")
                self.assertIn(self.rep["name"], identity.content)
                self.assertIn(self.rep["email"], identity.content)
                tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                self.assertEqual([m.name for m in tool_results], [name])
                payload = json.loads(tool_results[0].content)
                if name == "send_prospect_email":
                    self.assertEqual(payload["status"], "sent")
                    self.assertEqual(payload["from"], self.rep["email"])
                else:
                    self.assertEqual(payload["score"], 70)


if __name__ == "__main__":
    unittest.main()
