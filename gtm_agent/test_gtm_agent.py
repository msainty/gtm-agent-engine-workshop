import json
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent.gtm_agent import run_agent


class AgentTests(unittest.TestCase):
    def test_application_tools_and_signed_in_sender(self):
        requests = []
        replies = iter([
            AIMessage(content="", tool_calls=[{
                "name": "get_current_rep", "args": {}, "id": "rep",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email",
                "args": {
                    "prospect": {"name": "Test Prospect", "email": "prospect@example.com"},
                    "subject": "Demo", "body": "Can we schedule a demo?",
                },
                "id": "email",
            }]),
            AIMessage(content="Email sent."),
        ])

        def generate(model, messages, **kwargs):
            requests.append(model._get_request_payload(messages, **kwargs))
            return ChatResult(generations=[ChatGeneration(message=next(replies))])

        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            with patch.object(ChatOpenAI, "_generate", generate):
                result = run_agent(
                    "Email the prospect to schedule a demo.",
                    user_id="rep_jchen", environment="test", thread_id="test-thread",
                )

        self.assertEqual(result["reply"], "Email sent.")
        results = {
            message.name: json.loads(message.content)
            for message in result["messages"] if message.type == "tool"
        }
        self.assertTrue(results["get_current_rep"]["found"])
        self.assertEqual(results["get_current_rep"]["rep"]["rep_id"], "rep_jchen")
        email = results["send_prospect_email"]
        self.assertEqual(email["status"], "sent")
        self.assertEqual(email["from"], results["get_current_rep"]["rep"]["email"])
        self.assertEqual(email["to"], "prospect@example.com")
        self.assertEqual({tool["function"]["name"] for tool in requests[0]["tools"]}, {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info", "get_current_rep",
        })


if __name__ == "__main__":
    unittest.main()
