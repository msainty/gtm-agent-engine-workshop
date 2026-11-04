import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent.gtm_agent import data_service, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def send(self, prospect):
        return send_prospect_email.func(
            prospect=prospect,
            subject="Book a demo",
            body="Please book a demo.",
            runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_mrossi"}}),
        )

    def test_disqualified_prospect_cannot_send_with_forged_eligibility(self):
        with patch("gtm_agent.gtm_agent.uuid.uuid4") as message_id:
            result = self.send({
                "prospect_id": "LEAD-50001",
                "disqualified": False,
                "email": "forged@example.com",
            })
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
        })
        message_id.assert_not_called()

    def test_missing_record_or_id_cannot_send(self):
        for prospect in ({"prospect_id": "MISSING"}, {}):
            with self.subTest(prospect=prospect):
                with patch("gtm_agent.gtm_agent.uuid.uuid4") as message_id:
                    result = self.send({**prospect, "email": "forged@example.com"})
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(result["error"], "Prospect is disqualified; email not sent.")
                message_id.assert_not_called()

    def test_truthy_disqualification_cannot_send(self):
        with patch.object(data_service, "get_prospect_record", return_value={
            "disqualified": "do not contact", "email": "prospect@example.com",
        }), patch("gtm_agent.gtm_agent.uuid.uuid4") as message_id:
            result = self.send({"prospect_id": "LEAD-50001"})
        self.assertEqual(result["status"], "blocked")
        message_id.assert_not_called()

    def test_eligible_prospect_uses_stored_contact_details(self):
        record = data_service.get_prospect_record("LEAD-12853")
        result = self.send({
            "prospect_id": "LEAD-12853",
            "disqualified": True,
            "email": "forged@example.com",
            "name": "Forged name",
        })
        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(result["to"], record["email"])
        self.assertEqual(result["to_name"], record["name"])
        self.assertEqual(result["from"], "marco.rossi@northpoint.com")
        self.assertEqual(result["subject"], "Book a demo")
        self.assertEqual(result["body"], "Please book a demo.")

    def test_missing_stored_email_does_not_use_supplied_address(self):
        with patch.object(data_service, "get_prospect_record", return_value={
            "disqualified": False,
        }), patch("gtm_agent.gtm_agent.uuid.uuid4") as message_id:
            result = self.send({"prospect_id": "LEAD-12853", "email": "forged@example.com"})
        self.assertEqual(result["status"], "failed")
        message_id.assert_not_called()


if __name__ == "__main__":
    unittest.main()
