import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from topo_nader.agents import TopoNaderAgentOrchestrator
from topo_nader.agents.base import TopoNaderStructuredOutputError, parse_json_object
from topo_nader.agents.client import TopoNaderOpenAICompatibleClient
from topo_nader.architecture import TopoNaderArchitectureSpec, TopoNaderArchitectureValidator
from topo_nader.config import TopoNaderAgentConfig, TopoNaderArchitectureConfig


class StubTopoNaderClient:
    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self.calls == 1:
            content = json.dumps({"diagnosis": "overfit", "evidence": [], "risks": [], "recommended_focus": "regularize"})
        elif self.calls == 2:
            content = json.dumps({"proposals": [{"title": "dropout", "rationale": "test", "changes": {"dropout": 0.2}}]})
        else:
            payload = TopoNaderArchitectureSpec(name="TopoNaderChild", dropout=0.2).to_dict()
            content = json.dumps(payload)
        return {"content": content, "model": "stub", "prompt_tokens": 0, "completion_tokens": 0}


class TopoNaderAgentTests(unittest.TestCase):
    def test_parser_recovers_redundant_trailing_closing_brace(self):
        payload = parse_json_object('{"proposals":[{"title":"test"}]}}')
        self.assertEqual(payload["proposals"][0]["title"], "test")

    def test_invalid_structured_output_is_fed_back_and_retried(self):
        class FormatRetryClient:
            def __init__(self):
                self.messages = []

            def complete(self, messages):
                self.messages.append([dict(message) for message in messages])
                if len(self.messages) == 1:
                    content = '{"diagnosis":"broken" "evidence":[]}'
                else:
                    content = json.dumps({
                        "diagnosis": "recovered",
                        "evidence": [],
                        "risks": [],
                        "recommended_focus": "continue",
                    })
                return {"content": content, "model": "stub", "prompt_tokens": 1, "completion_tokens": 2}

        client = FormatRetryClient()
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = TopoNaderAgentOrchestrator(client, Path(directory), structured_output_retries=1)
            payload, record = orchestrator.reflector.diagnose({}, {}, [])
        self.assertEqual(payload["diagnosis"], "recovered")
        self.assertEqual(record["structured_output_attempts"], 2)
        self.assertEqual(record["prompt_tokens"], 2)
        self.assertEqual(client.messages[1][-2]["role"], "assistant")
        self.assertIn("Parser error", client.messages[1][-1]["content"])

    def test_invalid_structured_output_raises_after_retry_limit(self):
        class InvalidClient:
            def complete(self, messages):
                return {"content": '{"missing": "comma" "here": true}', "model": "stub"}

        with tempfile.TemporaryDirectory() as directory:
            orchestrator = TopoNaderAgentOrchestrator(
                InvalidClient(), Path(directory), structured_output_retries=1
            )
            with self.assertRaisesRegex(TopoNaderStructuredOutputError, "after 2 attempts"):
                orchestrator.reflector.diagnose({}, {}, [])

    def test_client_uses_openai_base_url_as_compatible_endpoint_fallback(self):
        environment = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://compatible.example/v1/",
        }
        with patch.dict(os.environ, environment, clear=True):
            client = TopoNaderOpenAICompatibleClient(
                TopoNaderAgentConfig(model="gpt-test", base_url="https://api.openai.com/v1")
            )
        self.assertEqual(client.base_url, "https://compatible.example/v1")
        self.assertEqual(client.model, "gpt-test")

    def test_reader_free_agent_chain_generates_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = TopoNaderAgentOrchestrator(StubTopoNaderClient(), Path(directory))
            candidates, trace = orchestrator.generate_candidates(TopoNaderArchitectureSpec(), {}, [], 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "TopoNaderChild")
        self.assertIn("diagnosis", trace)

    def test_invalid_modifier_output_is_fed_back_and_retried(self):
        class RetryClient:
            def __init__(self):
                self.calls = 0

            def complete(self, messages):
                self.calls += 1
                if self.calls == 1:
                    content = json.dumps({"diagnosis": "test", "evidence": [], "risks": [], "recommended_focus": "test"})
                elif self.calls == 2:
                    content = json.dumps({"proposals": [{"title": "candidate", "rationale": "test", "changes": {}}]})
                elif self.calls == 3:
                    content = json.dumps(TopoNaderArchitectureSpec(name="Invalid", input_dim=15).to_dict())
                else:
                    content = json.dumps(TopoNaderArchitectureSpec(name="Recovered", dropout=0.2).to_dict())
                return {"content": content, "model": "stub", "prompt_tokens": 0, "completion_tokens": 0}

        validator = TopoNaderArchitectureValidator(TopoNaderArchitectureConfig())
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = TopoNaderAgentOrchestrator(RetryClient(), Path(directory))
            candidates, trace = orchestrator.generate_candidates(
                TopoNaderArchitectureSpec(), {}, [], 1, validator=validator.validate, retry_limit=2
            )
        self.assertEqual([candidate.name for candidate in candidates], ["Recovered"])
        self.assertFalse(trace["candidate_attempts"][0]["accepted"])
        self.assertTrue(trace["candidate_attempts"][1]["accepted"])


if __name__ == "__main__":
    unittest.main()
