import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from topo_nader.agents import TopoNaderAgentOrchestrator
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
