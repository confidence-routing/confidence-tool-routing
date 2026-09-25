"""
coqa.py
=======

Loader for CoQA — Conversational Question Answering.

Source: Stanford NLP direct download
        https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json

~7.9k questions across 500 conversations in the dev set (the test set has
no public answers). Each conversation is grounded in a passage from one
of 7 domains (children's stories, literature, news, Wikipedia, science,
Reddit, exams).

Task: answer each question given the passage and conversation history.
Tool type: retrieval — the passage IS the retrieval result; the
    interesting test is whether the LLM can answer from parametric
    memory alone vs. needing the passage.

Each question becomes one TaskItem. The passage and conversation history
are stored in meta so the ground-truth pipeline can construct both the
"with tool" (passage provided) and "without tool" (no passage) prompts.
"""

from __future__ import annotations

import json
from typing import List

from .base import DatasetLoader, TaskItem, register

_COQA_DEV_URL = "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json"


@register("coqa")
class CoQALoader(DatasetLoader):

    @property
    def tool_type(self) -> str:
        return "retrieval"

    @property
    def description(self) -> str:
        return "CoQA conversational QA (~7.9k questions, 500 passages, 7 domains)"

    def download(self) -> None:
        self._download_file(_COQA_DEV_URL, "coqa-dev-v1.0.json")

    def parse(self) -> List[TaskItem]:
        data = self._read_json(self.data_dir / "coqa-dev-v1.0.json")

        items: List[TaskItem] = []
        for story in data["data"]:
            story_id = story["id"]
            passage = story["story"]
            source = story.get("source", "unknown")
            questions = story["questions"]
            answers = story["answers"]

            # Build conversation history incrementally
            history: list = []
            for q, a in zip(questions, answers):
                q_text = q["input_text"]
                a_text = a["input_text"]
                turn_id = q["turn_id"]
                evidence_span = a.get("span_text", "")

                items.append(TaskItem(
                    task_id=f"coqa_{story_id}_t{turn_id}",
                    query=q_text,
                    gold_answer=a_text,
                    dataset="coqa",
                    tool_type="retrieval",
                    meta={
                        "story_id": story_id,
                        "turn_id": turn_id,
                        "passage": passage,
                        "source_domain": source,
                        "evidence_span": evidence_span,
                        "conversation_history": list(history),
                    },
                ))

                history.append({"question": q_text, "answer": a_text})

        return items
