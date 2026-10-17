"""Starter policy: one zero-shot call per question, all questions at once."""

MANIFEST = {"name": "starter-zero-shot"}


def _prompt(question):
    return (
        question.text
        + "\n\nSolve the problem. Put your final answer on the last line, "
        "in the form `ANSWER: <answer>`."
    )


class Policy:
    """Sends every question to the student in one batch."""

    def run(self, questions, ctx):
        texts = ctx.student.generate(
            [{"prompt": _prompt(q), "question_id": q.id} for q in questions]
        )
        for question, text in zip(questions, texts, strict=True):
            if text:
                ctx.answer(question.id, text)
            ctx.log(f"one call, {len(text)} chars back", question_id=question.id)
