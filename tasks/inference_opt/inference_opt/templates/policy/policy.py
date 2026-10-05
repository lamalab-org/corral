"""Starter policy: one zero-shot call per question, all questions at once."""

MANIFEST = {"name": "starter-zero-shot"}


def _prompt(question):
    prompt = question.text
    if question.choices:
        prompt += "\n\n" + question.rendered_choices()
        if question.benchmark == "chembench":
            prompt += (
                "\n\nOne or more options may be correct. "
                "Answer with the letters of all of them, separated by commas."
            )
        else:
            prompt += "\n\nAnswer with the letter of the correct option."
    marker = (
        "`[ANSWER]<answer>[/ANSWER]`"
        if question.benchmark == "chembench"
        else "`ANSWER: <answer>`"
    )
    return prompt + f"\n\nExplain briefly, then finish with {marker}."


class Policy:
    """Sends every question to the student in one batch."""

    def run(self, questions, ctx):
        # A cheap guess first: if the budget or time runs out later, it still scores.
        ctx.answer(
            {q.id: f"ANSWER: {(q.choice_labels or ('A',))[0]}" for q in questions}
        )

        texts = ctx.student.generate(
            [{"prompt": _prompt(q), "question_id": q.id} for q in questions]
        )
        for question, text in zip(questions, texts, strict=True):
            if text:
                ctx.answer(question.id, text)
            ctx.log(f"one call, {len(text)} chars back", question_id=question.id)
