"""Starter policy: one zero-shot call per question."""

MANIFEST = {"name": "starter-zero-shot"}


class Policy:
    """Answers each question with one direct call to the student."""

    def run(self, questions, ctx):
        for question in questions:
            # A cheap guess first: if the budget runs out later, it still scores.
            ctx.submit(question.id, f"ANSWER: {(question.choice_labels or ('A',))[0]}")

        for question in questions:
            prompt = question.text
            if question.choices:
                prompt += "\n\n" + question.rendered_choices()
                prompt += "\n\nAnswer with the letter of the correct option."
            marker = (
                "`[ANSWER]<answer>[/ANSWER]`"
                if question.benchmark == "chembench"
                else "`ANSWER: <answer>`"
            )
            prompt += f"\n\nExplain briefly, then finish with {marker}."

            answer = ctx.student.generate(prompt, question_id=question.id)
            ctx.submit(question.id, answer)
            ctx.log(f"one call, {len(answer)} chars back", question_id=question.id)
