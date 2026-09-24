"""Starter policy: a single zero-shot call."""

MANIFEST = {
    "name": "starter-zero-shot",
    "max_calls_per_question": 1,
}


class Policy:
    """Answers each question with one direct call to the student."""

    def solve(self, question, ctx):
        # If anything below fails, this is what gets scored. Always set it: a
        # cheap guess beats no answer when the budget runs out mid-run.
        ctx.scratch["fallback"] = (question.choice_labels or ("A",))[0]

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

        answer = ctx.student.generate(prompt, temperature=0.0)
        ctx.log(f"one call, {len(answer)} chars back")
        return answer
