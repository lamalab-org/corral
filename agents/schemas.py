from pydantic import BaseModel, Field


class Action(BaseModel):
    action_type: str | None = Field(
        default=None, description="Tool to be used for the action"
    )
    action_input: str | None = Field(
        default=None, description="Input parameters for the action"
    )


class ReActOutput(BaseModel):
    thought: str = Field(description="Agent's reasoning process")
    action: Action | None = None
    final_answer: str | None = None
