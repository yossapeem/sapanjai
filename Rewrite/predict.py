import os
from cog import BasePredictor, Input
from openai import OpenAI

class Predictor(BasePredictor):
    def setup(self):
        self.client = OpenAI(api_key="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")

    def predict(
        self,
        user_message: str = Input(description="The message to make more supportive"),
    ) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Imagine the message is sent to a mentally sensitive individual."
                        "Rewrite the message to be as friendly and safe as possible while keeping the original meaning and context."
                        "Keep it short (no more than one full sentence)"
                        "Make it sound natural."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
        )
        return completion.choices[0].message.content
