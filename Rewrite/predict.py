import os
from dotenv import load_dotenv
from cog import BasePredictor, Input
from openai import OpenAI

load_dotenv()  # Load environment variables from .env

class Predictor(BasePredictor):
    def setup(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
