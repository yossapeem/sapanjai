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
                        "Your task is to rewrite the following message so it can be safely sent "
                        "to someone who may be emotionally sensitive. "
                        "Keep the meaning intact, but make the wording gentle, friendly, and supportive. "
                        "The response must be natural, positive in tone, and concise (one clear sentence only)."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
        )
        return completion.choices[0].message.content
