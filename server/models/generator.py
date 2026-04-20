"""Module for generating text based on a prompt and optional context."""

from openai import OpenAI

from server.config import resolve_generator_settings


_SYSTEM_PROMPT = (
      "You are a helpful assistant. Answer the user's question using only the provided context. "                                     
      "If the answer is not in the context, say you don't know."
  )                                                                                                                                   
   
_USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"          

class Generator:
    def __init__(self, model_type: str | None = None):
        self.model_type = model_type
        self.settings = resolve_generator_settings(self.model_type)

        self.client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
        )
        if self.settings.generation_model is not None:
            self.model = self.settings.generation_model
        else:
            listed = self.client.models.list()
            if not listed.data:
                raise RuntimeError("embeddings API returned no models; set embedding_model in config")
            self.model = listed.data[0].id
            # raise RuntimeError("generation_model must be set in config for generation to work")

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        retrieved_context = "\n\n".join(chunk["text"] for chunk in context_chunks)
        prompt = _SYSTEM_PROMPT.format(context=retrieved_context, question=query)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": _USER_TEMPLATE.format(context=retrieved_context, question=query)},],
            max_tokens=512,
            temperature=0,
        )
        return response.choices[0].message.content