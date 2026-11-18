"""仅供离线 reward_designer 调用的官方 Responses API。"""

import json

from openai import OpenAI

from .reward_schema import design_schema


def design(config, prompt, context, images):
    with OpenAI(max_retries=0) as client:
        response = client.responses.create(
            model=config["model"],
            max_output_tokens=config["max_output_tokens"],
            input=[dict(role="system", content=prompt), dict(role="user", content=[
                dict(type="input_text", text=json.dumps(context, ensure_ascii=False)), *images])],
            text={"format": {
                "type": "json_schema",
                "name": "reward_design",
                "strict": True,
                "schema": design_schema(context["available_terms"]),
            }},
        )
    return json.loads(response.output_text), response.id
