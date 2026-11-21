"""奖励设计使用的 Responses API。"""

import json

from openai import OpenAI


def design(config, prompt, context, images, *, schema, name):
    with OpenAI(max_retries=0) as client:
        response = client.responses.create(
            model=config["model"],
            max_output_tokens=config["max_output_tokens"],
            input=[dict(role="system", content=prompt), dict(role="user", content=[
                dict(type="input_text", text=json.dumps(context, ensure_ascii=False)), *images])],
            text={"format": {
                "type": "json_schema",
                "name": name,
                "strict": True,
                "schema": schema,
            }},
        )
    return json.loads(response.output_text), response.id
