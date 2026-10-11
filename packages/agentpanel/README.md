# agentpanel

A small, domain-agnostic core for multi-agent LLM workflows on
[LangGraph](https://github.com/langchain-ai/langgraph):

- **Specialists** produce artifacts in parallel.
- **Reviewers** judge each other's artifacts; rejections send the author back to revise,
  with the feedback, for a bounded number of rounds.
- A **coordinator** assembles the result, including anything still unresolved.

It runs on local models by default ([Ollama](https://ollama.com)), with bring-your-own-key
OpenAI and Anthropic support, and gets reliable structured output from small models.

```bash
pip install agentpanel            # Ollama only
pip install "agentpanel[cloud]"   # plus OpenAI and Anthropic
```

## Example

```python
import asyncio
from agentpanel import AgentContext, LLMReviewer, LLMSpecialist, Workflow
from agentpanel.config import LLMSettings
from agentpanel.llm import create_chat_model

model = create_chat_model(LLMSettings())  # qwen3:8b on local Ollama by default


class Join:
    async def assemble(self, ctx: AgentContext) -> str:
        return ctx.artifacts["writer"].content


workflow = Workflow(
    specialists=[
        LLMSpecialist(
            name="writer",
            kind="product description",
            model=model,
            instructions="You write concise product descriptions.",
            render=lambda ctx: f"Describe {ctx.inputs['product']} in two sentences.",
        )
    ],
    reviewers=[LLMReviewer("editor", model, "You edit marketing copy for accuracy.")],
    coordinator=Join(),
    max_revisions=1,
)
result = asyncio.run(workflow.run({"product": "a solar-powered kettle"}))
print(result.result, result.unresolved)
```

## Pieces

| Module | What it gives you |
|---|---|
| `agentpanel.graph` | `Workflow`: parallel specialists, cross-review, bounded revision, progress callbacks |
| `agentpanel.agents` | `Specialist`, `Reviewer`, `Coordinator` protocols: bring any implementation |
| `agentpanel.llm_agents` | `LLMSpecialist`, `LLMReviewer`: prompt-driven agents with revision handling |
| `agentpanel.structured` | `ainvoke_structured`: pydantic-validated output with error feedback and domain `check`s; constrained decoding on Ollama |
| `agentpanel.llm` | `create_chat_model`, `check_provider`: provider factory and health checks |
| `agentpanel.config` | `LLMSettings`: env/.env settings, per-agent model and reasoning overrides |
| `agentpanel.testing` | `ScriptedChatModel`: a deterministic fake chat model for tests |

## Configuration

`LLMSettings` reads `AGENTPANEL_*` environment variables (subclass it to use your own
prefix): `PROVIDER` (`ollama`, `openai`, `anthropic`), `MODEL`, `BASE_URL`, `TIMEOUT`,
`REASONING`, `CONTEXT_WINDOW`, `AGENT_MODELS`, `AGENT_REASONING`. Cloud keys use the
standard `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.

Built for and used by [API Shakedown](https://github.com/anzar-ahsan-commits/api-shakedown), where the
source lives. Issues and contributions: see that repository.

## License

MIT
