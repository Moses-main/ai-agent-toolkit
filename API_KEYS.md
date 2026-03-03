# 🔑 API Keys Guide

This guide explains how to get API keys for the different LLM providers supported by this framework.

---

## OpenAI (GPT-4, GPT-3.5)

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to **API Keys** (left sidebar)
4. Click **Create new secret key**
5. Copy and save your key (it won't be shown again!)

**Environment Variable:**
```bash
export OPENAI_API_KEY="sk-..."
```

**In Code:**
```python
from ai_agent.clients import OpenAIClient

client = OpenAIClient(api_key="sk-...")
```

---

## Anthropic (Claude)

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in
3. Click **API Keys** in the sidebar
4. Click **Create Key**
5. Copy and save your key

**Environment Variable:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**In Code:**
```python
from ai_agent.clients import AnthropicClient

client = AnthropicClient(api_key="sk-ant-...")
```

---

## Ollama (Local Models)

Ollama runs locally - no API key needed!

1. Install Ollama: [ollama.ai](https://ollama.ai)
2. Run: `ollama serve`
3. Pull a model: `ollama pull llama2`

**Environment Variable:** None needed

**In Code:**
```python
from ai_agent.clients import OllamaClient

client = OllamaClient(model="llama2")  # or "mistral", "codellama", etc.
```

**Default URL:** `http://localhost:11434`

---

## Azure OpenAI

1. Go to [Azure Portal](https://portal.azure.com)
2. Create an **Azure OpenAI** resource
3. Go to **Keys and Endpoint** in the sidebar
4. Copy your key and endpoint URL

**Environment Variable:**
```bash
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

**In Code:**
```python
from ai_agent.clients import AzureOpenAIClient

client = AzureOpenAIClient(
    api_key="your-key",
    api_version="2024-02-01",
    endpoint="https://your-resource.openai.azure.com/",
    deployment="gpt-4"
)
```

---

## 🔐 Security Best Practices

### 1. Never Commit Keys

Add to `.gitignore`:
```
.env
*.env
.env.local
```

### 2. Use Environment Variables

```python
import os

client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY")
)
```

### 3. Use .env Files

Create `.env`:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Load with python-dotenv:
```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file
```

### 4. For Production

- Use secret management services (AWS Secrets Manager, HashiCorp Vault)
- Never store keys in code
- Rotate keys regularly
- Use minimal required scopes

---

## 💰 Usage & Billing

| Provider | Free Tier | Paid |
|----------|-----------|------|
| OpenAI | $5 credit for new accounts | Pay-as-you-go |
| Anthropic | $5 credit for new accounts | Pay-as-you-go |
| Ollama | Completely free | Free |
| Azure | $200 credit for new accounts | Pay-as-you-go |

---

## 🆘 Troubleshooting

### "Invalid API key"
- Check key is correct
- Ensure no extra spaces
- Verify key hasn't expired

### "Rate limit exceeded"
- Wait and retry
- Upgrade to paid tier
- Implement retry logic (built-in)

### "Insufficient credits"
- Add payment method in provider dashboard
- Check billing section

---

## 📞 Support

- OpenAI: [platform.openai.com/support](https://platform.openai.com/support)
- Anthropic: [console.anthropic.com/support](https://console.anthropic.com/support)
- Ollama: [github.com/ollama/ollama/discussions](https://github.com/ollama/ollama/discussions)
