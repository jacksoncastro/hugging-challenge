from huggingface_hub import ModelCard, ModelCardData

import constants

def get_card() -> ModelCard:

    card_data = ModelCardData(
        language='pt',
        license='mit',
        library_name='ceares',
        base_model=constants.MODEL_BASE,
        tags=['português', 'ceares', 'cearense', 'brasil', 'brazil', 'pt', 'BR', 'sotaque']
    )

    model_description = f"""
---
{ card_data.to_yaml() }
---
# GPT-2 Cearês Translator

Este modelo é uma versão fine-tuned do GPT-2 em português (`{constants.MODEL_BASE}`) para o sotaque cearense.

## Detalhes do Modelo

- **Arquitetura:** GPT-2
- **Linguagem:** Português do Brasil
- **Fine-tuned para:** Tradução para sotaque cearense
- **Dataset:** Dataset customizado com palavras e expressões cearenses

## Como Usar

### Com a pipeline do Transformers

```python
from transformers import pipeline

translator = pipeline(
    "text-generation",
    model="{constants.MODEL_ID_HUGGINGFACE}",
    tokenizer="{constants.MODEL_ID_HUGGINGFACE}"
)

## Traduzir texto
input_text = "[PT] Esse rapaz é bobo [CE]"
result = translator(input_text, max_length=50, temperature=0.3)
print(result[0]['generated_text'])
```

## Links

- [Github](https://github.com/jacksoncastro/hugging-challenge)
"""

    return ModelCard(model_description)
