import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import re
import constants

@st.cache_resource
def load_model():
    """Carrega o modelo e tokenizer fine-tuned"""
    tokenizer = GPT2Tokenizer.from_pretrained(constants.MODEL_ID_HUGGINGFACE)
    model = GPT2LMHeadModel.from_pretrained(constants.MODEL_ID_HUGGINGFACE)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

@st.cache_resource
def load_ceares_dictionary():
    """Carrega o dicionário cearense do CSV"""
    df = pd.read_csv(constants.DATASET_PATH, header=None, names=['ceares', 'portugues'])
    
    ceares_dict = {}
    for _, row in df.iterrows():
        portugues_terms = [term.strip().lower() for term in row['portugues'].split(';')]
        for term in portugues_terms:
            ceares_dict[term] = row['ceares'].lower()
    
    return ceares_dict

def translate_to_ceares(text, tokenizer, model, ceares_dict):
    """Traduz texto para o sotaque cearense com controles rigorosos"""
    input_text = f"[PT] {text} [CE]"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # encontrar a posição do token [CE]
    ce_token_id = tokenizer.encode('[CE]')[0]
    ce_position = (input_ids[0] == ce_token_id).nonzero(as_tuple=True)[0]
    
    start_idx = ce_position[0] + 1 if len(ce_position) > 0 else len(input_ids[0])
    
    # gerar resposta com parâmetros mais restritivos
    output = model.generate(
        input_ids,
        max_length=start_idx + 20,  # Limitar bastante o comprimento
        num_return_sequences=1,
        temperature=0.3,  # Temperatura baixa para menos aleatoriedade
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=3.0,  # Alta penalidade de repetição
        no_repeat_ngram_size=3,   # Evitar repetição de trigramas
        early_stopping=True,
    )
    
    # extrair a tradução
    generated_tokens = output[0][start_idx:]
    translated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Limpeza e validação rigorosa
    return validate_and_clean_translation(translated, text, ceares_dict)

def validate_and_clean_translation(translated, original_text, ceares_dict):
    """Valida e limpa a tradução com critérios rigorosos"""
    # manter apenas até a primeira pontuação final
    for punctuation in ['.', '!', '?']:
        if punctuation in translated:
            translated = translated.split(punctuation)[0] + punctuation
            break
    
    # verificar se a tradução contém termos cearenses
    has_ceares_terms = any(term in translated.lower() for term in ceares_dict.values())
    
    # verificar se a estrutura da frase mantém a mesma forma básica
    original_words = original_text.lower().split()
    translated_words = translated.lower().split()
    
    # se não tem termos cearenses ou a estrutura é muito diferente, usar substituição direta
    if not has_ceares_terms or not is_similar_structure(original_words, translated_words, ceares_dict):
        return direct_translation(original_text, ceares_dict)
    
    return translated.strip()

def is_similar_structure(original_words, translated_words, ceares_dict):
    """Verifica se a estrutura da tradução é similar à original"""
    # contar palavras comuns (excluindo termos cearenses)
    common_words = 0
    for orig_word in original_words:
        clean_orig = re.sub(r'[^\w\s]', '', orig_word)
        if clean_orig not in ceares_dict:  # Não é uma palavra a ser traduzida
            for trans_word in translated_words:
                clean_trans = re.sub(r'[^\w\s]', '', trans_word)
                if clean_orig == clean_trans:
                    common_words += 1
                    break
    
    # pelo menos 50% das palavras não-cearenses devem estar presentes
    non_ceares_count = sum(1 for word in original_words if re.sub(r'[^\w\s]', '', word) not in ceares_dict)
    return common_words >= max(1, non_ceares_count * 0.5)

def direct_translation(text, ceares_dict):
    """Substituição direta baseada no dicionário, mantendo a estrutura da frase"""
    words = text.split()
    translated_words = []
    
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        punctuation = word[-1] if word and not word[-1].isalnum() else ''
        
        if clean_word in ceares_dict:
            # manter a capitalização original se a palavra estava em maiúscula
            if word.istitle():
                translated_word = ceares_dict[clean_word].title()
            elif word.isupper():
                translated_word = ceares_dict[clean_word].upper()
            else:
                translated_word = ceares_dict[clean_word]
            
            translated_words.append(translated_word + punctuation)
        else:
            translated_words.append(word)
    
    return ' '.join(translated_words)

# configuração da interface
st.set_page_config(
    page_title='Tradutor para Cearês',
    page_icon='🇧🇷',
    layout='centered'
)

# Carregar recursos
tokenizer, model = load_model()
ceares_dict = load_ceares_dictionary()

# Interface do usuário
st.title('🇧🇷 Tradutor para Cearês')
st.write('Transforme frases em português para o sotaque cearense!')

text_input = st.text_area(
    'Digite a frase em português:',
    height=100,
    placeholder='Ex: Esse rapaz é bobo'
)

if st.button('Traduzir para Cearês', type='primary'):
    if text_input.strip():
        with st.spinner('Traduzindo...'):
            translated_text = translate_to_ceares(text_input, tokenizer, model, ceares_dict)
            
            st.success('Tradução concluída!')
            st.text_area('Em Cearês:', value=translated_text, height=100)
            
            # Mostrar também a substituição direta para comparação
            direct_text = direct_translation(text_input, ceares_dict)
            if translated_text != direct_text:
                with st.expander('Ver substituição direta'):
                    st.write(direct_text)
    else:
        st.warning('Por favor, digite uma frase para traduzir.')

# Informações sobre o projeto
with st.expander('Sobre este projeto'):
    st.write("""
    Tradutor que utiliza IA para converter frases em português padrão
    para o sotaque cearense, usando um modelo GPT-2 fine-tuned.
    
    Exemplos:
    - "bobo" → "abestado"
    - "besteira" → "jaula" 
    - "maluco" → "abirolado"
    - "mexer" → "futricar"
    """)