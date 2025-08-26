from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json
import constants

def train():

    # carregar tokenizer e modelo
    tokenizer = GPT2Tokenizer.from_pretrained(constants.MODEL_BASE)
    model = GPT2LMHeadModel.from_pretrained(constants.MODEL_BASE)

    # adicionar tokens especiais
    special_tokens = {'additional_special_tokens': ['[PT]', '[CE]']}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # configurar o tokenizer para padding
    tokenizer.pad_token = tokenizer.eos_token

    # carregar dados de treinamento
    with open(constants.TRAIN_DATA_PATH, 'r', encoding='utf-8') as file:
        train_data = json.load(file)

    # preparar dados no formato correto
    def prepare_text_data(data):
        texts = []
        for item in data:
            texts.append(f"[PT] {item['original']} [CE] {item['ceares']}")
        return texts

    train_texts = prepare_text_data(train_data)

    # criar dataset
    dataset = Dataset.from_dict({'text': train_texts})

    # tokenizar o dataset
    def tokenize_function(examples):
        # tokenizar o texto
        tokenized = tokenizer(
            examples['text'], 
            truncation=True, 
            padding=False,  # Não fazer padding ainda
            max_length=128
        )
        
        # para modelos de linguagem, os labels são os próprios input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # configurar argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=constants.OUTPUT_MODEL_TRAIN,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=1,
        eval_strategy='no', # Não temos dados de avaliação
        prediction_loss_only=True,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=False, # Desativar se não tiver GPU compatível
    )

    # criar trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset
    )

    # iniciar treinamento
    print('Iniciando treinamento...')
    trainer.train()

    # salvar modelo e tokenizer
    model.save_pretrained(constants.OUTPUT_MODEL_TRAIN)
    tokenizer.save_pretrained(constants.OUTPUT_MODEL_TRAIN)
    print('Modelo fine-tuned salvo com sucesso!')
