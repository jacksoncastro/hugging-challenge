from huggingface_hub import HfApi, create_repo
from model_card import get_card
from dotenv import load_dotenv

import constants
import os

load_dotenv()

ACCESS_TOKEN_HUGGINGFACE = os.getenv('ACCESS_TOKEN_HUGGINGFACE')

# criar um repositório novo (se não existir)
create_repo(constants.MODEL_ID_HUGGINGFACE, token=ACCESS_TOKEN_HUGGINGFACE, exist_ok=True)

api = HfApi()

# fazer upload de toda a pasta do modelo
api.upload_folder(
    folder_path=constants.OUTPUT_MODEL_TRAIN,
    repo_id=constants.MODEL_ID_HUGGINGFACE,
    path_in_repo='',
    token=ACCESS_TOKEN_HUGGINGFACE
)

# publicar card
card = get_card()
card.push_to_hub(constants.MODEL_ID_HUGGINGFACE, token=ACCESS_TOKEN_HUGGINGFACE)
