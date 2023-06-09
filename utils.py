import datetime
import time
from torch import cuda
from typing import List, Dict
import pandas as pd
from pathlib import Path


def get_current_dir() -> Path:
    return Path(__file__).resolve().parent


def format_time(elapsed: time.time) -> str:
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def set_device() -> str:
    dev = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Using device: {dev}')
    return dev


def save_model(output_dir: str, model, tokenizer) -> None:
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_train_data(training_stats: List[Dict[str, float]]) -> None:
    train_stats_df = pd.DataFrame(training_stats)
    train_stats_df.set_index('epoch')
    train_stats_df.to_csv(get_current_dir() / 'data' /
                          'train_data.csv', index='epoch')


def build_dream_data() -> List[str]:
    directory = get_current_dir()
    old_data = pd.read_json(directory / 'data' / 'dreams_reddit_old.json')
    new_data = pd.read_json(directory / 'data' / 'dreams_reddit_new.json')
    old_data.columns = ['body']
    return pd.concat((old_data, new_data)).values.tolist()
