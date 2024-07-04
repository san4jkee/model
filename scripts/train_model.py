from comet_ml import Experiment
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import Dataset, DatasetDict
import torch

def train_model(tokenized_datasets, tokenizer):
    # Инициализация эксперимента Comet.ml
    experiment = Experiment(
        api_key="92fMiQ8vvc17lKxU4QFtYQOHv",
        project_name="general",
        workspace="san4jkee"
    )

    # Загрузка модели на GPU, если доступен
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("ai-forever/ruGPT-3.5-13B").to(device)

    training_args = TrainingArguments(
        output_dir="./models/trained_model",
        overwrite_output_dir=True,  # перезапись содержимого выходного каталога
        num_train_epochs=200,  # количество эпох обучения
        per_device_train_batch_size=32,  # размер пакета для обучения
        per_device_eval_batch_size=32,  # размер пакета для оценки
        warmup_steps=10,  # количество шагов прогрева для планировщика скорости обучения
        gradient_accumulation_steps=16,  # для увеличения «виртуального» размера пакета
        evaluation_strategy="epoch",  # оценивать модель каждую эпоху
        logging_strategy="epoch",  # логировать каждую эпоху
        logging_dir="./logs",  # каталог для логов
        save_strategy="epoch",  # сохранять модель каждую эпоху
        save_total_limit=3,  # ограничение на общее количество сохранений
        seed=42,  # случайное число для воспроизводимости
        disable_tqdm=False,  # включение tqdm для отображения прогресса
    )

    # Создание DataCollator для MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Создание Trainer для обучения модели
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],  # правильный датасет для обучения
        eval_dataset=tokenized_datasets["validation"],  # правильный датасет для валидации
        data_collator=data_collator,  # передаем data_collator для обработки данных
    )

    # Начало обучения
    trainer.train()

if __name__ == "__main__":
    # Загрузка токенизированного датасета
    tokens = torch.load('data/processed/tokenized_data.pt')

    # Проверка структуры загруженных данных
    if not tokens or "input_ids" not in tokens or "attention_mask" not in tokens:
        raise ValueError("Tokenized data is not in the expected format or is empty.")

    print(f"Loaded tokens keys: {tokens.keys()}")
    print(f"Number of samples: {len(tokens['input_ids'])}")

    # Разделение данных на train и validation
    total_size = len(tokens['input_ids'])
    if total_size == 0:
        raise ValueError("Tokenized data is empty.")
    elif total_size < 2:
        raise ValueError("Not enough data for train/validation split. Please add more data.")

    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError("Train or validation dataset is empty. Please check the tokenized data.")

    train_dataset = Dataset.from_dict({
        "input_ids": tokens['input_ids'][:train_size],
        "attention_mask": tokens['attention_mask'][:train_size],
        "labels": tokens['input_ids'][:train_size]
    })

    val_dataset = Dataset.from_dict({
        "input_ids": tokens['input_ids'][train_size:],
        "attention_mask": tokens['attention_mask'][train_size:],
        "labels": tokens['input_ids'][train_size:]
    })

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruGPT-3.5-13B")

    # Обучение модели с логированием в Comet.ml
    train_model(dataset, tokenizer)
