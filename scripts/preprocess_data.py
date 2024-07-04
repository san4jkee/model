from comet_ml import Experiment
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer
from transformers import LineByLineTextDataset
import torch

def create_train_dataset(file_path, tokenizer):
    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=64
    )

def train_model(train_dataset, tokenizer):
    # Инициализация эксперимента Comet.ml
    experiment = Experiment(
        api_key="92fMiQ8vvc17lKxU4QFtYQOHv",
        project_name="general",
        workspace="san4jkee"
    )

    # Загрузка модели на GPU, если доступен
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("ai-forever/ruGPT-3.5-13B").to(device)

    print(f"Model loaded on: {device}")

    training_args = TrainingArguments(
        output_dir="./models/trained_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=10,
        gradient_accumulation_steps=8,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="./logs",
        save_strategy="epoch",
        save_total_limit=3,
        seed=42,
        disable_tqdm=False,
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
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Начало обучения
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    file_path = 'data/processed/combined_text.txt'

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruGPT-3.5-13B")

    # Создание датасета
    train_dataset = create_train_dataset(file_path, tokenizer)

    # Обучение модели с логированием в Comet.ml
    train_model(train_dataset, tokenizer)
