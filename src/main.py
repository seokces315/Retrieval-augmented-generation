# GPU-specific settings
import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

# wandb-related settings
import wandb
# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="RAG-finetuning-WQ"
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

import torch
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from data import load_WQ_TV, load_WQ_test, load_WQ_train
from utils import get_args, wandb_hp_space


# Define custom-triainer for batch training
class BatchTrainer(Trainer) :
  # Override compute_loss method
  def compute_loss(self, model, inputs, return_outputs=False) :
    outputs = model(**inputs)
    # loss vectors of batch
    batch_loss = outputs.loss
    # scalarize the vectors
    mean_loss = batch_loss.mean()
    return (mean_loss, outputs) if return_outputs else mean_loss

# Define model_init
def model_init(trial) :
  
  # Pre-trained Retriever 생성
  retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed"
  )
  
  # Init Model 객체 반환
  return RagTokenForGeneration.from_pretrained(
    "facebook/rag-token-nq", retriever=retriever
  )
  
# Main function for WQ Task
def webQuest() :
  
  # 패키지 내 다른 모듈을 import해 데이터셋 반환받기
  train_path = "../datasets/WQ_datasets/trainmodel.json"
  val_path = "../datasets/WQ_datasets/val.json"
  test_path = "../datasets/WQ_datasets/devtest.json"
  
  train_dataset, val_dataset = load_WQ_TV(train_path, val_path, verbose=True)
  train_dicts, train_labels = load_WQ_train(train_path, verbose=True)
  test_dicts, test_labels = load_WQ_test(test_path, verbose=True)

  # pre-trained Tokenizer 생성
  tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
  
  # Shuffling Train dataset
  train_dataset = train_dataset.shuffle(seed=42)
  
  """
  # TrainingArguments 정의
  HP_search_args = TrainingArguments(
    "HP search",
    label_names=["labels"],
    evaluation_strategy="steps",
    eval_steps=100,
    report_to="wandb"
  )
  
  # Trainer 정의
  HP_searcher = BatchTrainer(
    args=HP_search_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    model_init=model_init
  )
  
  # Search best trial (i.e. hyper-parameter tuning)
  best_trial = HP_searcher.hyperparameter_search(
    direction="minimize",
    backend="wandb",
    hp_space=wandb_hp_space,
    n_trials=10
  )
  """
  
  # 2가지 종류의 pre-trained Retriever 생성
  dummy_retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq", use_dummy_dataset=True, index_name="exact"
  )
  retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed"
  )
  
  # 위에서 생성된 객체들을 활용해 RAG 모델 생성
  model = RagTokenForGeneration.from_pretrained(
    "facebook/rag-token-nq", retriever=retriever
  )
  
  """
  # 최적의 탐색 파라미터 불러오기
  best_HPs = best_trial.params
  model.config.update(best_HPs)
  """
  
  # 생성한 모델을 GPU로 전달
  model.to(device)
  
  # """
  # Input arguments 전달받기
  args = get_args()
  
  # TrainingArguments 정의
  """
  # TrainingArguments 정의
  training_args = TrainingArguments(
    evaluation_strategy="steps",
    label_names=["labels"],
    output_dir="./output",
    logging_dir="./logs",
    report_to="wandb",
    logging_steps=100,
    num_train_epochs=10,
    dataloader_drop_last=False,
    load_best_model_at_end=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8
  )
  """
  
  training_args = TrainingArguments(
    evaluation_strategy="epoch",
    label_names=["labels"],
    output_dir="./output",
    logging_dir="./logs",
    report_to="wandb",
    logging_steps=100,
    num_train_epochs=10,
    dataloader_drop_last=False,
    # load_best_model_at_end=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    max_grad_norm=args.max_grad_norm,
    warmup_ratio=args.warmup_ratio
  )
  # """
  
  # Define early stopping callback
  early_stopping = EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=0.01
  )
  
  # Trainer 정의
  trainer = BatchTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # callbacks=[early_stopping]
  )
  
  # Training
  trainer.train()
  
  # Wandb finished
  wandb.finish()
  
  # 평가 모드로 전환
  model.eval()
  
  # Test dicts의 prompt 별로 text generation 수행
  generated_strings = []
  
  with torch.no_grad() :
    # for input_dict in test_dicts :
    for input_dict in train_dicts :
      # Test input을 GPU로 전달
      input_dict = input_dict.to(device)
      # Model의 generate 메서드 호출
      generated_string = model.generate(input_ids=input_dict['input_ids'])
      # Decoding
      decoded_string = tokenizer.decode(generated_string[0], skip_special_token=True)
      print(decoded_string)
      generated_strings.append(decoded_string)
      # 현재 input을 다시 CPU로 전달
      input_dict.to('cpu')
      
  # Metric 평가 - EM Scores
  # total_cnt = len(test_labels)
  total_cnt = len(train_labels)
  em_cnt = 0
  
  print("==================================================")
  # for string, labels in zip(generated_strings, test_labels) :
  for string, labels in zip(generated_strings, train_labels) :
    # 비교를 위해 decoded string을 cleansing
    cleaned_string = string.replace("</s>", "").replace("<s>", "").strip()
    
    # Multiple answers 중 하나만 맞아도 인정!
    labels = [str(label).lower() for label in labels] # 소문자로 통일
    for label in labels :
      print(cleaned_string)
      print(label)
      print() # 줄바꿈
      
      if cleaned_string == label :
        em_cnt += 1
        break
      
  print("==================================================")
  
  print() # 줄바꿈
      
  # 최종 Score 반환
  emScore = em_cnt / total_cnt * 100
  print(f'em_cnt: {em_cnt}, total_cnt: {total_cnt}')
  print(f'EM Score -> {emScore}%')


if __name__ == "__main__" :
  
  # Check for 'gpu' device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("==================================================")
  print("Current Device ->", device)
  print("==================================================")
  
  # Call "main" function!
  webQuest()
  
  print("Success!")
