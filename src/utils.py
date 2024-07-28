import argparse


# 프로그램 실행 시 전달받은 인자를 parsing하여 반환하는 메서드
def get_args() :
  
  # Create a parser for args
  parser = argparse.ArgumentParser(description="RAG-finetuning-WQ")
  
  # Add args into parser
  parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Train set's batch size")
  parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Eval set's batch size")
  parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
  parser.add_argument("--weight_decay", type=float,  default=0.0, help="Weight decay for regularization")
  parser.add_argument("--max_grad_norm", type=float,  default=1.0, help="Maximum grad norm for gradient clipping")
  parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio for lr scheduling")
  
  # Parsing args
  args = parser.parse_args()
  
  return args


# Define wandb's hyper-parameter space
def wandb_hp_space(trial) :
  return {
    "method": "bayes",
    "metric": {"name": "EM score", "goal": "maximize"},
    "parameters": {
      "learning_rate": {"min": 1e-6, "max": 1e-2},
      "weight_decay": {"min": 0.0, "max": 0.1},
      "max_grad_norm": {"min": 1.0, "max": 5.0},
      "warmup_ratio": {"min": 0.0, "max": 0.1}
    }
  }
