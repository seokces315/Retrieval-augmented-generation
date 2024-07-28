from transformers import RagTokenizer
from datasets import Dataset

import json

# Global variable
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")


# Another version of preprocessing dataset
def pp_WQ_TV_(dataset) :
  
  # Prepare lists for prompts & labels
  prompts = []
  labels = []
  for rec in dataset :
    for i in range(len(rec["answers"])) :
      prompts.append(rec["qText"])
      labels.append(rec["answers"][i].lower())
  
  # Tokenizing prompts
  input_dicts = tokenizer(
    prompts,
    padding="max_length",
    truncation=True,
    max_length=18,
    return_tensors="pt"
  )
  
  # Tokenizing labels
  labels = tokenizer(
    labels,
    padding="max_length",
    truncation=True,
    max_length=18,
    return_tensors="pt"
  )
  # Merge those matrices
  input_dicts['labels'] = labels['input_ids']
  
  return input_dicts

# Method that preprocess the dataset
def pp_WQ_TV(dataset) :
  
  # Prepare lists for prompts & labels
  prompts = []
  labels = []
  for rec in dataset :
    for i in range(len(rec["answers"])) :
      prompts.append(rec["qText"])
      labels.append(rec["answers"][i].lower())
      # labels.append(rec["answers"][0].lower())

  # Tokenizing the whole input batch
  input_dicts = tokenizer(
    text=prompts,
    text_target=labels,
    padding="max_length",
    truncation=True,
    max_length=18,
    return_tensors="pt"
  )
      
  return input_dicts

# Train/Validation dataset load method
def load_WQ_TV(train_path, val_path, verbose=False) :
  
  # train_path = "../datasets/WQ_datasets/trainmodel.json"
  # val_path = "../datasets/WQ_datasets/val.json"
    
  # Load json as python object
  with open(train_path, "r") as train_file :
    train_dataset = json.load(train_file)
  with open(val_path, "r") as val_file :
    val_dataset = json.load(val_file)
      
  # Print the info. of Train/Validation dataset
  if verbose :
    train_file_path = train_path.split("/")
    train_path_name = train_file_path[-1].replace(".json", "")
    print("==================================================")
    print(f"# File category : {train_path_name}")
    print(f"Number of Samples -> {len(train_dataset)}")
    print("==================================================")
    val_file_path = val_path.split("/")
    val_path_name = val_file_path[-1].replace(".json", "")
    print("==================================================")
    print(f"# File category : {val_path_name}")
    print(f"Number of Samples -> {len(val_dataset)}")
    print("==================================================")
  
  # Preprocessing each datasets
  train_dicts = pp_WQ_TV(train_dataset)
  val_dicts = pp_WQ_TV(val_dataset)
  
  # Transform the results : Dicts -> Dataset
  train_dataset = Dataset.from_dict(train_dicts)
  val_dataset = Dataset.from_dict(val_dicts)
  
  return train_dataset, val_dataset
  
# Test_dataset load method
def load_WQ_test(test_path, verbose=False) :
  
  # test_path = "../datasets/WQ_datasets/devtest.json"
  
  # Load json as python object
  with open(test_path, "r") as test_file :
    test_dataset = json.load(test_file)
    
  # Print the info. of test dataset
  if verbose :
    file_path = test_path.split("/")
    path_name = file_path[-1].replace(".json", "")
    print("==================================================")
    print(f"# File category : {path_name}")
    print(f"Number of Samples -> {len(test_dataset)}")
    print("==================================================")
    
  # Prepare prompts & labels
  prompts = [rec["qText"] for rec in test_dataset]
  labels = [rec["answers"] for rec in test_dataset]
  
  # Tokenizing the prompts
  input_dicts = []
  for prompt in prompts :
    input_dict = tokenizer(
      prompt,
      padding="max_length",
      truncation=True,
      max_length=18,
      return_tensors="pt"
    )
    input_dicts.append(input_dict)
    
  return input_dicts, labels

# For debugging, train dataset load method
def load_WQ_train(train_path, verbose=False) :
  
  # train_path = "../datasets/WQ_datasets/trainmodel.json"
  
  # Load json as python object
  with open(train_path, "r") as train_file :
    train_dataset = json.load(train_file)
    
  # Print the info. of test dataset
  if verbose :
    file_path = train_path.split("/")
    path_name = file_path[-1].replace(".json", "")
    print("==================================================")
    print(f"# File category : {path_name}")
    print(f"Number of Samples -> {len(train_dataset)}")
    print("==================================================")
    
  # Prepare prompts & labels
  prompts = [rec["qText"] for rec in train_dataset]
  labels = [rec["answers"] for rec in train_dataset]
  
  # Tokenizing the prompts
  input_dicts = []
  for prompt in prompts :
    input_dict = tokenizer(
      prompt,
      padding="max_length",
      truncation=True,
      max_length=18,
      return_tensors="pt"
    )
    input_dicts.append(input_dict)
    
  return input_dicts, labels
