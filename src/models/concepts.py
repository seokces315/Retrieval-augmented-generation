from transformers import AutoTokenizer, RagRetriever, RagTokenForGeneration
import torch


tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
)
# model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
model = RagTokenForGeneration.from_pretrained(
    "facebook/rag-token-nq", use_dummy_dataset=True
)

query = "who holds the record in 100m freestyle"
inputs = tokenizer(query, return_tensors="pt")
# targets = tokenizer(text_target="Michael Phelps hold the record in 100m freestyle", return_tensors="pt")
input_ids = inputs["input_ids"]
# labels = targets["input_ids"]
# outputs = model(input_ids=input_ids, labels=labels)

question_hidden_states = model.question_encoder(input_ids)[0]
docs_dict = retriever(
    input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt"
)
doc_scores = torch.bmm(
    question_hidden_states.unsqueeze(1),
    docs_dict["retrieved_doc_embeds"].float().transpose(1, 2),
).squeeze(1)

# outputs = model(
#    context_input_ids=docs_dict["context_input_ids"],
#    context_attention_mask=docs_dict["context_attention_mask"],
#    doc_scores=doc_scores,
#    decoder_input_ids=labels,
# )
generated = model.generate(
    context_input_ids=docs_dict["context_input_ids"],
    context_attention_mask=docs_dict["context_attention_mask"],
    doc_scores=doc_scores,
)
generated_string = tokenizer.batch_decode(generated, skip_speical_tokens=True)
generated_string = generated_string[0].replace("</s>", "").replace("<s>", "").strip()

print("[ Query : %s]" % query)
print("Answer) %s" % generated_string)
print("Success!\n")
