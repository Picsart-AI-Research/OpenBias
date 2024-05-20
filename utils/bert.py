from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class SBERTModel:
    
    def __init__(self, ckpt="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            print("Using SBERT on GPU")
            
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
    def embed_sentences(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.model.device))
            
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.detach().cpu()

    def similarity(self, sentence1, sentence2):
        sentence1_embedding = self.embed_sentences([sentence1])
        sentence2_embedding = self.embed_sentences([sentence2])
        return torch.matmul(sentence1_embedding, sentence2_embedding.T).item()

    def get_embedding(self, sentence):
        sentence_embedding = self.embed_sentences([sentence])
        return sentence_embedding
    
    def multiple_choice(self, answer, choices):
        answer_embedding = self.embed_sentences([answer])
        choices_embedding = self.embed_sentences(choices)
        top_choice_index = torch.argmax(torch.matmul(choices_embedding, answer_embedding.T)).item()
        return choices[top_choice_index]
