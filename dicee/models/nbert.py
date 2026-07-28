import torch 
from torch import nn
from torch.nn import functional as F 

class NBert(nn.Module): 
    def __init__(self, args, tokenizer, bert_encoder):
        super().__init__() 
        self.device = torch.device(args["device"])
        self.tokenizer = tokenizer
        self.bert_encoder = bert_encoder

        self.entity_begin_idx = args["text_entity_begin_idx"]
        self.entity_end_idx = args["text_entity_end_idx"] 

        self.bert_encoder.resize_token_embeddings(len(self.tokenizer))

    def link_prediction(self, batch):
        head_data = batch["text_head_prompts"] #gets the 2 prompt types
        tail_data = batch["text_tail_prompts"] 

        head_input_ids = head_data["input_ids"].to(self.device) 
        head_token_type_ids = head_data["token_type_ids"].to(self.device) 
        head_attention_mask = head_data["attention_mask"].to(self.device) 
        head_mask_pos = head_data["mask_pos"].to(self.device)  #head_mask_pos tells where [MASK] is in each sequence

        tail_input_ids = tail_data["input_ids"].to(self.device)
        tail_token_type_ids = tail_data["token_type_ids"].to(self.device)
        tail_attention_mask = tail_data["attention_mask"].to(self.device)
        tail_mask_pos = tail_data["mask_pos"].to(self.device)

        head_labels = batch["head_labels"].to(self.device) #Correct entity labels
        tail_labels = batch["tail_labels"].to(self.device)
        code = batch["code"] #Keeps triple ids/codes so we can later export scores

        head_output = self.bert_encoder( #Runs BERT on the head-prediction prompt
            input_ids = head_input_ids,
            token_type_ids = head_token_type_ids,
            attention_mask = head_attention_mask,
            output_hidden_states = True
        )

        tail_output = self.bert_encoder( #Runs BERT on the tail-prediction prompt
            input_ids = tail_input_ids,
            token_type_ids = tail_token_type_ids,
            attention_mask = tail_attention_mask,
            output_hidden_states = True
        )

        head_logits = head_output.logits[ #extracts BERT’s scores at the [MASK] position, but only for entity tokens
            head_mask_pos[:, 0],
            head_mask_pos[:, 1],
            self.entity_begin_idx : self.entity_end_idx
        ]

        tail_logits = tail_output.logits[
            tail_mask_pos[:, 0],
            tail_mask_pos[:, 1],
            self.entity_begin_idx : self.entity_end_idx
        ]

        batch_idx = torch.arange(head_labels.size(0), device=self.device) #Compute Correct Entity Scores

        head_logits_score = F.log_softmax(head_logits, dim=-1) #Converts logits into log-probabilities.
        tail_logits_score = F.log_softmax(tail_logits, dim=-1) 

        #final triple plausibility score
        bert_score = ( 
            head_logits_score[batch_idx, head_labels] + tail_logits_score[batch_idx, tail_labels]
        )

        #BERT’s final hidden vector at the [MASK] position
        head_repr = head_output.hidden_states[-1][
            head_mask_pos[:, 0],
            head_mask_pos[:, 1],
            :
        ]

        tail_repr = tail_output.hidden_states[-1][
            tail_mask_pos[:, 0],
            tail_mask_pos[:, 1],
            :
        ]

        #plain Python pairs for saving
        score_pairs = [
            (c, float(s)) for c, s in zip(code, bert_score.detach().cpu().tolist())
            #remove from gradient graph, move to CPU, convert to Python numbers
        ]

        """
        bert_score  -> scalar N-BERT plausibility score per triple
        head_repr   -> BERT representation for head prediction
        tail_repr   -> BERT representation for tail prediction
        score_pairs -> exportable scores
        head_logits -> full head entity logits
        tail_logits -> full tail entity logits
        """
        return {
            "bert_score": bert_score,
            "head_repr": head_repr,
            "tail_repr": tail_repr,
            "score_pairs": score_pairs,
            "head_logits": head_logits,
            "tail_logits": tail_logits
        }