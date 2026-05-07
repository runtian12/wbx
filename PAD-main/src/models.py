import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Optional, Union, Tuple, List
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, BertConfig
import copy


@dataclass
class SequenceClassifierOutputWithNextTokenLogits(SequenceClassifierOutputWithPast):
    next_token_logits: torch.FloatTensor = None

@dataclass
class TokenClassifierOutputWithNextTokenLogits(SequenceClassifierOutputWithNextTokenLogits):
    pass


# Subclass the LlamaForCausalLM to add a classification head.
class MultiHeadLlamaForSequenceClassification(nn.Module):

    def __init__(
        self,
        causal_model,
        pad_token_id,
        num_labels=2,
        problem_type="single_label_classification",
        mlp_n_layers=1,
        mlp_proj_dim=None,
        class_weights=None,
        mlp_dropout_rate=0.1,
        score_head_type="LLamaDecoderLayer", # options: "LLamaDecoderLayer", "bert", "linear"
        tokenizer=None,
    ):
        super().__init__()
        self.setup_config(causal_model, pad_token_id, num_labels, problem_type, class_weights, score_head_type, mlp_n_layers, mlp_proj_dim, mlp_dropout_rate, tokenizer)
        self.setup_score_head()

    def setup_config(self, causal_model, pad_token_id, num_labels, problem_type, class_weights, score_head_type, mlp_n_layers, mlp_proj_dim, mlp_dropout_rate, tokenizer):
        self.causal_model = causal_model  # LLaMA transformer
        self.config = causal_model.config
        self.config.problem_type = problem_type
        self.config.pad_token_id = pad_token_id
        self.num_labels = num_labels
        self.device = self.causal_model.device
        self.class_weights = class_weights
        self.score_head_type = score_head_type
        self.mlp_n_layers = mlp_n_layers
        self.mlp_proj_dim = mlp_proj_dim
        self.mlp_dropout_rate = mlp_dropout_rate
        self.hf_auto_model_func = AutoModelForSequenceClassification
        self.tokenizer = tokenizer

    def setup_score_head(self):
        if self.score_head_type == "LLamaDecoderLayer":
            self.setup_llama_decoder_head()
        elif self.score_head_type == "bert":
            self.setup_bert_head()
        else:
            self.setup_mlp_head()

    def setup_llama_decoder_head(self):
        head_config = copy.deepcopy(self.config)
        head_config.num_hidden_layers = 1
        head_config.num_labels = 2
        self.score = self.hf_auto_model_func.from_config(head_config)
        self.score = self.score.to(self.device)

    def setup_bert_head(self):
        head_model_name = "bert-base-uncased"
        head_config = BertConfig.from_pretrained(head_model_name)
        head_config.num_labels = 2
        # head_config.hidden_size = self.config.hidden_size
        # head_config.num_attention_heads = 16
        self.projec = nn.Linear(self.config.hidden_size, head_config.hidden_size, device=self.device)
        self.score = self.hf_auto_model_func.from_config(head_config)
        self.score = self.score.to(self.device)

        # head_model_name = "bert-base-uncased"
        # self.score = AutoModelForTokenClassification.from_pretrained(head_model_name, num_labels=2)
        # head_config = self.score.config
        # self.score = self.score.to(self.device)
        # self.projec = nn.Linear(self.config.hidden_size, head_config.hidden_size, device=self.device)

    def setup_mlp_head(self):
        # Classification head
        # self.score = nn.Linear(self.config.hidden_size, self.num_labels, device=self.device)
        # Modified classification head with regularization
        
        if self.mlp_n_layers == 1:
            self.score = nn.Sequential(
                nn.Dropout(self.mlp_dropout_rate),
                nn.Linear(self.config.hidden_size, self.num_labels, device=self.device)
            )
        else:
            if not self.mlp_proj_dim:
                self.mlp_proj_dim = self.config.hidden_size

            self.score = nn.Sequential(
                nn.Dropout(self.mlp_dropout_rate),
                nn.Linear(self.config.hidden_size, self.mlp_proj_dim, device=self.device),
                # nn.LayerNorm(self.config.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.mlp_dropout_rate),
                nn.Linear(self.mlp_proj_dim, self.num_labels, device=self.device)
            )

        # Initialize weights
        for module in self.score.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithNextTokenLogits]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.causal_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True, # TODO: this my not be optimal since it requires storing all hidden states
            return_dict=True,  # we need access to hidden_states
        )

        # hidden_states = transformer_outputs[0]  # last hidden state
        next_token_logits = transformer_outputs.logits  # shape: (batch_size, seq_len, vocab_size)
        hidden_states = transformer_outputs.hidden_states[-1]  # last hidden state

        hidden_states = hidden_states.float()
        if self.score_head_type == "bert":
            ## Didn't work well, maybe should try again
            raise NotImplementedError("Bert head not implemented yet, We don't have a CLS token")
            # take the last 512 tokens
            hidden_states = hidden_states[:, -512:, :]
            attention_mask = attention_mask[:, -512:]
            hidden_states = self.projec(hidden_states)
            head_output = self.score(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = head_output.logits
            loss = head_output.loss
            pooled_logits = logits
        elif self.score_head_type == "LLamaDecoderLayer" or self.score_head_type == "mlp":
            if self.score_head_type == "LLamaDecoderLayer":
                head_output = self.score(
                    inputs_embeds=hidden_states,
                    attention_mask=attention_mask,
                )
                logits = head_output.logits
            else:
                logits = self.score(hidden_states)  # shape: (batch_size, seq_len, num_labels)

            batch_size = input_ids.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError(
                    "Cannot handle batch sizes > 1 if no pad token is defined."
                )

            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    # Find the index of the padding token, and get the last non-padding token
                    sequence_lengths = (
                        torch.eq(input_ids, self.config.pad_token_id)
                        .int()
                        .argmax(dim=-1)
                        - 1
                    )
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(logits.device)
                else:
                    sequence_lengths = -1

            # pooled_logits: take the last token of each sequence (based on padding) to classify
            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device), sequence_lengths
            ]

            loss = None
            if labels is not None:
                labels = labels.to(pooled_logits.device)
                # Infer problem type
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                    ):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    raise NotImplementedError("Regression is not implemented yet.")
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(pooled_logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    # + class weights
                    if self.class_weights is not None:
                        self.class_weights = self.class_weights.to(pooled_logits.device)
                    loss_fct = CrossEntropyLoss(weight=self.class_weights)
                    loss = loss_fct(
                        pooled_logits.view(-1, self.num_labels), labels.view(-1)
                    )
                elif self.config.problem_type == "multi_label_classification":
                    raise NotImplementedError(
                        "Multi-label classification is not implemented yet."
                    )
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(pooled_logits, labels.float())

        else:
            raise ValueError(f"Invalid score_head_type: {self.score_head_type}")
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithNextTokenLogits(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            next_token_logits=next_token_logits,
        )

    def save_head(self, path):
        print(f"Saving head to {path}")
        torch.save(self.score.state_dict(), path)

    def load_head(self, path):
        self.score.load_state_dict(torch.load(path))


class MultiHeadLlamaForTokenClassification(MultiHeadLlamaForSequenceClassification):

    def setup_config(self, *args, **kwargs):
        super().setup_config(*args, **kwargs)
        self.hf_auto_model_func = AutoModelForTokenClassification
        

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutputWithNextTokenLogits]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.causal_model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True, # TODO: this my not be optimal since it requires storing all hidden states
                return_dict=True,  # we need access to hidden_states
            )
        next_token_logits = transformer_outputs.logits  # shape: (batch_size, seq_len, vocab_size)
        hidden_states = transformer_outputs.hidden_states[-1]  # last hidden state
        hidden_states = hidden_states.float()

        if self.score_head_type == "LLamaDecoderLayer":
            head_output = self.score(
                    inputs_embeds=hidden_states,
                    attention_mask=attention_mask,
                    # labels=labels,
                )
            logits = head_output.logits
            # loss = head_output.loss
        elif self.score_head_type == "bert":
            # hidden_states = hidden_states[:, -512:, :]
            # labels = labels[:, -512:]
            # attention_mask = attention_mask[:, -512:]
            # print(hidden_states.shape)
            # print(labels.shape)
            hidden_states = self.projec(hidden_states)
            head_output = self.score(
                    inputs_embeds=hidden_states,
                    attention_mask=attention_mask,
                    # labels=labels,
                )
            logits = head_output.logits
            # loss = head_output.loss
        else:
            logits = self.score(hidden_states)  # shape: (batch_size, seq_len, num_labels)

        if labels is not None:
            if self.class_weights is not None:
                self.class_weights = self.class_weights.to(logits.device)
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputWithNextTokenLogits(
                loss=loss,
                logits=logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                next_token_logits=next_token_logits,
            )
