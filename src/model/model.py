import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (BertPreTrainedModel, BertModel, BertConfig,
                          RobertaModel, PreTrainedModel, RobertaConfig)

# for debugging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


class RobertaPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class RobertaForShuffledWordClassification(RobertaPreTrainedModel):
    """RoBERTa model for pre-training with shuffled word classification.

    References:
        https://huggingface.co/transformers/model_doc/roberta.html?highlight=roberta#transformers.RobertaModel
    """
    def __init__(self, config):
        super().__init__(config)

        # if add_pooling_layer is `True`, this will add a dense layer 
        # + `tanh` activation.
        self.roberta = RobertaModel(config, add_pooling_layer=True)

        # for token-level classification of stopwords
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_shuffle = nn.Linear(config.hidden_size, 1)

        self.init_weights()

        """
        torch.nn.init.xavier_uniform_(self.cls_shuffle.weight)
        self.cls_shuffle.bias.data.fill_(0.0)
        """


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        shuffled_word_mask=None,
        active_indices=None,
        **kwargs
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # -> (bs, seq_len, hs)
        logits = self.cls_shuffle(sequence_output).squeeze(2) # -> (bs, seq_len)

        # loss for shuffled word classification
        loss_cls_fct = nn.BCEWithLogitsLoss(reduction="none") # 5.67, 2.84, 1.42
        cls_loss = loss_cls_fct(logits.view(-1), shuffled_word_mask.view(-1).to(torch.float)) 
        # -> (bs * seq_len, )
                
        # only keep active parts of the loss
        active_indices = attention_mask.view(-1) == 1
        cls_loss = torch.masked_select(cls_loss, active_indices) # non-padding tokens
        cls_loss = torch.mean(cls_loss)
        # cls_loss = cls_loss * 10.0

        output = (logits,) + outputs[2:]
        return ((cls_loss,) + output)


class RobertaForShuffleRandomThreeWayClassification(RobertaPreTrainedModel):
    """RoBERTa model for pre-training with three-way shuffled/random/non-replaced classification.

    References:
        https://huggingface.co/transformers/model_doc/roberta.html?highlight=roberta#transformers.RobertaModel
    """
    def __init__(self, config):
        super().__init__(config)

        # if add_pooling_layer is `True`, this will add a dense layer 
        # + `tanh` activation.
        self.roberta = RobertaModel(config, add_pooling_layer=True)

        # for token-level three-way classification
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, 3) # 0 (non-replaced), 1 (shuffled), 2 (random)

        self.init_weights()
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        shuffle_random_mask=None,
        **kwargs
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # -> (bs, seq_len, hs)
        logits = self.dense(sequence_output) # -> (bs, seq_len, 3)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 3), shuffle_random_mask.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output)


class RobertaForFourWayTokenTypeClassification(RobertaPreTrainedModel):
    """RoBERTa model for pre-training with four-way token type classification."""
    def __init__(self, config):
        super().__init__(config)

        # if add_pooling_layer is `True`, this will add a dense layer 
        # + `tanh` activation.
        self.roberta = RobertaModel(config, add_pooling_layer=True)

        # for token-level four-way classification
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, 4)
        # -> 0: context, 1: stop word, 2: digit, 3: punctuation

        self.init_weights()
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        masked_word_labels=None,
        **kwargs
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # -> (bs, seq_len, hs)
        logits = self.dense(sequence_output) # -> (bs, seq_len, 4)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 4), masked_word_labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output)


class RobertaForFirstCharPrediction(RobertaPreTrainedModel):
    """RoBERTa model for pre-training with masked first character classification."""
    def __init__(self, config):
        super().__init__(config)

        # if add_pooling_layer is `True`, this will add a dense layer 
        # + `tanh` activation.
        self.roberta = RobertaModel(config, add_pooling_layer=True)

        # for token-level four-way classification
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, 29)
        # -> 0~25: alphabet, 26: digit, 27: punctuation, 28: exception

        self.init_weights()
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        masked_word_labels=None,
        **kwargs
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # -> (bs, seq_len, hs)
        logits = self.dense(sequence_output) # -> (bs, seq_len, 29)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 29), masked_word_labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output)


class RobertaForRandomWordClassification(RobertaPreTrainedModel):
    """RoBERTa model for pre-training with random word classification."""
    def __init__(self, config):
        super().__init__(config)

        # if add_pooling_layer is `True`, this will add a dense layer 
        # + `tanh` activation.
        self.roberta = RobertaModel(config, add_pooling_layer=True)

        # for token-level classification of stopwords
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_random = nn.Linear(config.hidden_size, 1)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        random_word_mask=None,
        active_indices=None,
        **kwargs
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # -> (bs, seq_len, hs)
        logits = self.cls_random(sequence_output).squeeze(2) # -> (bs, seq_len)

        # loss for shuffled word classification
        loss_cls_fct = nn.BCEWithLogitsLoss(reduction="none")
        cls_loss = loss_cls_fct(logits.view(-1), random_word_mask.view(-1).to(torch.float)) 
        # -> (bs * seq_len, )
                
        # only keep active parts of the loss
        active_indices = attention_mask.view(-1) == 1
        cls_loss = torch.masked_select(cls_loss, active_indices) # non-padding tokens
        cls_loss = torch.mean(cls_loss)

        output = (logits,) + outputs[2:]
        return ((cls_loss,) + output)
