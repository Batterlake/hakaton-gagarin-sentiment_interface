import pytorch_lightning as pl
import torch.nn as nn
from transformers import AdamW, BertModel


class BERTClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "",
        num_issuers_classes: int = 250,
        num_sentiment_classes: int = 5,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.issuers_head = nn.Sequential(
            nn.Linear(312, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_issuers_classes),
            nn.Softmax(dim=1),
        )
        self.sentiment_head = nn.Sequential(
            nn.Linear(312, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_sentiment_classes),
            nn.Softmax(dim=1),
        )

    def criterion(
        self, issuers_output, issuer_labels, sentiment_output, sentiment_labels
    ):
        return nn.BCELoss()(issuers_output, issuer_labels) + nn.BCELoss()(
            sentiment_output, sentiment_labels
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output
        issuer_output = self.issuers_head(bert_output)
        sentiment_output = self.sentiment_head(bert_output)

        return issuer_output, sentiment_output

    def training_step(self, batch, batch_idx):
        issuer_output, sentiment_output = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.criterion(
            issuer_output,
            batch["issuer_labels"],
            sentiment_output,
            batch["sentiment_labels"],
        )
        self.log("loss/train", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        issuer_output, sentiment_output = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.criterion(
            issuer_output,
            batch["issuer_labels"],
            sentiment_output,
            batch["sentiment_labels"],
        )
        self.log("loss/val", loss, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        issuer_output, sentiment_output = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        loss = self.criterion(
            issuer_output,
            batch["issuer_labels"],
            sentiment_output,
            batch["sentiment_labels"],
        )
        self.log("loss/test", loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3)
