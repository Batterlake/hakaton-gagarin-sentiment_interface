import pytorch_lightning as pl
from transformers import AdamW, T5ForConditionalGeneration


class NERModel(pl.LightningModule):
    def __init__(self, m_name: str = "t5-small", lr=0.001, use_freeze=False):
        super().__init__()
        self.lr = lr
        self.model = T5ForConditionalGeneration.from_pretrained(
            m_name, return_dict=True
        )

        if use_freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.lm_head.parameters():
                param.requires_grad = True  # Unfreeze the last layer


    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        return output.loss, output.logits
    

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
