import lightning as pl 
import re
from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig,
    T5ForConditionalGeneration
)
import torch 
from torch.optim import AdamW

class CALMT5(pl.LightningModule):

    def __init__(self, config: dict):

        super().__init__()
        self.config = config         
        

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            #bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config["alias"],
            trust_remote_code=True,
            torch_dtype="auto",
            #quantization_config=nf4_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["alias"])

    def training_step(self, batch, batch_idx): 
        outputs = self.model.forward(**batch, return_dict=True)
        loss = outputs["loss"]  
        
        self.log("loss", loss, prog_bar=True)     
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs["loss"]  
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True) 
        
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config["lr"])
        return optimizer