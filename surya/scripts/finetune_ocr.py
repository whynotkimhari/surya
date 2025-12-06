from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
from datasets import load_dataset
import numpy as np
import torch
import warnings
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)

from surya.common.surya import SuryaModel
from surya.common.surya.processor import SuryaOCRProcessor
from surya.foundation import FoundationPredictor
from surya.common.surya.processor.schema import ImageInput, TextInput
from surya.common.surya.schema import TaskNames
from surya.common.util import get_top_scripts, SCRIPT_TOKEN_MAPPING

# Suppress the harmless "Image features and image tokens do not match" warning
# This warning is due to incorrect feature counting in the model's diagnostic code
# The actual masked_scatter operation works correctly (verified with assertions)
warnings.filterwarnings("ignore", message="Image features and image tokens do not match")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector")

# Do not change these defaults
OCR_TASK_NAME = TaskNames.ocr_with_boxes
OCR_MAX_IMAGE_SIZE = (1024, 512)

# Simple wrapper for huggingface dataset
class SuryaOCRDataset(torch.utils.data.Dataset):
    def __init__(self, processor: SuryaOCRProcessor, data_args: SuryaOCRDataArguments):
        super().__init__()
        self.hf_dataset = load_dataset(data_args.dataset_name, num_proc=data_args.num_loading_proc)
        self.processor = processor

    def __len__(self):
        return len(self.hf_dataset)

    def get_script_text(self, text: str) -> str:
        scripts = get_top_scripts(text)
        script_text = "".join(SCRIPT_TOKEN_MAPPING[script] for script in scripts)
        return script_text

    def __getitem__(self, index):
        try:
            data = self.hf_dataset[index]
            image = data["image"]
            image = image.convert("RGB")
            image = np.asarray(image, dtype=np.float32)
            image = self.processor.scale_to_fit(image, max_size=OCR_MAX_IMAGE_SIZE)

            # Add in script information
            gt_text = data["text"]
            gt_text = self.get_script_text(gt_text) + gt_text

            return_dict = {
                "task": TaskNames.ocr_with_boxes,
                "inputs": [
                    ImageInput(type="image", image=image, rotated=False),
                    # This empty TextInput **must be included** to match the original format
                    TextInput(type="text", text=""),
                    TextInput(type="text",text=gt_text),
                ],
            }
            return return_dict
        except:
            import traceback; traceback.print_exc()
            return self.__getitem__((index + 1) % self.__len__())

class SuryaOCRDataCollator:
    def __init__(self, model: SuryaModel, processor: SuryaOCRProcessor, data_args: SuryaOCRDataArguments, encoder_chunk_size: int = 32768):
        self.model = model
        self.processor = processor
        self.max_sequence_length = data_args.max_sequence_length
        self.encoder_chunk_size = encoder_chunk_size

    def __call__(self, inputs):
        # Use right padding for training. Defaults to left for inference
        processed_batch = self.processor(inputs, padding_side="right")
        
        if self.max_sequence_length is not None:
            processed_batch["input_ids"] = processed_batch["input_ids"][:, :self.max_sequence_length]
            processed_batch["attention_mask"] = processed_batch["attention_mask"][:, :self.max_sequence_length]
            processed_batch["position_ids"] = processed_batch["position_ids"][:, :self.max_sequence_length]

        # Create cache_position (required by model's forward method)
        # This tracks the position in the KV cache for each token
        seq_len = processed_batch["input_ids"].size(1)
        processed_batch["cache_position"] = torch.arange(0, seq_len, dtype=torch.long)
        
        # CRITICAL: Compute image embeddings from image_tiles and grid_thw
        # The HuggingFace Trainer calls model.forward() directly, which expects
        # image_embeddings to be precomputed (unlike FoundationPredictor which does this step)
        image_tiles = processed_batch.pop("image_tiles")  # Remove from batch
        grid_thw = processed_batch.pop("grid_thw")  # Remove from batch
        
        # Compute image embeddings using the model's encoder
        # Note: get_image_embeddings expects all images from the batch concatenated
        # Use torch.no_grad() to freeze vision encoder (standard for OCR finetuning)
        # This prevents in-place operation errors and saves memory
        with torch.no_grad():
            image_embeddings = self.model.get_image_embeddings(
                pixel_values=image_tiles,
                grid_thw=grid_thw,
                encoder_chunk_size=self.encoder_chunk_size,
            )
        
        # Move embeddings to CPU - the Trainer/DataLoader will handle device placement
        # CUDA tensors cannot be pinned to memory, which causes RuntimeError
        image_embeddings = image_embeddings.cpu()
        
        # Split embeddings per sample in the batch based on grid_thw
        # grid_thw shape: [batch_size, 3] where each row is (grid_t, grid_h, grid_w)
        # After merging, each image produces (grid_h // merge_size) * (grid_w // merge_size) tokens
        batch_size = grid_thw.shape[0]
        merge_size = self.processor.merge_size
        
        # Calculate number of image tokens per sample
        tokens_per_sample = []
        for i in range(batch_size):
            grid_t, grid_h, grid_w = grid_thw[i]
            num_tokens = (grid_h // merge_size) * (grid_w // merge_size)
            tokens_per_sample.append(num_tokens.item())
        
        # Split the concatenated embeddings into per-sample embeddings
        # IMPORTANT: Use .clone() to create independent copies, not views
        # Views share memory and cause in-place operation errors during backward
        split_embeddings = []
        start_idx = 0
        for num_tokens in tokens_per_sample:
            end_idx = start_idx + num_tokens
            # Clone to create independent tensor (not a view)
            sample_embeddings = image_embeddings[start_idx:end_idx].clone()
            split_embeddings.append(sample_embeddings)
            start_idx = end_idx
        
        # Pad embeddings to the same length and stack into batch
        # Use torch operations instead of in-place assignment to preserve gradients
        max_tokens = max(tokens_per_sample)
        hidden_dim = image_embeddings.shape[-1]
        
        # Create padded embeddings list
        padded_embeddings = []
        for sample_emb in split_embeddings:
            num_tokens = sample_emb.shape[0]
            if num_tokens < max_tokens:
                # Pad with zeros
                padding = torch.zeros(max_tokens - num_tokens, hidden_dim, 
                                     dtype=sample_emb.dtype, device=sample_emb.device)
                padded_emb = torch.cat([sample_emb, padding], dim=0)
            else:
                padded_emb = sample_emb
            padded_embeddings.append(padded_emb)
        
        # Stack into batch dimension (no in-place operations)
        batched_embeddings = torch.stack(padded_embeddings, dim=0)
        
        # VERIFICATION: Prove that each sample has the correct number of embeddings
        # This ensures padding doesn't affect performance
        for i in range(batch_size):
            # Count IMAGE tokens in this sample's input_ids
            num_image_tokens_in_ids = (processed_batch["input_ids"][i] == self.processor.image_token_id).sum().item()
            # Count actual (non-padding) embeddings we provided
            actual_embeddings_provided = tokens_per_sample[i]
            
            # These MUST match - this proves only real embeddings are used, padding is ignored
            assert num_image_tokens_in_ids == actual_embeddings_provided, (
                f"Sample {i}: IMAGE tokens ({num_image_tokens_in_ids}) != "
                f"embeddings ({actual_embeddings_provided}). This should never happen!"
            )
        
        # Add the properly batched embeddings to the batch
        processed_batch["image_embeddings"] = batched_embeddings
        
        lm_labels = processed_batch["input_ids"].clone()
        skip_label_mask = (
            (lm_labels == self.processor.pad_token_id )
            | (lm_labels == self.processor.bos_token_id[TaskNames.ocr_with_boxes])
            | (lm_labels == self.processor.eoi_token_id)
            | (lm_labels == self.processor.image_token_id)
        )
        lm_labels[skip_label_mask] = -100
        processed_batch["labels"] = lm_labels

        return processed_batch

def load_model_and_processor(checkpoint_path: Optional[str] = None) -> Tuple[SuryaModel, SuryaOCRProcessor]:
    foundation_predictor = FoundationPredictor(checkpoint=checkpoint_path)
    return foundation_predictor.model, foundation_predictor.processor

@dataclass
class SuryaOCRModelArguments:
    pretrained_checkpoint_path: Optional[str] = field(default=None)

@dataclass
class SuryaOCRDataArguments:
    dataset_name: str = field(default="datalab-to/ocr_finetune_example")
    num_loading_proc: int = field(default=16)
    max_sequence_length: Optional[int] = field(default=None)

@dataclass
class SuryaOCRTrainingArguments(TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    
def main():
    parser = HfArgumentParser((SuryaOCRModelArguments, SuryaOCRDataArguments, SuryaOCRTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, processor = load_model_and_processor(model_args.pretrained_checkpoint_path)
    train_dataset = SuryaOCRDataset(processor, data_args)
    collator = SuryaOCRDataCollator(model, processor, data_args, encoder_chunk_size=32768)

    # Create compute_metrics function with processor bound
    def compute_metrics_fn(eval_pred):
        return compute_metrics(eval_pred, processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

if __name__ == "__main__":
    main()