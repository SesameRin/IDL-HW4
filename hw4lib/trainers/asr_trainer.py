from .base_trainer import BaseTrainer
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from ..decoding.sequence_generator import SequenceGenerator
from ..utils import create_scheduler, create_optimizer
from ..model import DecoderOnlyTransformer
import torchaudio.functional as aF
import json
import torchmetrics.text as tmt
from torch.utils.data import Subset
import pandas as pd


class ASRTrainer(BaseTrainer):
    """
    ASR (Automatic Speech Recognition) Trainer class that handles training, validation, and recognition loops.

    This trainer implements:
    1. Training loop with gradient accumulation, mixed precision training, and optional CTC loss
    2. Validation loop for model evaluation
    3. Recognition capabilities with different decoding strategies (greedy, beam search)
    4. Language model shallow fusion during recognition

    Implementation Tasks:
    - TODO: Initialize CE and CTC loss in __init__
    - TODO: Implement key parts of the training loop in _train_epoch
    - TODO: Implement recognition functionality in recognize
    - TODO: Implement key parts of the validation loop in _validate_epoch
    - TODO: Implement key parts of the full training loop in train

    Implementation Notes:
    1. For __init__:
        - Initialize CrossEntropyLoss with appropriate padding index and label smoothing
        - Initialize CTCLoss if ctc_weight > 0
        
    2. For _train_epoch:
        - Unpack the batch (features, shifted targets, golden targets, lengths)
        - Get model predictions, attention weights and CTC inputs
        - Calculate CE loss and CTC loss if enabled
        - Backpropagate the loss
        
    3. For _validate_epoch:
        - Use recognize() to generate transcriptions
        - Extract references and hypotheses from recognition results
        
    4. For train:
        - Set maximum transcript length
        - Implement epoch loop with training and validation
        
    5. For recognize:
        - Run inference
        - Handle both greedy and optionally beam search decoding
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)

        # ── 1. Cross-Entropy (CE) loss ────────────────────────────────────────────
        ls_val = self.config["loss"].get("label_smoothing", 0.0)
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id, label_smoothing=ls_val
        )

        # ── 2. CTC loss（可选）─────────────────────────────────────────────────────
        self.ctc_weight = self.config["loss"].get("ctc_weight", 0.0)
        self.ctc_criterion = None
        if self.ctc_weight > 0:
            # 使用 pad_id 作为 blank token，避免冲突
            self.ctc_criterion = nn.CTCLoss(
                blank=self.tokenizer.pad_id, zero_infinity=True
            )

        # ── 3. 优化器 / 学习率调度器（如果还没创建）───────────────────────────────
        if self.optimizer is None:
            self.optimizer = create_optimizer(self.model, self.config["optimizer"])
        if self.scheduler is None:
            self.scheduler = create_scheduler(
                self.optimizer, self.config["scheduler"], None
            )  # dataloader 训练时再 set total_steps

    def _train_epoch(self, dataloader):
        """One epoch training loop."""
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                         desc="[Training ASR]", leave=False)
        running_ce, running_ctc, running_joint, total_tokens = 0, 0, 0, 0
        running_att = {}

        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # ---- unpack & move to device --------------------------------------------------------
            feats, tgt_shifted, tgt_golden, feat_lens, txt_lens = [
                x.to(self.device) if torch.is_tensor(x) else x for x in batch
            ]

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                # ---- forward --------------------------------------------------------------------
                seq_out, att_dict, ctc_inputs = self.model(
                    feats, tgt_shifted, feat_lens, txt_lens
                )
                # keep latest attention for visualization
                running_att = att_dict

                # ---- CE loss --------------------------------------------------------------------
                ce_loss = self.ce_criterion(
                    seq_out.reshape(-1, seq_out.size(-1)),
                    tgt_golden.reshape(-1)
                )

                # ---- CTC loss (optional) --------------------------------------------------------
                if self.ctc_weight > 0:
                    logp = ctc_inputs["log_probs"]          # (T, B, C)
                    enc_lens = ctc_inputs["lengths"]         # (B,)
                    # concat target without padding/SOS
                    targets_flat = torch.cat([
                        tgt_golden[b, :txt_lens[b]].to(torch.long)
                        for b in range(tgt_golden.size(0))
                    ])
                    ctc_loss = self.ctc_criterion(
                        logp, targets_flat,
                        enc_lens.to(torch.long),
                        txt_lens.to(torch.long)
                    )
                    loss = ce_loss + self.ctc_weight * ctc_loss
                else:
                    ctc_loss = torch.tensor(0.0, device=self.device)
                    loss = ce_loss

            # ---- statistics --------------------------------------------------------------------
            token_cnt = txt_lens.sum().item()
            total_tokens += token_cnt
            running_ce += ce_loss.item() * token_cnt
            running_ctc += ctc_loss.item() * token_cnt
            running_joint += loss.item() * token_cnt

            # ---- backward & optim --------------------------------------------------------------
            loss = loss / self.config['training']['gradient_accumulation_steps']
            self.scaler.scale(loss).backward()

            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

            # ---- progress bar ------------------------------------------------------------------
            avg_ce = running_ce / total_tokens
            avg_ctc = running_ctc / total_tokens
            avg_joint = running_joint / total_tokens
            batch_bar.set_postfix(
                ce=f"{avg_ce:.3f}",
                ctc=f"{avg_ctc:.3f}",
                joint=f"{avg_joint:.3f}"
            )
            batch_bar.update()

            # free memory
            del feats, tgt_shifted, tgt_golden, seq_out, loss
            torch.cuda.empty_cache()

        # handle leftover gradients
        if (len(dataloader) % self.config['training']['gradient_accumulation_steps']) != 0:
            self.scaler.step(self.optimizer)
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        batch_bar.close()
        ppl_token = torch.exp(torch.tensor(running_ce / total_tokens))
        ppl_char = torch.exp(torch.tensor(
            running_ce / total_tokens /
            dataloader.dataset.get_avg_chars_per_token()
        ))

        return {
            "ce_loss": running_ce / total_tokens,
            "ctc_loss": running_ctc / total_tokens,
            "joint_loss": running_joint / total_tokens,
            "perplexity_token": ppl_token.item(),
            "perplexity_char": ppl_char.item()
        }, running_att

    def _validate_epoch(self, dataloader):
        """Greedy decoding + metric computation on validation set."""
        greedy_cfg = {
            "beam_width": 1,
            "temperature": 1.0,
            "repeat_penalty": 1.0,
            "num_batches": None,
            "lm_weight": 0.0,
            "lm_model": None,
        }
        results = self.recognize(dataloader, greedy_cfg, "val_greedy")

        refs = [r["target"] for r in results if "target" in r]
        hyps = [r["generated"] for r in results]
        metrics = self._calculate_asr_metrics(refs, hyps)
        return metrics, results

    def train(self, train_dataloader, val_dataloader, epochs: int):
        """Standard training loop with validation & checkpointing."""
        # max transcript length（供 recognize）
        self.text_max_len = max(
            val_dataloader.dataset.text_max_len, train_dataloader.dataset.text_max_len
        )

        # set scheduler total steps if需要
        if (
            self.scheduler is None
            or getattr(self.scheduler, "_initialized", False) is False
        ):
            self.scheduler = create_scheduler(
                self.optimizer, self.config["scheduler"], train_dataloader
            )

        best_cer = float("inf")

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            print(f"\n=== Epoch {epoch} / {self.current_epoch + epochs - 1} ===")
            train_metrics, train_att = self._train_epoch(train_dataloader)
            val_metrics, val_results = self._validate_epoch(val_dataloader)

            # plateau LR
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["cer"])

            # logging & plots
            metrics_pack = {"train": train_metrics, "val": val_metrics}
            self._log_metrics(metrics_pack, epoch)

            if train_att:  # 保存第一层 self-attn 与最后层 cross-attn
                self._save_attention_plot(
                    train_att[[k for k in train_att if "dec_self" in k][0]][0],
                    epoch,
                    "decoder_self",
                )
                self._save_attention_plot(
                    train_att[[k for k in train_att if "dec_cross" in k][-1]][0],
                    epoch,
                    "decoder_cross",
                )

            self._save_generated_text(val_results, f"val_epoch_{epoch}")

            # checkpoint
            self.save_checkpoint("checkpoint-last-epoch-model.pth")
            if val_metrics["cer"] < best_cer:
                best_cer = val_metrics["cer"]
                self.best_metric = best_cer
                self.save_checkpoint("checkpoint-best-metric-model.pth")

            self.current_epoch += 1

    def evaluate(self, dataloader, max_length: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on the test set. Sequentially evaluates with each recognition config.
        
        Args:
            dataloader: DataLoader for test data
            max_length: Optional[int], maximum length of the generated sequence
        Returns:
            Dictionary containing recognition results for each recognition config
            Each result is a pandas DataFrame with columns 'id' and 'transcription'
        """

        # Get recognition configs
        recognition_configs = self._get_evaluation_recognition_configs()

        eval_results = {}
        # Evaluate with each recognition config
        for config_name, config in recognition_configs.items():
            try:
                print(f"Evaluating with {config_name} config")
                results = self.recognize(dataloader, config, config_name, max_length)     
                # Calculate metrics on full batch
                generated = [r['generated'] for r in results]
                results_df = pd.DataFrame(
                    {
                        'id': range(len(generated)),
                        'transcription': generated
                    }
                )
                eval_results[config_name] = results_df
                self._save_generated_text(results, f'test_{config_name}_results')
            except Exception as e:
                print(f"Error evaluating with {config_name} config: {e}")
                continue

        return eval_results

    def recognize(
        self, dataloader, recognition_config=None, config_name=None, max_length=None
    ):
        """Inference / decoding entry (greedy or beam)."""
        if max_length is None:
            max_length = getattr(self, "text_max_len", None)
        if max_length is None:
            raise ValueError("max_length 未设置，先训练一次或手动指定。")

        # 默认 greedy
        if recognition_config is None:
            recognition_config = {
                "num_batches": None,
                "beam_width": 1,
                "temperature": 1.0,
                "repeat_penalty": 1.0,
                "lm_weight": 0.0,
                "lm_model": None,
            }
            config_name = "greedy"

        # 启用 LM（浅融合）
        if recognition_config.get("lm_model") is not None:
            recognition_config["lm_model"].eval().to(self.device)

        generator = SequenceGenerator(
            score_fn=None,
            tokenizer=self.tokenizer,
            max_length=max_length,
            device=self.device,
        )

        self.model.eval()
        results = []
        batch_bar = tqdm(
            total=len(dataloader),
            dynamic_ncols=True,
            desc=f"[Recognizing] {config_name}",
            leave=False,
        )

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                feats, _, tgt_gold, feat_lens, txt_lens = [
                    x.to(self.device) if torch.is_tensor(x) else x for x in batch
                ]
                enc_out, pad_mask_src, _, _ = self.model.encode(feats, feat_lens)

                # 定义 score_fn（支持浅融合）
                def score_fn(x):
                    asr_logits = self.model.score(x, enc_out, pad_mask_src)
                    if recognition_config["lm_model"] is not None:
                        lm_logits = recognition_config["lm_model"].score(x)
                        return asr_logits + recognition_config["lm_weight"] * lm_logits
                    return asr_logits

                generator.score_fn = score_fn

                batch_size = feats.size(0)
                prompts = torch.full(
                    (batch_size, 1),
                    self.tokenizer.sos_id,
                    dtype=torch.long,
                    device=self.device,
                )

                # ---- Decode --------------------------------------------------
                if recognition_config["beam_width"] > 1:
                    seqs, scores = generator.generate_beam(
                        prompts,
                        beam_width=recognition_config["beam_width"],
                        temperature=recognition_config["temperature"],
                        repeat_penalty=recognition_config["repeat_penalty"],
                    )
                    seqs, scores = seqs[:, 0, :], scores[:, 0]  # 取 best beam
                else:
                    seqs, scores = generator.generate_greedy(
                        prompts,
                        temperature=recognition_config["temperature"],
                        repeat_penalty=recognition_config["repeat_penalty"],
                    )

                preds_pp = generator.post_process_sequence(seqs, self.tokenizer)

                if tgt_gold is not None:
                    tgts_pp = generator.post_process_sequence(tgt_gold, self.tokenizer)
                    for j in range(batch_size):
                        results.append(
                            {
                                "target": self.tokenizer.decode(
                                    tgts_pp[j].tolist(), skip_special_tokens=True
                                ),
                                "generated": self.tokenizer.decode(
                                    preds_pp[j].tolist(), skip_special_tokens=True
                                ),
                                "score": scores[j].item(),
                            }
                        )
                else:
                    for j in range(batch_size):
                        results.append(
                            {
                                "generated": self.tokenizer.decode(
                                    preds_pp[j].tolist(), skip_special_tokens=True
                                ),
                                "score": scores[j].item(),
                            }
                        )

                batch_bar.update()
                if (
                    recognition_config["num_batches"] is not None
                    and i >= recognition_config["num_batches"] - 1
                ):
                    break

                # mem 清理
                del feats, enc_out, pad_mask_src, prompts
                torch.cuda.empty_cache()

        batch_bar.close()
        return results

    def _get_evaluation_recognition_configs(self, lm_model: Optional[DecoderOnlyTransformer] = None, lm_weight: float = 0.0) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of recognition configurations for seqential evaluation.
        
        Returns:
            Dictionary containing recognition configurations
        """

        common_config = {
            'num_batches': None,
            'temperature': 1.0,
            'repeat_penalty': 1.0,
            'lm_weight': lm_weight,
            'lm_model': lm_model
        }
        greedy_config = common_config.copy()
        greedy_config.update({
            'beam_width': 1,
        })

        beam_10_config = common_config.copy()
        beam_10_config.update({
            'beam_width': 10,
        })

        beam_20_config = common_config.copy()
        beam_20_config.update({
            'beam_width': 20,
        })

        return {
            'greedy': greedy_config,
            'beam_10': beam_10_config,
            'beam_20': beam_20_config
        }

    def _calculate_asr_metrics(self, references: Union[str, List[str]], hypotheses: Union[str, List[str]]) -> Tuple[float, float, float]:
        """
        Calculate Levenshtein distance, WER, CER for strings or lists of strings.
        
        Args:
            references: Reference string(s)
            hypotheses: Hypothesis string(s)
        Returns:
            Tuple of (word_dist, wer, cer)
        """
        # Initialize metrics
        wer_metric = tmt.WordErrorRate()
        word_edit_metric = tmt.EditDistance(reduction='mean')
        cer_metric = tmt.CharErrorRate()

        # Calculate metrics
        word_dist = word_edit_metric(hypotheses, references)
        wer = wer_metric(hypotheses, references)  # torchmetrics returns as decimal
        cer = cer_metric(hypotheses, references)  # torchmetrics returns as decimal

        return {
            'word_dist': word_dist.item(),
            'wer': wer.item() * 100,
            'cer': cer.item() * 100
        }

# -------------------------------------------------------------------------------------------------

class ProgressiveTrainer(ASRTrainer):
    """
    Progressive Trainer class that implements curriculum learning for ASR training.

    This trainer extends ASRTrainer to implement:
    1. Stage-based training with increasing model complexity
    2. Gradual unfreezing of model layers
    3. Dynamic data subsetting
    4. Smooth transition to full model training

    Implementation Tasks:
    - Store original model layers in __init__
    - Configure model for each stage in configure_stage
    - Implement progressive training loop in progressive_train
    - Handle transition to full training in transition_to_full_training
    - Create data subsets in get_subset_dataloader

    Implementation Notes:
    1. For __init__:
        - Store original encoder and decoder layers
        - Initialize stage counter
        
    2. For configure_stage:
        - Update dropout and label smoothing
        - Activate specified encoder and decoder layers
        - Handle layer freezing based on configuration
        - Print detailed configuration information
        
    3. For progressive_train:
        - Configure model for each stage
        - Create appropriate data subset
        - Train using parent class methods
        
    4. For transition_to_full_training:
        - Restore all model layers
        - Reset loss function parameters
        - Unfreeze all parameters
        - Reset best metrics
        
    5. For get_subset_dataloader:
        - Create subset while preserving dataset attributes
        - Maintain collate function and other dataloader settings

    # -------------------------------------------------------------------------------------------------
    ##### Stage Configuration

    Each stage is defined as a dictionary with the following parameters:
    ```python
    {
        'name': str,                        # Name of the training stage
        'epochs': int,                      # Number of epochs to train in this stage
        'encoder_active_layers': List[int], # Which encoder layers to use
        'decoder_active_layers': List[int], # Which decoder layers to use
        'encoder_freeze': List[bool],       # Whether to freeze each encoder layer
        'decoder_freeze': List[bool],       # Whether to freeze each decoder layer
        'dropout': float,                   # Dropout rate for this stage
        'label_smoothing': float,           # Label smoothing value
        'data_subset': float                # Fraction of training data to use (0.0-1.0)
    }
    ```
    #### Example
    It is best understood by an example. Here is a breakdown of the stages defined below for a model with 6 encoder and 6 decoder layers:

    stages = [
                {
                    # `Initial (1 layers)`:
                    # This stage starts with a model with only 1 encoder and 1 decoder layer.
                    # No freezing or regularization is applied.
                    # It uses 20% of the training data.
                    'name': 'Initial (1 Encoder + 1 Decoder layers)',
                    'epochs': 5,
                    'encoder_active_layers': list(range(1)),
                    'decoder_active_layers': list(range(1)),
                    'encoder_freeze': [False],
                    'decoder_freeze': [False],
                    'dropout': 0.0,
                    'label_smoothing': 0.0,
                    'data_subset': 0.2
                },
                {
                    # `2 layers`:
                    # This stage increases the number of layers to 2 for both the encoder and decoder.
                    # The previous layer (encoder layer 1 and decoder layer 1) are frozen.
                    # No regularization is applied.
                    # It uses 20% of the training data.
                    'name': '2 Encoder + 2 Decoder layers',
                    'epochs': 5,
                    'encoder_active_layers': list(range(2)),
                    'decoder_active_layers': list(range(2)),
                    'encoder_freeze': [True, False],
                    'decoder_freeze': [True, False],
                    'dropout': 0.0,
                    'label_smoothing': 0.0,
                    'data_subset': 0.2
                },
                {
                    # `4 layers`:
                    # This stage increases the number of layers to 4 for both the encoder and decoder.
                    # The previous layers (encoder layers 1 and 2 and decoder layers 1 and 2) are frozen.
                    # Dropout is set to 0.05 and label smoothing is set to 0.0.
                    # It uses 20% of the training data.
                    'name': '4 Encoder + 4 Decoder layers',
                    'epochs': 5,
                    'encoder_active_layers': list(range(4)),
                    'decoder_active_layers': list(range(4)),
                    'encoder_freeze': [True, True, False, False],
                    'decoder_freeze': [True, True, False, False],
                    'dropout': 0.05,
                    'label_smoothing': 0.0,
                    'data_subset': 0.2
                },
                {
                    # `All 6 layers`:
                    # This stage uses all 6 encoder and 6 decoder layers.
                    # The 4 previous layers are frozen and the last 2 layers are trained.
                    # Dropout is set to 0.1 and label smoothing is set to 0.0.
                    # It uses 20% of the training data.
                    'name': '6 Encoder + 6 Decoder layers',
                    'epochs': 5,
                    'encoder_active_layers': list(range(6)),
                    'decoder_active_layers': list(range(6)),
                    'encoder_freeze': [True, True, True, True, False, False],
                    'decoder_freeze': [True, True, True, True, False, False],
                    'dropout': 0.1,
                    'label_smoothing': 0.0,
                    'data_subset': 0.2
                },
                {
                    # `Final (with label smoothing)`:
                    # This stage uses all 6 encoder and 6 decoder layers.
                    # All layers are unfrozen and trained.
                    # Dropout is set to 0.1 and label smoothing is set to 0.1.
                    # It uses 20% of the training data.
                    'name': 'Final (with label smoothing)',
                    'epochs': 5,
                    'encoder_active_layers': list(range(6)),
                    'decoder_active_layers': list(range(6)),
                    'encoder_freeze': [False, False, False, False, False, False],
                    'decoder_freeze': [False, False, False, False, False, False],
                    'dropout': 0.1,
                    'label_smoothing': 0.1,
                    'data_subset': 0.2
                }
            ]    

    ##### Important Notes
    - Ensure `encoder_freeze` and `decoder_freeze` lists match the length of their respective `active_layers`
    - `data_subset` should be between 0 and 1
    - Stage transitions are handled automatically by the trainer
    - The same optimizer and scheduler are used for all stages so keep that in mind while setting the learning rates and other parameters
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        self.current_stage = 0
        # Store original layer states
        self.all_encoder_layers = list(self.model.enc_layers)
        self.all_decoder_layers = list(self.model.dec_layers)


    def configure_stage(self, stage_config):
        """Configure model for current training stage"""
        # Create a pretty header
        print("\n" + "="*80)
        print(f"Starting Stage: {stage_config['name']}".center(80))
        print("="*80)
        
        # Print key configuration details
        print(f"\nConfiguration Details:")
        print(f"├── Data Subset: {stage_config['data_subset']*100:.1f}% of training data")
        print(f"├── Training Epochs: {stage_config['epochs']}")
        print(f"├── Dropout: {stage_config['dropout']}")
        print(f"├── Label Smoothing: {stage_config['label_smoothing']}")
        
        # Update dropout and label smoothing
        self.model.dropout.p = stage_config['dropout']
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=stage_config['label_smoothing']
        )
        
        # Get freeze configurations
        encoder_freeze = stage_config.get('encoder_freeze', [])
        decoder_freeze = stage_config.get('decoder_freeze', [])
        
        # Activate and configure encoder layers
        encoder_active_layers = stage_config['encoder_active_layers']
        if encoder_freeze and len(encoder_freeze) != len(encoder_active_layers):
            raise ValueError(f"Encoder freeze list length ({len(encoder_freeze)}) must match number of active encoder layers ({len(encoder_active_layers)})")
        
        # Set the active encoder layers of the model
        self.model.enc_layers = nn.ModuleList([
            self.all_encoder_layers[i] for i in encoder_active_layers
        ])
        self.model.num_encoder_layers = len(encoder_active_layers)
        
        # Activate and configure decoder layers
        decoder_active_layers = stage_config['decoder_active_layers']
        if decoder_freeze and len(decoder_freeze) != len(decoder_active_layers):
            raise ValueError(f"Decoder freeze list length ({len(decoder_freeze)}) must match number of active decoder layers ({len(decoder_active_layers)})")
        
        # Set the active decoder layers of the model
        self.model.dec_layers = nn.ModuleList([
            self.all_decoder_layers[i] for i in decoder_active_layers
        ])
        self.model.num_decoder_layers = len(decoder_active_layers)

        # Handle layer freezing
        frozen_count = 0
        trainable_count = 0
        
        # Configure encoder layers freezing
        print("├── Encoder Layers:")
        for idx, layer in enumerate(self.model.enc_layers):
            should_freeze = encoder_freeze[idx]
            for param in layer.parameters():
                param.requires_grad = not should_freeze
                if should_freeze:
                    frozen_count += param.numel()
                else:
                    trainable_count += param.numel()
            print(f"│   ├── Layer {encoder_active_layers[idx]}: {'Frozen' if should_freeze else 'Trainable'}")
        
        # Configure decoder layers
        print("├── Decoder Layers:")
        for idx, layer in enumerate(self.model.dec_layers):
            should_freeze = decoder_freeze[idx]
            for param in layer.parameters():
                param.requires_grad = not should_freeze
                if should_freeze:
                    frozen_count += param.numel()
                else:
                    trainable_count += param.numel()
            print(f"│   ├── Layer {decoder_active_layers[idx]}: {'Frozen' if should_freeze else 'Trainable'}")
        
        print(f"├── Frozen Parameters: {frozen_count:,}")
        print(f"└── Trainable Parameters: {trainable_count:,}")
    

    def progressive_train(self, train_dataloader, val_dataloader, stages: List[Dict[str, Any]]):
        """
        Progressive training through stages
        Each stage configuration is defined as a dictionary with the following parameters:

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            stages: List of dictionaries containing stage configuration
        """
        # Train through stages
        for stage_idx, stage_config in enumerate(stages):
            self.current_stage = stage_idx
            self.configure_stage(stage_config)
            # Get subset of train_dataloader
            subset_train_dataloader = self.get_subset_dataloader(train_dataloader, stage_config['data_subset'])
            super().train(subset_train_dataloader, val_dataloader, epochs=stage_config['epochs'])

    def transition_to_full_training(self):
        """Transition from progressive training to full training"""
        print("\n=== Transitioning to Full Training ===")
        
        # Restore all layers
        self.model.enc_layers = nn.ModuleList(self.all_encoder_layers)
        self.model.dec_layers = nn.ModuleList(self.all_decoder_layers)
        self.model.num_encoder_layers = len(self.all_encoder_layers)
        self.model.num_decoder_layers = len(self.all_decoder_layers)

        # Restore CrossEntropyLoss
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=self.config['loss']['label_smoothing']
        )
        
        # Unfreeze all parameters
        unfrozen_count = 0
        for param in self.model.parameters():
            param.requires_grad = True
            unfrozen_count += param.numel()
        print(f"├── Total Unfrozen Parameters: {unfrozen_count:,}")
        
        # Reset best metrics for new training phase
        self.best_metric = float('inf')

    
    def train(self, train_dataloader, val_dataloader, epochs):
        """
        Run full training phase.
        It is recommended to set the optimizer and scheduler explicitly before calling this function.
        like this:
        cls.optimizer = create_optimizer(self.model, self.config['optimizer'])
        cls.scheduler = create_scheduler(cls.optimizer, cls.config['scheduler'], train_dataloader)
        cls.progressive_train(train_dataloader, val_dataloader, stages)
        """
        self.transition_to_full_training()
        super().train(train_dataloader, val_dataloader, epochs=epochs)


    def get_subset_dataloader(self, dataloader, subset_fraction):
        """
        Creates a new DataLoader with a subset of the original data while preserving dataset attributes.
        
        Args:
            dataloader: Original DataLoader
            subset_fraction: Float between 0 and 1 indicating what fraction of data to keep
        
        Returns:
            New DataLoader containing only the subset of data
        """
        # Calculate how many samples we want to keep
        dataset = dataloader.dataset
        total_samples = len(dataset)
        subset_size = int(total_samples * subset_fraction)
        
        # Create random indices for the subset
        indices = torch.randperm(total_samples)[:subset_size]
        
        # Create a Subset dataset
        subset_dataset = Subset(dataset, indices)
        
        # Add necessary attributes from original dataset to subset
        subset_dataset.text_max_len = dataset.text_max_len
        subset_dataset.feat_max_len = dataset.feat_max_len
        subset_dataset.get_avg_chars_per_token = dataset.get_avg_chars_per_token
        
        # Create new DataLoader with same configuration as original
        subset_loader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['NUM_WORKERS'],
            collate_fn=dataset.collate_fn,
            pin_memory=True
        )
        
        return subset_loader
