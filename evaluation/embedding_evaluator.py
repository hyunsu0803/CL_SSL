import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def align_loss(x, y, alpha=2):
    """
    Alignment metric - measures how well positive pairs are aligned
    Lower is better (more aligned)
    
    Args:
        x: embeddings of positive samples (B, D)
        y: embeddings of corresponding positive samples (B, D)
        alpha: power parameter (default: 2)
    
    Returns:
        alignment loss value
    """
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    """
    Uniformity metric - measures how uniformly embeddings are distributed
    Lower is better (more uniform)
    
    Args:
        x: embeddings (B, D)
        t: temperature parameter (default: 2)
    
    Returns:
        uniformity loss value
    """
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class EmbeddingEvaluator:
    """
    Evaluates embedding quality using alignment and uniformity metrics
    """
    
    def __init__(self, alpha=2, t=2):
        self.alpha = alpha
        self.t = t
        
    def evaluate_batch(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                      vad_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate embeddings for a single batch
        
        Args:
            embeddings: (B, D) or (B, T, D) tensor of embeddings
            labels: (B,) or (B, T) tensor of labels
            vad_mask: (B, T) optional mask for valid frames
            
        Returns:
            Dictionary with alignment and uniformity scores
        """
        
        # Handle temporal embeddings
        if embeddings.dim() == 3:
            B, T, D = embeddings.shape
            embeddings = embeddings.reshape(-1, D)  # (B*T, D)
            labels = labels.reshape(-1)  # (B*T,)
            if vad_mask is not None:
                vad_mask = vad_mask.reshape(-1)  # (B*T,)
                
        # Filter out invalid frames and ignore labels
        if vad_mask is not None:
            valid_mask = vad_mask.bool()
            embeddings = embeddings[valid_mask]
            labels = labels[valid_mask]
            
        # Filter out ignore labels (like 360, 1000)
        valid_label_mask = (labels != 360) & (labels != 1000)
        embeddings = embeddings[valid_label_mask]
        labels = labels[valid_label_mask]
        
        if len(embeddings) < 2:
            return {'alignment': float('nan'), 'uniformity': float('nan')}
            
        # Normalize embeddings for better metric computation
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute uniformity (using all embeddings)
        uniformity = uniform_loss(embeddings, self.t)
        
        # Compute alignment (using positive pairs with same label)
        alignment = self._compute_alignment(embeddings, labels)
        
        return {
            'alignment': alignment.item() if not torch.isnan(alignment) else float('nan'),
            'uniformity': uniformity.item() if not torch.isnan(uniformity) else float('nan')
        }
    
    def _compute_alignment(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment using positive pairs with same labels
        """
        unique_labels = torch.unique(labels)
        alignments = []
        
        for label in unique_labels:
            mask = labels == label
            label_embeddings = embeddings[mask]
            
            if len(label_embeddings) < 2:
                continue
                
            # Create all pairs within the same label
            n = len(label_embeddings)
            indices = torch.combinations(torch.arange(n), 2)
            
            if len(indices) > 0:
                x = label_embeddings[indices[:, 0]]
                y = label_embeddings[indices[:, 1]]
                alignment = align_loss(x, y, self.alpha)
                alignments.append(alignment)
        
        if len(alignments) == 0:
            return torch.tensor(float('nan'))
            
        return torch.stack(alignments).mean()
    
    def accumulate_embeddings(self, embeddings: torch.Tensor, labels: torch.Tensor,
                            vad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare embeddings and labels for accumulation across batches
        """
        # Handle temporal embeddings
        if embeddings.dim() == 3:
            B, T, D = embeddings.shape
            embeddings = embeddings.reshape(-1, D)  # (B*T, D)
            labels = labels.reshape(-1)  # (B*T,)
            if vad_mask is not None:
                vad_mask = vad_mask.reshape(-1)  # (B*T,)
                
        # Filter out invalid frames and ignore labels
        if vad_mask is not None:
            valid_mask = vad_mask.bool()
            embeddings = embeddings[valid_mask]
            labels = labels[valid_mask]
            
        # Filter out ignore labels (like 360, 1000)
        valid_label_mask = (labels != 360) & (labels != 1000)
        embeddings = embeddings[valid_label_mask]
        labels = labels[valid_label_mask]
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.detach().cpu(), labels.detach().cpu()
    
    def evaluate_accumulated(self, all_embeddings: torch.Tensor, 
                           all_labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate accumulated embeddings from multiple batches
        """
        if len(all_embeddings) < 2:
            return {'alignment': float('nan'), 'uniformity': float('nan')}
            
        all_embeddings = all_embeddings.to(all_labels.device)
        
        # Compute uniformity
        uniformity = uniform_loss(all_embeddings, self.t)
        
        # Compute alignment
        alignment = self._compute_alignment(all_embeddings, all_labels)
        
        return {
            'alignment': alignment.item() if not torch.isnan(alignment) else float('nan'),
            'uniformity': uniformity.item() if not torch.isnan(uniformity) else float('nan')
        }


class EmbeddingAccumulator:
    """
    Helper class to accumulate embeddings across batches for evaluation
    """
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.reset()
        
    def reset(self):
        self.embeddings = []
        self.labels = []
        self.total_samples = 0
        
    def add_batch(self, embeddings: torch.Tensor, labels: torch.Tensor,
                  vad_mask: Optional[torch.Tensor] = None, evaluator: EmbeddingEvaluator = None):
        """
        Add a batch of embeddings and labels
        """
        if evaluator is None:
            evaluator = EmbeddingEvaluator()
            
        emb, lab = evaluator.accumulate_embeddings(embeddings, labels, vad_mask)
        
        if len(emb) > 0:
            self.embeddings.append(emb)
            self.labels.append(lab)
            self.total_samples += len(emb)
            
        # Limit total samples to prevent memory issues
        if self.total_samples > self.max_samples:
            self._subsample()
            
    def _subsample(self):
        """
        Randomly subsample to stay within max_samples limit
        """
        all_emb = torch.cat(self.embeddings, dim=0)
        all_lab = torch.cat(self.labels, dim=0)
        
        indices = torch.randperm(len(all_emb))[:self.max_samples]
        
        self.embeddings = [all_emb[indices]]
        self.labels = [all_lab[indices]]
        self.total_samples = len(indices)
        
    def get_accumulated(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all accumulated embeddings and labels
        """
        if len(self.embeddings) == 0:
            return torch.tensor([]), torch.tensor([])
            
        return torch.cat(self.embeddings, dim=0), torch.cat(self.labels, dim=0)
    
    def evaluate(self, evaluator: EmbeddingEvaluator = None) -> Dict[str, float]:
        """
        Evaluate accumulated embeddings
        """
        if evaluator is None:
            evaluator = EmbeddingEvaluator()
            
        all_emb, all_lab = self.get_accumulated()
        return evaluator.evaluate_accumulated(all_emb, all_lab)
