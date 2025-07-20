#!/usr/bin/env python3
"""
CLIP Encoder Module

This module provides CLIP-based image encoding functionality for converting
images into high-dimensional vector representations.
"""

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
import numpy as np
from typing import Union, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """
    A class for encoding images using CLIP models.
    """
    
    def __init__(self, 
                 model_name: str = "ViT-B/32", 
                 pretrained: str = "openai",
                 device: str = "auto"):
        """
        Initialize the CLIP encoder.
        
        Args:
            model_name (str): CLIP model architecture name
            pretrained (str): Pretrained weights to use
            device (str): Device to run the model on ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = self._get_device(device)
        
        # Load model and preprocessing
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._load_model()
        
        logger.info(f"CLIP encoder initialized with {model_name} on {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the appropriate device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self):
        """Load the CLIP model and preprocessing pipeline."""
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            self.model.eval()
            
            # Get model info
            if hasattr(self.model.visual, 'output_dim'):
                self.embedding_dim = self.model.visual.output_dim
            else:
                # Fallback: encode a dummy image to get dimension
                dummy_img = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    features = self.model.encode_image(dummy_img)
                    self.embedding_dim = features.shape[-1]
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def encode_image(self, 
                    image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
                    normalize: bool = True) -> np.ndarray:
        """
        Encode a single image into a vector representation.
        
        Args:
            image: Input image (file path, PIL Image, numpy array, or torch tensor)
            normalize: Whether to normalize the output vector
            
        Returns:
            np.ndarray: Image embedding vector
        """
        # Preprocess image
        image_tensor = self._preprocess_image(image)
        
        # Encode image
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            
            if normalize:
                image_features = F.normalize(image_features, dim=-1)
        
        return image_features.cpu().numpy()[0]  # Return first (and only) embedding
    
    def encode_images_batch(self,
                           images: List[Union[str, Path, Image.Image, np.ndarray]],
                           batch_size: int = 32,
                           normalize: bool = True) -> List[np.ndarray]:
        """
        Encode multiple images in batches.
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            normalize: Whether to normalize the output vectors
            
        Returns:
            List[np.ndarray]: List of image embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_embeddings = self._encode_batch(batch, normalize)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _encode_batch(self, 
                     images: List[Union[str, Path, Image.Image, np.ndarray]],
                     normalize: bool = True) -> List[np.ndarray]:
        """Encode a batch of images."""
        try:
            # Preprocess all images in batch
            batch_tensors = []
            valid_indices = []
            
            for idx, image in enumerate(images):
                try:
                    tensor = self._preprocess_image(image)
                    batch_tensors.append(tensor)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Failed to preprocess image {idx}: {e}")
                    continue
            
            if not batch_tensors:
                return []
            
            # Stack tensors
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Encode batch
            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                
                if normalize:
                    features = F.normalize(features, dim=-1)
            
            return [feat.cpu().numpy() for feat in features]
            
        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            return []
    
    def _preprocess_image(self, 
                         image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess an image for CLIP encoding.
        
        Args:
            image: Input image in various formats
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, (str, Path)):
            # Load from file path
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image, 'RGB')
            else:
                raise ValueError(f"Unsupported numpy array shape: {image.shape}")
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        elif isinstance(image, torch.Tensor):
            # Assume it's already preprocessed
            return image.unsqueeze(0).to(self.device)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply CLIP preprocessing
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def encode_text(self, 
                   texts: Union[str, List[str]], 
                   normalize: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode text(s) into vector representation(s).
        
        Args:
            texts: Input text or list of texts
            normalize: Whether to normalize the output vectors
            
        Returns:
            Vector representation(s) of the text(s)
        """
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
        
        # Tokenize texts
        text_tokens = self.tokenizer(texts).to(self.device)
        
        # Encode texts
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            
            if normalize:
                text_features = F.normalize(text_features, dim=-1)
        
        embeddings = [feat.cpu().numpy() for feat in text_features]
        
        return embeddings[0] if return_single else embeddings
    
    def compute_similarity(self, 
                          image_embedding: np.ndarray, 
                          text_embedding: np.ndarray) -> float:
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_embedding: Image embedding vector
            text_embedding: Text embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        # Ensure embeddings are normalized
        image_norm = image_embedding / np.linalg.norm(image_embedding)
        text_norm = text_embedding / np.linalg.norm(text_embedding)
        
        return float(np.dot(image_norm, text_norm))
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'pretrained': self.pretrained,
            'device': str(self.device),
            'embedding_dim': self.embedding_dim,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }