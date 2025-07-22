import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import torch.nn.functional as F


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, q_loss = self.vqgan.encode(x)
        

        
        # Flatten codebook_indices from (batch,16,16) into (batch,256) for transfomer
    
        return codebook_mapping, codebook_indices.view(codebook_mapping.shape[0], -1)

        
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return  lambda gamma : 1-gamma
        elif mode == "cosine":
            return lambda gamma: np.cos(gamma*np.pi/2)
        elif mode == "square":
            return lambda gamma: 1 - gamma**2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):

        """
            MaskGit Stage 2
            
            Using bidirectional transfomer to predict tokens
            first, apply Masked Visual Token Modeling（MVTM）technique
                MVTM : randomly select some token and mask it 
                transfomer need to predict the mask token based on context 
        """
        
        z_indices=None #ground truth
        logits = None  #transformer predict the probability of tokens

        #Ground Truth 
        _ , z_indices = self.encode_to_z(x) 
        

        # mask id : self.mask_token_id
        # Normal distribution for percentage of masking
        mask_ratio = np.random.uniform(0.05, 0.9)
        # True : mask
        mask = torch.bernoulli(mask_ratio* torch.ones(z_indices.shape, device=z_indices.device)).bool()
        
        masked_indices = torch.where(mask, self.mask_token_id, z_indices)        
        logits = self.transformer(masked_indices)
        
        z_indices=z_indices # ground truth
        logits = logits  # Probabilities that predicted by transformer
        
        
        return logits, z_indices

        

    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self,z_indices,mask ,mask_num, ratio):

        # Generate masked token sequence
        # True : mask, False : unmask
        masked_indices = torch.where(mask, self.mask_token_id, z_indices)

        # Predict token probabilities using transformer
        logits = self.transformer(masked_indices)  # Shape: (batch_size, seq_len, num_codebook_vectors)
        
        probs = F.softmax(logits, dim=-1)

        
        # Sample predicted tokens randomly for diversity
        z_indices_predict = torch.distributions.categorical.Categorical(logits=logits).sample()
        while torch.any(z_indices_predict == self.mask_token_id):
            z_indices_predict = torch.distributions.categorical.Categorical(logits=logits).sample()
            
        z_indices_predict = torch.where(mask, z_indices_predict, z_indices)
        
        # Get probabilities of predicted tokens
        z_indices_predict_prob = probs.gather(-1, z_indices_predict.unsqueeze(-1)).squeeze(-1)

        # Calculate number of tokens to unmask based on mask scheduling
        mask_ratio = self.gamma(ratio)
        num_unmask = torch.floor(mask_num * mask_ratio).long()

        # Add Gumbel noise for confidence-based selection
        gumbel_noise = torch.distributions.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices.device)
        temperature = self.choice_temperature * (1 - mask_ratio)
        confidence = z_indices_predict_prob + temperature * gumbel_noise
        
        
        confidence[mask] = -torch.inf
        _, idx = confidence.topk(num_unmask, dim=-1, largest=True)

        # 更新掩碼
        mask_bc = torch.zeros_like(mask, dtype=torch.bool, device=z_indices.device)
        mask_bc.scatter_(dim=1, index=idx, value=True)
        mask_bc = mask_bc | mask  # 保留原始已知 token

        # 更新 token
        z_indices_predict = torch.where(mask_bc, z_indices_predict, masked_indices)
        
        return z_indices_predict, mask_bc
        
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


