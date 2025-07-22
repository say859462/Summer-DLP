import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.dim = dim
        self.attn_drop =attn_drop
        
        self.head_dim = dim // num_heads
        
        # Weight matrix of Q,K,V
        self.W_Q = nn.Linear(dim,dim)
        self.W_K = nn.Linear(dim,dim)
        self.W_V = nn.Linear(dim,dim)
        
        self.dropout = nn.Dropout(attn_drop)
        
        self.output = nn.Linear(dim,dim)
    def forward(self, x):
        
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        
        """
            There are num_image_tokens tokens, each tokens are embedded as dim length long
            
            fianl output dimension is 768
            num_heads = 16 , d_k , d_v for one head will be 768//16, for each output of head,we concate them into 768 dimension outupt
            
            I : dim x num_image_tokens
            W(Q,K,V) dim x dim
            
            After linear tranforming , we divide it into num_heads heads, divided vector size (batch_size,num_heads,num_image_tokens,head_dim)
        """
        
        batch_size,num_image_tokens,dim = x.shape
        
        Q = self._split_head(self.W_Q(x))
        K = self._split_head(self.W_K(x))
        V = self._split_head(self.W_V(x))
        
        scale = math.sqrt(self.head_dim)        # matrix multiplication
        attention_scores = (Q @ K.transpose(-2,-1)) / scale
        
        attention_prob = attention_scores.softmax(dim=-1)
        attention_drop = self.dropout(attention_prob)
        
        # Concate
        attention_weight = (attention_drop @ V)
        
        output = attention_weight.permute(0,2,1,3).reshape(batch_size,num_image_tokens,dim)
        
        output =self.output(output)
        
        return output
        
    def _split_head(self,x):
        batch_size,num_image_tokens,dim = x.shape
        # Input size (batch_size, num_image_tokens, dim)
        # first divide dim into (num_head,head_dim) => (batch_size,num_image_tokens,num_head,head_dim)
        
        # Then transform it into (batch_size,num_head,num_image_tokens,head_dim)
        return x.view(batch_size,num_image_tokens,self.num_heads,self.head_dim).permute(0,2,1,3)

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    