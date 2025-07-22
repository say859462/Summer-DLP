import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS, learning_rate, scheduler_type):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        
        self.optim,self.scheduler = self.configure_optimizers(learning_rate,scheduler_type)
        
        self.prepare_training(learning_rate,scheduler_type)
        
        self.train_losses = []  # record each epoch training loss
        self.val_losses = []    # record each epoch validate loss
        self.args = args        
        
        self.best_train_loss = np.inf
        self.best_val_loss =np.inf
        
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
    @staticmethod
    def prepare_training(learning_rate, scheduler_type):
        checkpoint_dir = os.path.join("transformer_checkpoints", f"lr_{learning_rate}_sched_{scheduler_type}")
        # os.makedirs("transformer_checkpoints", exist_ok=True)
        os.makedirs(checkpoint_dir,exist_ok=True)
        
    def train_one_epoch(self,train_loader,epoch,args):
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch}|{args.epochs}]", total=num_batches)
        for step,images in enumerate(progress_bar):

            images = images.to(args.device)
            
            # Forward ,get logits and true tokens
            logits, z_indices = self.model(images)  # logits: (batch_size, 256, 1024), z_indices: (batch_size, 256)
            
            # logits: (batch_size * 256,1024) , z_indice : (batch_size*256,1)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))           
            
            loss.backward()
            if (step + 1) % args.accum_grad == 0 :
                self.optim.step()
                self.optim.zero_grad()
            
            total_loss += loss.item() 
            progress_bar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}"})
        
        avg_loss = total_loss / num_batches
        
        self.train_losses.append(avg_loss) 
        
        if self.scheduler_type == "ReduceLROnPlateau":
            self.scheduler.step(avg_loss)  # Step based on training loss
        else:
            self.scheduler.step()  # Step for LinearLR + CosineAnnealing
            
        return avg_loss
            

            
            
            

    def eval_one_epoch(self,val_loader,args):
        tqdm.write("Validating.......")
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for images in val_loader:
                images = images.to(args.device)
                
                # Forward pass
                logits, z_indices = self.model(images)
                
               
                
                # Compute loss for masked tokens
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))   
                total_loss += loss.item()
            
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def configure_optimizers(self, learning_rate = 1e-4, scheduler_type = None):
        
        optimizer = torch.optim.Adam(self.model.parameters(),lr = args.learning_rate)
        
        # learning rate start with 0.1% of learning_rate
        if scheduler_type == "LinearLR_CosineAnnealing":
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=500)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 500)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[500])
        elif scheduler_type == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        else:
            scheduler = None

        return optimizer, scheduler

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label=f'Training Loss (min: {min(self.train_losses):.4f})')
        plt.plot(epochs, self.val_losses, label=f'Validation Loss (min: {min(self.val_losses):.4f})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss (lr={self.learning_rate}, sched={self.scheduler_type})')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join("loss_plots", f"loss_lr_{self.learning_rate}_sched_{self.scheduler_type}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Loss plot saved to {save_path}")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="lab3_dataset/train", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="lab3_dataset/val", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    # train_transformer = TrainTransformer(args, MaskGit_CONFIGS)
    
    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)

    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
    best_train = np.inf
    best_val = np.inf
    
    # Define test configurations
    learning_rates = [1e-4, 1e-3]
    scheduler_types = ["LinearLR_CosineAnnealing", "ReduceLROnPlateau"]
    

    for lr in learning_rates:
        for sched in scheduler_types:
            print(f"Starting training with learning_rate={lr}, scheduler={sched}")
            train_transformer = TrainTransformer(args, MaskGit_CONFIGS, lr, sched)
            
            for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
                train_loss = train_transformer.train_one_epoch(train_loader, epoch, args)
                val_loss = train_transformer.eval_one_epoch(val_loader, args)
                
                # Save best model
                if val_loss < train_transformer.best_val_loss:
                    train_transformer.best_val_loss = val_loss
                    checkpoint_dir = os.path.join("transformer_checkpoints", f"lr_{lr}_sched_{sched}")
                    torch.save(train_transformer.model.transformer.state_dict(), 
                               os.path.join(checkpoint_dir, "best_val.pth"))
                

            
            # Plot and save loss curve
            train_transformer.plot_loss()