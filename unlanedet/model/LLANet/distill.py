import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureLoss(nn.Module):
    """
    Feature Distillation Loss using MSE.
    Includes an adapter (1x1 conv) to align student channels to teacher channels.
    """
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.adapter = nn.Conv2d(student_channels, teacher_channels, kernel_size=1)
        self.criterion = nn.MSELoss()

    def forward(self, student_feat, teacher_feat):
        """
        student_feat: (B, Cs, H, W)
        teacher_feat: (B, Ct, H', W')
        """
        # Align channels
        student_adapted = self.adapter(student_feat)
        
        # Align spatial resolution (Upsample student to match teacher if needed)
        if student_adapted.shape[-2:] != teacher_feat.shape[-2:]:
            student_adapted = F.interpolate(
                student_adapted, 
                size=teacher_feat.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            
        return self.criterion(student_adapted, teacher_feat)

class LogitsLoss(nn.Module):
    """
    Logits Distillation Loss using KL Divergence.
    Softens distribution with temperature T.
    """
    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits):
        """
        student_logits: (B, NumClasses, ...)
        teacher_logits: (B, NumClasses, ...)
        """
        # Softmax with temperature
        p_s = F.log_softmax(student_logits / self.T, dim=1)
        p_t = F.softmax(teacher_logits / self.T, dim=1)
        
        # KL Divergence * T^2
        loss = self.criterion(p_s, p_t) * (self.T ** 2)
        return loss

class DistillationManager(nn.Module):
    """
    Manages the distillation process between Teacher and Student.
    """
    def __init__(self, student_model, teacher_model, feature_pairs, alpha_feat=1.0, alpha_logits=1.0):
        """
        feature_pairs: List of dicts [{'student_idx': 0, 'teacher_idx': 0, 's_dim': 64, 't_dim': 256}, ...]
        """
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.alpha_feat = alpha_feat
        self.alpha_logits = alpha_logits
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Initialize feature losses
        self.feat_losses = nn.ModuleList()
        self.feature_pairs = feature_pairs
        
        for pair in feature_pairs:
            self.feat_losses.append(
                FeatureLoss(pair['s_dim'], pair['t_dim'])
            )
            
        self.logits_loss = LogitsLoss()

    def forward(self, x):
        # Teacher inference (no grad)
        with torch.no_grad():
            t_feats = self.teacher.extract_features(x) # Assuming this method exists
            t_logits = self.teacher(x)['logits'] # Assuming dict output
            
        # Student inference
        s_feats = self.student.extract_features(x)
        s_out = self.student(x)
        s_logits = s_out['logits']
        
        # Calculate losses
        total_distill_loss = 0.0
        
        # 1. Feature Loss
        feat_loss_val = 0.0
        for i, pair in enumerate(self.feature_pairs):
            s_f = s_feats[pair['student_idx']]
            t_f = t_feats[pair['teacher_idx']]
            loss = self.feat_losses[i](s_f, t_f)
            feat_loss_val += loss
        
        total_distill_loss += self.alpha_feat * feat_loss_val
        
        # 2. Logits Loss
        logits_loss_val = self.logits_loss(s_logits, t_logits)
        total_distill_loss += self.alpha_logits * logits_loss_val
        
        # Return student output and distillation loss
        return s_out, total_distill_loss
