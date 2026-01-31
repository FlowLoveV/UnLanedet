import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    # Attempt to import mamba_ssm if available in the environment
    from mamba_ssm import Mamba
except ImportError:
    # Fallback or placeholder if mamba_ssm is not installed
    # For the purpose of this code generation, we define a basic structure
    # that would wrap the official implementation or a custom one.
    Mamba = None

class SS2D(nn.Module):
    """
    2D Selective Scan Module (SS2D) for Visual Mamba.
    Scans the feature map in 4 directions to capture global context with linear complexity.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Convolution (Depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Activation
        self.act = nn.SiLU()

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Selective Scan parameters projection
        # x_proj takes input and projects to (delta, B, C)
        # dt_rank = math.ceil(d_model / 16)
        self.dt_rank = (self.d_inner + 16 - 1) // 16
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        
        # Delta projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize special parameters A and D
        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        L = H * W
        
        x_flat = x.view(B, L, C) # (B, L, C)
        
        # 1. Project to higher dimension
        xz = self.in_proj(x_flat) # (B, L, 2*d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1) # (B, L, d_inner) each

        # 2. Conv1d processing
        x_branch = x_branch.transpose(1, 2) # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :L] # Causal conv
        x_branch = x_branch.transpose(1, 2) # (B, L, d_inner)
        x_branch = self.act(x_branch)
        
        # 3. SS2D Core: 4-Direction Scanning
        # For simplicity in this implementation, we simulate the 4-direction scan
        # by flipping the sequence. In a full implementation, this would be 
        # optimized with a custom CUDA kernel or using mamba_ssm.
        
        # Direction 1: Top-left -> Bottom-right (Original)
        y1 = self.selective_scan(x_branch)
        
        # Direction 2: Bottom-right -> Top-left (Flip H*W)
        x_flip = torch.flip(x_branch, dims=[1])
        y2 = torch.flip(self.selective_scan(x_flip), dims=[1])
        
        # Direction 3: Top-right -> Bottom-left (Transpose -> Scan -> Transpose back)
        # Reshape to 2D to transpose
        x_2d = x_branch.view(B, H, W, -1).transpose(1, 2).reshape(B, L, -1)
        y3_trans = self.selective_scan(x_2d)
        y3 = y3_trans.view(B, W, H, -1).transpose(1, 2).reshape(B, L, -1)
        
        # Direction 4: Bottom-left -> Top-right (Transpose -> Flip -> Scan -> ...)
        x_2d_flip = torch.flip(x_2d, dims=[1])
        y4_trans_flip = self.selective_scan(x_2d_flip)
        y4 = torch.flip(y4_trans_flip, dims=[1]).view(B, W, H, -1).transpose(1, 2).reshape(B, L, -1)
        
        # Sum all directions
        y = y1 + y2 + y3 + y4
        
        # 4. Gating
        y = y * self.act(z_branch)
        
        # 5. Output projection
        out = self.out_proj(y)
        
        return out.view(B, H, W, C)

    def selective_scan(self, u):
        """
        Simplified selective scan implementation (S6).
        u: (B, L, d_inner)
        """
        B, L, D = u.shape
        
        # Project to parameters
        delta_b_c = self.x_proj(u) # (B, L, dt_rank + 2*d_state)
        delta, B_ssm, C_ssm = torch.split(delta_b_c, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta)) # (B, L, D)
        
        # Discretization (Simplified)
        # In a real implementation, this runs in a recurrent loop or parallel scan
        # Here we just show the tensor shapes and logic flow
        
        # This part requires a parallel scan implementation (like heinsen_sequence)
        # or a sequential loop. For efficiency in python, a loop is slow.
        # We assume a placeholder function for the actual scan.
        return u # Placeholder: returns input as identity for structure demonstration


class TSSM(nn.Module):
    """
    Temporal Selective Scan Module (TSSM)
    Aggregates spatiotemporal features using Visual Mamba.
    """
    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.ss2d = SS2D(d_model=in_channels, d_state=16, d_conv=4, expand=2)
        
        # Content-aware temporal gating
        self.temporal_gate = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.SiLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x, prev_state=None):
        """
        x: (B, C, H, W) - Current frame features
        prev_state: (B, C, H, W) - Previous aggregated state (optional)
        """
        B, C, H, W = x.shape
        
        # Channel last for SS2D
        x_in = x.permute(0, 2, 3, 1) # (B, H, W, C)
        x_norm = self.norm(x_in)
        
        # Spatial Modeling with SS2D
        x_spatial = self.ss2d(x_norm) # (B, H, W, C)
        x_spatial = x_spatial.permute(0, 3, 1, 2) # (B, C, H, W)
        
        # Temporal Aggregation
        if prev_state is not None:
            # Calculate temporal gate based on current content
            # Global average pooling for gate context
            ctx = torch.mean(x, dim=[2, 3]) # (B, C)
            gate = self.temporal_gate(ctx).view(B, C, 1, 1)
            
            # Weighted fusion
            out = gate * x_spatial + (1 - gate) * prev_state
        else:
            out = x_spatial
            
        return out

class TemporalLaneNet(nn.Module):
    """
    Example wrapper class showing how TSSM is integrated into LLANet
    """
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.tssm = TSSM(in_channels=neck.out_channels) # Assuming neck output matches
        self.head = head
        
        # State buffer for inference
        self.register_buffer('prev_state', None)
        
    def forward(self, x):
        # x: (B, T, 3, H, W) or (B, 3, H, W)
        if x.dim() == 5:
            # Training with sequence
            B, T, C, H, W = x.shape
            outputs = []
            prev_state = None
            
            for t in range(T):
                frame = x[:, t]
                feat = self.backbone(frame)
                neck_feat = self.neck(feat)
                
                # Apply TSSM
                curr_state = self.tssm(neck_feat[-1], prev_state) # Apply to last level or all
                prev_state = curr_state
                
                # Head prediction
                out = self.head([curr_state]) # Simplified
                outputs.append(out)
            return outputs
        else:
            # Inference (Single frame streaming)
            feat = self.backbone(x)
            neck_feat = self.neck(feat)
            
            curr_state = self.tssm(neck_feat[-1], self.prev_state)
            self.prev_state = curr_state.detach() # Update state
            
            return self.head([curr_state])

    def reset_state(self):
        self.prev_state = None
