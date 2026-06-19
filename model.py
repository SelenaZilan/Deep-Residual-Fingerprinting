import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class PacketEncoder(nn.Module):
    def __init__(self, out_dim=256, embed_dim=64):
        super().__init__()
        self.byte_embedding = nn.Embedding(257, embed_dim, padding_idx=256)
        self.dilations = [1, 2, 4, 8, 16] 
        self.internal_channels = 128
        self.conv_layers = nn.ModuleList()
        curr_channels = embed_dim
        for d in self.dilations:
            self.conv_layers.append(ConvBlock(curr_channels, self.internal_channels, kernel_size=5, dilation=d))
            curr_channels = self.internal_channels
        self.proto_embedding = nn.Embedding(256, 16)
        self.fc = nn.Sequential(nn.Linear(self.internal_channels * 2 + 16 + 1, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, out_dim))
        self.output_norm = nn.LayerNorm(out_dim)

    def forward(self, bytes_tsr, protos_tsr, lens_tsr):
        byte_mask = (bytes_tsr != 256).float().unsqueeze(1)
        x = self.byte_embedding(bytes_tsr).transpose(1, 2)
        for conv in self.conv_layers:
            identity = x
            x = conv(x)
            if identity.shape == x.shape: x = x + identity 
        x_avg = (x * byte_mask).sum(dim=2) / torch.clamp(byte_mask.sum(dim=2), min=1.0)
        x_max = torch.max(x.masked_fill(byte_mask == 0, -1e9), dim=2)[0]
        fused = torch.cat([x_max, x_avg, self.proto_embedding(protos_tsr), lens_tsr.unsqueeze(1)], dim=1)
        return self.output_norm(self.fc(fused))
    
class AutoregressiveFlowTransformer(nn.Module):
    def __init__(self, packet_dim=256, iat_dim=32, len_dim=32, nhead=8, num_layers=4, steps=3):
        super().__init__()
        self.steps = steps
        self.packet_dim = packet_dim
        self.iat_dim = iat_dim
        self.len_dim = len_dim
        
        # 1. Compute the total dimension: 256 (Packet+Pos+Dir) + 32 (IAT) + 32 (Len) = 320.
        self.total_dim = packet_dim + iat_dim + len_dim

        # Core encoder for raw packet bytes.
        self.packet_encoder = PacketEncoder(out_dim=packet_dim)
        
        # Direction and position embeddings are added to packet features with the same packet_dim.
        self.dir_embedding = nn.Embedding(2, packet_dim)
        self.pos_embedding = nn.Embedding(50, packet_dim) # Assumes MAX_PACKETS=50.
        
        # Projection layers for IAT and packet length before concatenation.
        self.iat_proj = nn.Sequential(
            nn.Linear(1, 64), 
            nn.GELU(), 
            nn.Linear(64, iat_dim)
        )
        self.len_proj = nn.Sequential(
            nn.Linear(1, 64), 
            nn.GELU(), 
            nn.Linear(64, len_dim)
        )

        self.input_norm = nn.LayerNorm(self.total_dim) 
        self.dropout = nn.Dropout(0.2)

        # Transformer d_model must match total_dim.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.total_dim, 
            nhead=nhead, 
            dim_feedforward=1024, # Use a larger FFN dimension.
            dropout=0.2, 
            batch_first=True, 
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction heads use the updated total_dim input.
        self.predictor_shared = nn.Sequential(
            nn.Linear(self.total_dim, 512), 
            nn.GELU(), 
            nn.Dropout(0.2)
        )
        self.log_vars = nn.Parameter(torch.zeros(3)) 
        self.pred_feat = nn.Linear(512, packet_dim * steps)
        self.pred_iat = nn.Linear(512, steps)
        self.pred_len = nn.Linear(512, steps)

    def generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def forward(self, batch):
        b_bytes = batch['byte_vectors'] 
        B, Seq, Bytes = b_bytes.shape
        
        # --- Step 1: Base packet features with position and direction. ---
        packet_feats = self.packet_encoder(
            b_bytes.view(-1, Bytes), 
            batch['protocols'].view(-1), 
            batch['orig_lens'].view(-1)
        ).view(B, Seq, -1)
        
        positions = torch.arange(0, Seq, device=b_bytes.device).unsqueeze(0).expand(B, Seq)
        # Add position and direction embeddings to packet semantic features.
        core_x = packet_feats + self.dir_embedding(batch['directions']) + self.pos_embedding(positions)
        
        # --- Step 2: Project IAT and packet length independently. ---
        iat_x = self.iat_proj(batch['iats'].unsqueeze(-1))     # [B, Seq, iat_dim]
        len_x = self.len_proj(batch['orig_lens'].unsqueeze(-1).float()) # [B, Seq, len_dim]
        
        # --- Step 3: Concatenate the three feature streams. ---
        # Output shape: [B, Seq, total_dim].
        x = torch.cat([core_x, iat_x, len_x], dim=-1)
        
        x = self.dropout(self.input_norm(x))

        # --- Step 4: Transformer encoding. ---
        float_pad_mask = batch['attention_mask'].float().masked_fill(batch['attention_mask']==True, float('-inf')).masked_fill(batch['attention_mask']==False, 0.0)
        out = self.transformer(x, mask=self.generate_causal_mask(Seq).to(x.device), src_key_padding_mask=float_pad_mask)
        
        # --- Step 5: Prediction heads. ---
        shared_out = self.predictor_shared(out)
        
        return (
            self.pred_feat(shared_out).view(B, Seq, self.steps, -1), 
            self.pred_iat(shared_out).view(B, Seq, self.steps), 
            self.pred_len(shared_out).view(B, Seq, self.steps), 
            packet_feats.detach()
        )