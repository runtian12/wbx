from torch import nn
import torch


class AnchorClassifierExtendedTarget(nn.Module):
    def __init__(
        self, target_hidden_dim=4096, t_embed=64, s_embed=64, hidden_dim=128, **kwargs
    ):
        super().__init__()
        # self.target_proj = nn.Linear(target_hidden_dim, t_embed)
        self.target_proj = nn.Linear(target_hidden_dim, t_embed)
        small_features_dim = (
            4  # target_entropy, draft_entropy, target_logit, draft_logit
        )
        self.small_features_proj = nn.Linear(
            small_features_dim, s_embed
        )  # 4 -> s_embed

        self.classifier = nn.Sequential(
            nn.Linear(t_embed + s_embed, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        # self.classifier = nn.Linear(target_hidden_dim, 2)

    def forward(
        self,
        target_hidden,
        target_entropy,
        draft_entropy,
        target_logit,
        draft_logit,
        *args,
        **kwargs
    ):

        t_embed = self.target_proj(target_hidden)
        small_features = torch.cat(
            [target_entropy, draft_entropy, target_logit, draft_logit], dim=-1
        )
        s_embed = self.small_features_proj(small_features)
        combined = torch.cat([t_embed, s_embed], dim=-1)
        logits = self.classifier(combined)

        # logits = self.classifier(target_hidden)

        return logits



class AnchorClassifierExtendedTargetV2(nn.Module):
    def __init__(
        self, target_hidden_dim=4096, t_embed=64, s_embed=8, hidden_dim=128, **kwargs
    ):
        super().__init__()
        # self.target_proj = nn.Linear(target_hidden_dim, t_embed)
        self.target_proj = nn.Linear(target_hidden_dim, t_embed)
        small_features_dim = (
            2  # target_entropy, draft_entropy, target_logit, draft_logit
        )
        self.small_features_proj = nn.Linear(
            small_features_dim, s_embed
        )  # 4 -> s_embed

        self.classifier = nn.Sequential(
            nn.Linear(t_embed + s_embed, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        # self.classifier = nn.Linear(target_hidden_dim, 2)

    def forward(
        self,
        target_hidden,
        target_entropy,
        draft_entropy,
        target_logit,
        draft_logit,
        *args,
        **kwargs
    ):

        t_embed = self.target_proj(target_hidden)
        small_features = torch.cat(
            [target_entropy, target_logit], dim=-1
        )
        s_embed = self.small_features_proj(small_features)
        combined = torch.cat([t_embed, s_embed], dim=-1)
        logits = self.classifier(combined)

        # logits = self.classifier(target_hidden)

        return logits



class TorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TorchMLP, self).__init__()
        if len(hidden_dim) == 0:
            self.hidden_layers = []
            self.output_layer = nn.Linear(input_dim, output_dim)
        else:
            n_hidden_layers = len(hidden_dim)
            self.hidden_layers = nn.ModuleList()
            self.hidden_layers.append(nn.Linear(input_dim, hidden_dim[0]))
            for i in range(1, n_hidden_layers):
                self.hidden_layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            self.output_layer = nn.Linear(hidden_dim[-1], output_dim)
        self.relu = nn.ReLU()

    def forward(self,
            target_hidden,
            target_entropy,
            draft_entropy,
            target_logit,
            draft_logit,
            *args,
            **kwargs):
        x = target_hidden
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x