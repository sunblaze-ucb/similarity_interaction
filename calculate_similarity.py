import metrics
from interaction.models.model import Model
import json, torch, os
from tqdm import trange
import torch.nn.functional as F
from pathlib import Path

def whiten(feats: torch.Tensor):
    """
    feats: (n_samples, d_features)  float32 / float64
    returns: whitened tensor with per-feature mean = 0, std = 1
    """
    mean = feats.mean(dim=0, keepdim=True)
    std  = feats.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    return (feats - mean) / std

def calculate_similarity(dataset: str, model_id1: str,model_id2: str, method='last', metric='cka', metric_parameter=None, conversation_history1=[], conversation_history2=[], figure_name = 'result', batch_size=10, output_dir='results', model_tensor_filename1 = 'model_tensor1',model_tensor_filename2 = 'model_tensor2', refresh=False):
    """
    method: pooling method (last or avg)
    metric: representation similarity score type
    returns: layer-wise similarity scores
    """
    SUPPORTED_METRICS = ["cka"]

    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unrecognized metric: {metric}")

    metric_fn = getattr(metrics, metric)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float32 
    torch.set_grad_enabled(False)

    if refresh==False and (f'{model_tensor_filename1}.pt' in os.listdir(output_dir)):
        Xf = torch.load(Path(output_dir) / f'{model_tensor_filename1}.pt', map_location=device).to(dtype)
        L1 = Xf.shape[0]
    else:
        probe=[]
        
        with open(f"data/{dataset}.jsonl", "r", encoding="utf-8") as f:
            for _, line in enumerate(f):
                obj = json.loads(line)
                probe.append(obj['prompt'])

        model1=Model(model_id1)

        X = []
        for i in trange(0, len(probe), batch_size):
            if len(conversation_history1)>0 and conversation_history1[-1]["role"]=="user":
                messages1=[conversation_history1[:-1]+[{"role": "user", "content": conversation_history1[-1]['content']+'\n\n'+p}] for p in probe[i:i+batch_size]]
            else:
                messages1=[conversation_history1+[{"role": "user", "content": p}] for p in probe[i:i+batch_size]]
            

            if method=='avg':
                tokens1,v1=model1.get_hidden_states(messages1)
                v1 = torch.stack(v1).permute(1, 0, 2, 3)
                mask1 = tokens1["attention_mask"].unsqueeze(-1).unsqueeze(1) 
                
                v1 = (v1 * mask1).sum(2) / mask1.sum(2)
                
            elif method=='last':
                _,v1=model1.get_hidden_states_last_only(messages1)

                v1 = v1.permute(1, 0, 2) 
            else:
                raise NotImplementedError(f"unknown pooling {method}")
            X.append(v1)

        X=torch.cat(X,0)

        L1 = X.shape[1]

        Xf = X.permute(1, 0, 2)
        del X, model1
        Xf = Xf.to(device, dtype=dtype)  

        # Create the folder if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the tensors
        torch.save(Xf, os.path.join(output_dir, f'{model_tensor_filename1}.pt'))

    if refresh==False and (f'{model_tensor_filename2}.pt' in os.listdir(output_dir)):
        Yf = torch.load(Path(output_dir) / f'{model_tensor_filename2}.pt', map_location=device).to(dtype)
        L2 = Yf.shape[0]
    else:
        probe=[]
        
        with open(f"data/{dataset}.jsonl", "r", encoding="utf-8") as f:
            for _, line in enumerate(f):
                obj = json.loads(line)
                probe.append(obj['prompt'])

        model2=Model(model_id2)

        Y = []
        for i in trange(0, len(probe), batch_size):
            if len(conversation_history2)>0 and conversation_history2[-1]["role"]=="user":
                messages2=[conversation_history2[:-1]+[{"role": "user", "content": conversation_history2[-1]['content']+'\n\n'+p}] for p in probe[i:i+batch_size]]
            else:
                messages2=[conversation_history2+[{"role": "user", "content": p}] for p in probe[i:i+batch_size]]
            
            if method=='avg':
                tokens2,v2=model2.get_hidden_states(messages2)
                v2 = torch.stack(v2).permute(1, 0, 2, 3)
                mask2 = tokens2["attention_mask"].unsqueeze(-1).unsqueeze(1)

                v2 = (v2 * mask2).sum(2) / mask2.sum(2)
        
            elif method=='last':
                _,v2=model2.get_hidden_states_last_only(messages2)

                v2 = v2.permute(1, 0, 2) 
            else:
                raise NotImplementedError(f"unknown pooling {method}")
            Y.append(v2)

        Y=torch.cat(Y,0)

        L2 = Y.shape[1]

        Yf = Y.permute(1, 0, 2)          # (L2, N, H2)
        del Y, model2
        Yf = Yf.to(device, dtype=dtype)
        
        # Create the folder if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the tensors
        torch.save(Yf, os.path.join(output_dir, f'{model_tensor_filename2}.pt'))

    cka_mat = torch.empty(L1, L2, dtype=dtype, device=device) 

    for i in range(L1):
        for j in range(L2):
            if metric_parameter==None:
                cka_mat[i, j] = metric_fn(Xf[i], Yf[j])
            else:
                cka_mat[i, j] = metric_fn(Xf[i], Yf[j], **metric_parameter)

    cka_mat=cka_mat.cpu().numpy()

    import matplotlib.pyplot as plt
    plt.imshow(cka_mat, vmin=0, vmax=1, aspect="auto", origin="lower")

    model_id1=model_id1[model_id1.index('/')+1:]
    model_id2=model_id2[model_id2.index('/')+1:]
    plt.xlabel(model_id2)
    plt.ylabel(model_id1)
    plt.colorbar(label=metric)

    figures_path = Path("figures")
    figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_path / f"{figure_name}.png", dpi=150)
    plt.clf()

    del Xf, Yf
    torch.cuda.empty_cache()

    return cka_mat