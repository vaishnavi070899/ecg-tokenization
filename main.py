import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data.load_data import PTBXLDataset
from models.vqvae import VQVAE

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--resume",        action="store_true",
                    help="Load vqvae_best.pt and continue training")
parser.add_argument("--start-epoch",   type=int,   default=1,
                    help="Epoch to start from (use with --resume)")
parser.add_argument("--best-val-loss", type=float, default=float("inf"),
                    help="Best val loss so far (use with --resume)")
args = parser.parse_args()

# ── Data ───────────────────────────────────────────────────────────────────────
train_dataset = PTBXLDataset(folds=list(range(1, 10)), n_records=config.N_RECORDS)
val_dataset   = PTBXLDataset(folds=[10],               n_records=config.N_RECORDS)

train_loader  = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False)

print(f"EMA val: {config.EMA_DECAY}  |  Commitment cost β: {config.COMMITMENT_COST}")
print(f"Train: {len(train_dataset)} records  |  Val: {len(val_dataset)} records")

# ── Model ──────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = VQVAE(input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM,
               num_embeddings=config.NUM_EMBEDDINGS, decay=config.EMA_DECAY,
               buffer_size=config.BUFFER_SIZE,
               num_rvq_stages=config.NUM_RVQ_STAGES).to(device)

if args.resume:
    model.load_state_dict(torch.load("vqvae_best.pt", map_location=device))
    print(f"Resumed from vqvae_best.pt  |  start_epoch={args.start_epoch}  |  best_val={args.best_val_loss:.6f}")

optimizer     = torch.optim.Adam(model.parameters(), lr=config.LR)
recon_loss_fn = nn.MSELoss()

best_val_loss = args.best_val_loss

# ── Training loop ──────────────────────────────────────────────────────────────
for epoch in range(args.start_epoch, config.EPOCHS + 1):

    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    total_recon = total_vq = total_perp = 0.0
    total_residuals = None   # will become a list of per-stage accumulators

    for x in train_loader:
        x = x.to(device)
        x_recon, vq_loss, perplexity, residual_norms = model(x)

        recon_loss = recon_loss_fn(x_recon, x)
        loss       = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_recon += recon_loss.item()
        total_vq    += vq_loss.item()
        total_perp  += perplexity.item()

        if total_residuals is None:
            total_residuals = [0.0] * len(residual_norms)
        for i, r in enumerate(residual_norms):
            total_residuals[i] += r

    n = len(train_loader)
    res_str = "  residual=[" + "|".join(f"{v/n:.6f}" for v in total_residuals) + "]"
    print(f"Epoch {epoch:3d}/{config.EPOCHS}  "
          f"train_recon={total_recon/n:.6f}  "
          f"vq={total_vq/n:.6f}  "
          f"perplexity={total_perp/n:.1f}/{config.NUM_EMBEDDINGS}"
          f"{res_str}",
          end="")

    # ── Validate ───────────────────────────────────────────────────────────────
    model.eval()
    val_recon = 0.0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            x_recon, _, _, _ = model(x)
            val_recon += recon_loss_fn(x_recon, x).item()

    avg_val = val_recon / len(val_loader)
    print(f"  val_recon={avg_val:.6f}", end="")

    # ── Save best checkpoint ───────────────────────────────────────────────────
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "vqvae_best.pt")
        print("  * saved", end="")

    print()
    # if need to resume: python main.py --resume --start-epoch 6 --best-val-loss 0.045544   #from the last saved checkpoint

print(f"\nBest val recon loss: {best_val_loss:.6f}  ->  vqvae_best.pt")
