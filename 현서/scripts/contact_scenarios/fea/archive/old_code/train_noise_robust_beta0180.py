"""노이즈 강건성 확인: 캐시된 15만개(segment_bfield_singleprobe_beta0180_4seg.npz)를
재사용해서, 학습 중에 매 배치마다 무작위 크기(0~NOISE_MAX uT)의 가우시안 노이즈를 입력에
섞어 학습(노이즈 augmentation). 노이즈 없이 학습한 원본 모델은 노이즈 0.01uT부터 급격히
무너졌음(R^2 0.94->0.69, 0.05uT에서 음수) - 이 신호(중앙값 0.0012uT) 자체가 극도로 미세해서
현실적인 센서 노이즈 수준에서 생존 가능한지 재학습으로 테스트.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

FEA_DATA_DIR = os.path.join("..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join("..", "..", "..", "models")

NOISE_MAX_UT = float(os.environ.get("NOISE_MAX_UT", 0.02))  # 학습 시 0~이 값 사이 무작위 노이즈
N_EPOCHS = int(os.environ.get("N_EPOCHS", 60))


class SingleProbeClassifier(nn.Module):
    def __init__(self, n_probes=1, n_classes=4, n_force=2, n_config=2):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(64 * n_probes, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)
        self.s_head = nn.Linear(128, 1)
        self.config_head = nn.Linear(128, n_config)

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.config_head(h)


def r2(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


if __name__ == "__main__":
    data = np.load(os.path.join(FEA_DATA_DIR, "segment_bfield_singleprobe_beta0180_4seg.npz"))
    X_all, y_all, f_all, s_all, c_all = data["X"], data["y"], data["f"], data["s"], data["c"]
    print(f"데이터 {len(y_all)}개 로드, 학습 중 노이즈: Uniform(0, {NOISE_MAX_UT})uT 가우시안 표준편차")

    X_mean2, X_std2 = X_all.mean(), X_all.std()
    X_norm_clean = (X_all - X_mean2) / X_std2  # 노이즈는 원 스케일(uT)에서 섞고 나중에 정규화

    fxy_all = f_all[:, :2]
    f_mean, f_std = fxy_all.mean(axis=0), fxy_all.std(axis=0)
    f_std[f_std < 1e-12] = 1.0
    f_norm = (fxy_all - f_mean) / f_std

    s_mean, s_std = s_all.mean(), s_all.std()
    s_norm = (s_all - s_mean) / s_std

    c_mean, c_std = c_all.mean(axis=0), c_all.std(axis=0)
    c_std[c_std < 1e-12] = 1.0
    c_norm = (c_all - c_mean) / c_std

    rng2 = np.random.default_rng(0)
    idx = rng2.permutation(len(X_all))
    split = int(0.9 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train_raw = X_all[train_idx]  # 원 스케일(uT) - 노이즈는 여기 섞음
    device = torch.device("cpu")

    train_ds = TensorDataset(
        torch.tensor(X_train_raw).float(), torch.tensor(y_all[train_idx]).long(),
        torch.tensor(f_norm[train_idx]).float(), torch.tensor(s_norm[train_idx]).float(),
        torch.tensor(c_norm[train_idx]).float())
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    val_X_raw = X_all[val_idx]
    val_y = torch.tensor(y_all[val_idx]).long()
    val_f_phys = f_all[val_idx][:, :2]
    val_s_phys = s_all[val_idx]

    model = SingleProbeClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    seg_criterion = nn.CrossEntropyLoss()
    force_criterion = nn.MSELoss()
    s_criterion = nn.MSELoss()
    config_criterion = nn.MSELoss()

    rng_train = np.random.default_rng(1)
    for epoch in range(N_EPOCHS):
        model.train()
        for bx_raw, by, bf, bs, bc in train_loader:
            noise_std = rng_train.uniform(0, NOISE_MAX_UT)
            bx = bx_raw + torch.randn_like(bx_raw) * noise_std
            bx = (bx - X_mean2) / X_std2
            optimizer.zero_grad()
            seg_logits, force_pred, s_pred, config_pred = model(bx)
            loss = (seg_criterion(seg_logits, by) + force_criterion(force_pred, bf)
                    + s_criterion(s_pred, bs) + config_criterion(config_pred, bc))
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vx = torch.tensor((val_X_raw - X_mean2) / X_std2).float()
                seg_logits, _, s_pred, _ = model(vx)
                acc = (seg_logits.argmax(dim=1) == val_y).float().mean().item()
                s_pred_phys = s_pred.numpy() * s_std + s_mean
                print(f"epoch {epoch+1}/{N_EPOCHS}  val_acc(노이즈0)={acc*100:.1f}%  s_R2(노이즈0)={r2(s_pred_phys, val_s_phys):.3f}")

    model.eval()
    print("\n=== 노이즈 강건성 비교 (재학습 후) ===")
    print(f"{'noise_std(uT)':>14} {'s_R2':>8} {'s_MAE(mm)':>10} {'Fx_R2':>8} {'Fy_R2':>8}")
    for noise_std in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]:
        rng_n = np.random.default_rng(42)
        X_noisy = val_X_raw + rng_n.normal(0, noise_std, val_X_raw.shape).astype(np.float32)
        X_norm = (X_noisy - X_mean2) / X_std2
        with torch.no_grad():
            _, force_pred, s_pred, _ = model(torch.tensor(X_norm).float())
        s_pred_phys = s_pred.numpy() * s_std + s_mean
        f_pred_phys = force_pred.numpy() * f_std + f_mean
        sr2 = r2(s_pred_phys, val_s_phys)
        mae = np.mean(np.abs(s_pred_phys - val_s_phys))
        fxr2 = r2(f_pred_phys[:, 0], val_f_phys[:, 0])
        fyr2 = r2(f_pred_phys[:, 1], val_f_phys[:, 1])
        print(f"{noise_std:>14.3f} {sr2:>8.3f} {mae:>10.2f} {fxr2:>8.3f} {fyr2:>8.3f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "X_mean": X_mean2, "X_std": X_std2,
                "f_mean": f_mean, "f_std": f_std, "s_mean": s_mean, "s_std": s_std,
                "c_mean": c_mean, "c_std": c_std, "noise_max_ut": NOISE_MAX_UT},
               os.path.join(MODELS_DIR, "position_segment_classifier_noiserobust_beta0180_4seg.pth"))
    print(f"\n저장: {MODELS_DIR}/position_segment_classifier_noiserobust_beta0180_4seg.pth")
