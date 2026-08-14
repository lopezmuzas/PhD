"""Transformer minimo por caracteres (Lab 4). Estructura tipo nanoGPT.

Uso:
    python train_cervantes.py data/quijote.txt

El corpus: https://www.gutenberg.org/ebooks/2000 (dominio publico).
"""
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- hiperparametros ---
BLOQUE, LOTE = 256, 32          # contexto y tamano de lote
N_EMB, N_CABEZAS, N_CAPAS = 384, 6, 6
DROPOUT, LR, PASOS = 0.2, 3e-4, 5000
EVAL_CADA, EVAL_ITERS = 500, 20
DISP = "cuda" if torch.cuda.is_available() else "cpu"

# --- datos ---
RUTA = sys.argv[1] if len(sys.argv) > 1 else "data/quijote.txt"
texto = open(RUTA, encoding="utf-8").read()
vocab = sorted(set(texto))
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]                    # noqa: E731
decode = lambda l: "".join(itos[i] for i in l)             # noqa: E731

datos = torch.tensor(encode(texto), dtype=torch.long)
corte = int(0.9 * len(datos))
train, val = datos[:corte], datos[corte:]


def lote(split):
    d = train if split == "train" else val
    ix = torch.randint(len(d) - BLOQUE, (LOTE,))
    x = torch.stack([d[i:i + BLOQUE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOQUE + 1] for i in ix])
    return x.to(DISP), y.to(DISP)


# --- modelo ---
class Cabeza(nn.Module):
    def __init__(self, tam):
        super().__init__()
        self.key = nn.Linear(N_EMB, tam, bias=False)
        self.query = nn.Linear(N_EMB, tam, bias=False)
        self.value = nn.Linear(N_EMB, tam, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOQUE, BLOQUE)))
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # escalado 1/sqrt(dk)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal
        att = self.drop(F.softmax(att, dim=-1))
        return att @ v


class MultiCabeza(nn.Module):
    def __init__(self, n, tam):
        super().__init__()
        self.cabezas = nn.ModuleList([Cabeza(tam) for _ in range(n)])
        self.proj = nn.Linear(N_EMB, N_EMB)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.proj(
            torch.cat([h(x) for h in self.cabezas], dim=-1)))


class Bloque(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiCabeza(N_CABEZAS, N_EMB // N_CABEZAS)
        self.ff = nn.Sequential(
            nn.Linear(N_EMB, 4 * N_EMB), nn.ReLU(),
            nn.Linear(4 * N_EMB, N_EMB), nn.Dropout(DROPOUT))
        self.ln1, self.ln2 = nn.LayerNorm(N_EMB), nn.LayerNorm(N_EMB)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))          # conexiones residuales
        return x + self.ff(self.ln2(x))


class ModeloCervantes(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(len(vocab), N_EMB)
        self.pos = nn.Embedding(BLOQUE, N_EMB)
        self.bloques = nn.Sequential(*[Bloque() for _ in range(N_CAPAS)])
        self.ln = nn.LayerNorm(N_EMB)
        self.head = nn.Linear(N_EMB, len(vocab))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.ln(self.bloques(x))
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperatura=0.8):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOQUE:])
            probs = F.softmax(logits[:, -1, :] / temperatura, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# --- entrenamiento ---
@torch.no_grad()
def evaluar(modelo):
    modelo.eval()
    out = {}
    for split in ("train", "val"):
        out[split] = torch.stack(
            [modelo(*lote(split))[1] for _ in range(EVAL_ITERS)]).mean().item()
    modelo.train()
    return out


def main():
    modelo = ModeloCervantes().to(DISP)
    n = sum(p.numel() for p in modelo.parameters()) / 1e6
    print(f"{n:.1f}M parametros | vocabulario {len(vocab)} | disp {DISP}")
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR)

    for paso in range(PASOS + 1):
        if paso % EVAL_CADA == 0:
            p = evaluar(modelo)
            print(f"paso {paso:5d} | train {p['train']:.4f} | val {p['val']:.4f}")
            # muestra en cada evaluacion: la pelicula del aprendizaje
            semilla = torch.zeros((1, 1), dtype=torch.long, device=DISP)
            muestra = decode(modelo.generate(semilla, 200)[0].tolist())
            with open(f"muestras/paso-{paso:05d}.txt", "w",
                      encoding="utf-8") as f:
                f.write(f"train {p['train']:.4f} val {p['val']:.4f}\n\n{muestra}")
        x, y = lote("train")
        _, loss = modelo(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.save({"modelo": modelo.state_dict(), "vocab": vocab},
               "checkpoints/cervantes.pt")

    semilla = torch.tensor([encode("En un lugar de la Mancha")],
                           dtype=torch.long, device=DISP)
    print(decode(modelo.generate(semilla, 500)[0].tolist()))


if __name__ == "__main__":
    main()
