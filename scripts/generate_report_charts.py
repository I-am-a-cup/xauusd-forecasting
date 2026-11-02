"""Графики аналитического README; только из сохранённых результатов."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"results/arima_rf"
TARGET=ROOT/"results/report"
NAMES={"naive":"Naive","arima":"ARIMA","random_forest":"Random Forest","mlp":"MLP"}
ORDER=list(NAMES)

def main():
    m=pd.read_csv(DATA/"comparison.csv")
    blocks=pd.read_csv(DATA/"temporal_blocks.csv")
    assert len(m)==8 and (m.n==47).all() and set(m.model)==set(ORDER)
    TARGET.mkdir(exist_ok=True,parents=True)

    fig, axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for h,ax in zip((1,5),axes):
        data=m[m.horizon.eq(h)].set_index("model").loc[ORDER]
        vals=data.mae.to_numpy()
        ax.bar(range(4),vals)
        ax.set_xticks(range(4),[NAMES[x] for x in ORDER],rotation=15)
        ax.set_ylabel("MAE (USD per troy oz)")
        ax.set_title(f"Horizon: {h} trading day(s)")
        ax.set_ylim(0,vals.max()*1.25)
        for i,y in enumerate(vals):
            ax.text(i,y+vals.max()*.027,f"{y:.2f}",ha="center",fontsize=9)
    fig.suptitle("XAU/USD | Same 47 forecast origins")
    fig.savefig(TARGET/"mae_comparison.png",dpi=155)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for h,ax in zip((1,5),axes):
        for name in ORDER:
            d=blocks[blocks.model.eq(name)&blocks.horizon.eq(h)].sort_values("block")
            ax.plot(d.block.to_numpy(),d.mae.to_numpy(),marker="o",label=NAMES[name])
        ax.set_title(f"MAE by test block: h={h}")
        ax.set_xticks((1,2,3))
        ax.set_ylabel("MAE (USD per troy oz)")
        ax.set_xlabel("Consecutive block")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(TARGET/"temporal_stability.png",dpi=155)
    plt.close(fig)
    print("Created 2 figures:",TARGET)

if __name__=="__main__":
    main()
