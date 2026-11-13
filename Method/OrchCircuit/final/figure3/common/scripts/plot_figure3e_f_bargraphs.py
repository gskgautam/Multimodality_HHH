#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

R=(next((p for p in Path(__file__).resolve().parents if (p / "README.md").is_file()), Path.cwd()) / "Method/OrchCircuit/final/figure3")
D=json.loads((R/"human_rating_analysis_PRIVATE.json").read_text())
GREEN="#25835b"; RED="#bd4b4b"; GRAY="#9299a1"; SEL="#ba4b46"; CTL="#3679a7"

# 3e: paired differences, all nine selector-control pairs
rows=D["matchups"]; labels=[f"{r['selector_name']} vs {r['control_name']}" for r in rows]; y=np.arange(len(rows))
fig,axs=plt.subplots(1,3,figsize=(17,7.4),sharey=True,constrained_layout=True)
panels=[
("WR net preference","net_preference_pp","net_preference_ci_pp",True),
("SS change in unsafe flags","delta_unsafe_pp","delta_unsafe_ci_pp",False),
("TI change in joint-positive rate","delta_joint_ti_pp","delta_joint_ti_ci_pp",True)]
for ax,(title,key,ckey,pos) in zip(axs,panels):
 v=np.array([r[key] for r in rows]); ci=np.array([r[ckey] for r in rows])
 err=np.maximum(0,np.vstack((v-ci[:,0],ci[:,1]-v)))
 cols=[GREEN if z>0 else RED if z<0 else GRAY for z in (v if pos else -v)]
 ax.barh(y,v,color=cols,alpha=.82,height=.68,zorder=2)
 ax.errorbar(v,y,xerr=err,fmt="none",ecolor="#343a40",capsize=3,zorder=3)
 ax.axvline(0,color="#20252a",lw=.9); ax.set_title(title,fontsize=11,fontweight="bold")
 ax.set_xlabel("Difference (percentage points)"); ax.grid(axis="x",alpha=.22,zorder=0)
 ax.set_yticks(y,labels,fontsize=8); ax.invert_yaxis()
axs[0].set_xlim(-14,14); axs[1].set_xlim(-1,1); axs[2].set_xlim(-40,40)
fig.suptitle("Figure 3e: OrchCircuit selector versus fixed control",fontsize=14,fontweight="bold")
fig.text(.5,-.02,"95% prompt-bootstrap intervals; positive favors selector for WR/TI, negative favors it for unsafe flags. Safety flags: 0/200 for every system.",ha="center",fontsize=9)
fig.savefig(R/"3e/figure_3e_bargraph.png",dpi=300,bbox_inches="tight")
fig.savefig(R/"3e/figure_3e_bargraph.pdf",bbox_inches="tight"); plt.close(fig)

# 3f: WR win share vs TI product; SS is constant and stated explicitly.
S=D["systems"]; order=["sel_wr","sel_ss","sel_ti","control_wr","control_ss","control_ti"]
names=[S[s]["system_name"] for s in order]; y=np.arange(6); cols=[SEL if s.startswith("sel_") else CTL for s in order]
fig,axs=plt.subplots(1,2,figsize=(12.5,6.3),sharey=True,constrained_layout=True)
for ax,key,ckey,xlab,lim in [
(axs[0],"wr_win_share_pct","wr_win_share_ci_pct","WR win share / 600 pairings (%)",(0,17)),
(axs[1],"ti_product_pct","ti_product_ci_pct","TI product (%)",(0,90))]:
 v=np.array([S[s][key] for s in order]); ci=np.array([S[s][ckey] for s in order])
 err=np.maximum(0,np.vstack((v-ci[:,0],ci[:,1]-v)))
 ax.barh(y,v,color=cols,alpha=.84,height=.68,zorder=2)
 ax.errorbar(v,y,xerr=err,fmt="none",ecolor="#343a40",capsize=3,zorder=3)
 ax.set_xlabel(xlab); ax.set_xlim(*lim); ax.grid(axis="x",alpha=.22,zorder=0)
 ax.set_yticks(y,names,fontsize=9); ax.invert_yaxis()
axs[0].set_title("Helpfulness preference",fontweight="bold")
axs[1].set_title("Truthfulness × informativeness",fontweight="bold")
fig.suptitle("Figure 3f: system-level human ratings",fontsize=14,fontweight="bold")
fig.text(.5,-.02,"Bars show point estimates; whiskers show 95% prompt-bootstrap intervals. Safety: 0 unsafe flags / 200 responses for every system.",ha="center",fontsize=9)
fig.legend(handles=[Patch(facecolor=SEL,label="OrchCircuit policy"),Patch(facecolor=CTL,label="Fixed control")],loc="lower center",bbox_to_anchor=(.5,-.085),ncol=2,frameon=False,fontsize=9)
fig.savefig(R/"3f/figure_3f_bargraph.png",dpi=300,bbox_inches="tight")
fig.savefig(R/"3f/figure_3f_bargraph.pdf",bbox_inches="tight"); plt.close(fig)
print("Wrote Figure 3e/3f bargraph PNG and PDF.")
