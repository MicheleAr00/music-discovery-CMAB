"""Build report assets from small official aggregates and saved notebook outputs.

Never executes notebook cells, reads the listening TSV, or replays policies.
"""
from pathlib import Path
from html.parser import HTMLParser
import base64
import hashlib
import json
import sys
import textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
for sub in ("figures", "tables", "data", "sections"):
    (OUT / sub).mkdir(exist_ok=True)
LABEL = {"random": "Random", "popularity": "Global Popularity", "greedy": "Personalized Greedy",
         "epsilon_greedy": r"$\varepsilon$-Greedy", "linucb": "LinUCB"}
ORDER = list(LABEL)
COLORS = ["#7d8893", "#d19742", "#39766b", "#4f8cae", "#775395"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
manifest = json.loads((ROOT / "experiments/configs/final_selection_manifest.json").read_text(encoding="utf-8"))
vpath = ROOT / manifest["validation_results"]["aggregate_csv"]
tpath = ROOT / "experiments/results/test/044f9269d4e3-5b076e7a/runs.csv"
val, test = pd.read_csv(vpath), pd.read_csv(tpath)
assert len(val) == 36 and val.run_id.is_unique and len(test) == 5 and test.run_id.is_unique
assert set(val.configuration_hash) == {manifest["validation_config_hash"]}
assert set(test.configuration_hash) == {manifest["final_test_config_hash"]}
assert set(test.policy) == set(ORDER) and set(test.candidate_pool_depth) == {100}
assert set(val.dataset_content_hash) == set(test.dataset_content_hash) == {manifest["dataset_content_hash"]}
assert hashlib.sha256(vpath.read_bytes()).hexdigest() == manifest["validation_results"]["aggregate_csv_sha256"]
assert hashlib.sha256(tpath.read_bytes()).hexdigest() == "7089ff59433a1725254da3f5cdd4157fafbe8a11d0140f108cabf18385a27831"
sys.path.insert(0,str(ROOT))
from src.experiments.config import ExperimentConfig
for name,field in [("validation.json","validation_config_hash"),("final_test.json","final_test_config_hash")]:
    assert ExperimentConfig.from_json(ROOT/"experiments/configs"/name).configuration_hash == manifest[field]
assert hashlib.sha256((ROOT/"experiments/configs/final_test.json").read_bytes()).hexdigest() == manifest["final_test_config_file_sha256"]
assert all((test.catalog_coverage == test["catalog_coverage@20"]) & (test.artist_coverage == test["artist_coverage@20"]))
selected = val[val.run_id.isin([x["run_id"] for x in manifest["selected_validation_runs"].values()])].set_index("policy").loc[ORDER]
test = test.set_index("policy").loc[ORDER]
def esc(x):
    return str(x).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
def fmt(x): return f"{float(x):.6f}".replace(".", ",")
def table(name, rows, headers, spec=None):
    spec = spec or "l" + "r" * (len(headers)-1)
    text = r"\begin{tabular}{" + spec + "}\n\\toprule\n" + " & ".join(headers) + r" \\" + "\n\\midrule\n"
    text += "\n".join(" & ".join(str(v) for v in row) + r" \\" for row in rows)
    text += "\n\\bottomrule\n\\end{tabular}\n"
    (OUT / "tables" / (name+".tex")).write_text(text, encoding="utf-8")
def label(p): return LABEL[p].replace("$", "").replace(r"\varepsilon", r"$\varepsilon$")
metrics = ["ndcg@10", "precision@10", "recall@10", "average_reward", "track_novelty@10", "artist_novelty@10", "artist_ild@10"]
for name, df in [("validation",selected),("test",test)]:
    table(name, [[label(p), *[fmt(df.loc[p,m]) for m in metrics]] for p in ORDER],
          ["Policy","NDCG","Prec.","Recall","Reward","TN","AN","ILD"])
    table(name+"_coverage", [[label(p), *[fmt(df.loc[p,f"{m}@{k}"]) for k in (5,10,20) for m in ("catalog_coverage","artist_coverage")]] for p in ORDER],
          ["Policy",r"$C_I@5$",r"$C_G@5$",r"$C_I@10$",r"$C_G@10$",r"$C_I@20$",r"$C_G@20$"])
table("test_cutoffs", [[label(p), k, *[fmt(test.loc[p,f"{m}@{k}"]) for m in ("ndcg","precision","recall","track_novelty","artist_novelty","artist_ild")]] for p in ORDER for k in (5,20)],
      ["Policy","K","NDCG","Prec.","Recall","TN","AN","ILD"])
table("runtimes", [[label(p), *[f"{test.loc[p,m]/60:.2f}".replace(".",",") for m in
       ["runtime_seconds","warmup_runtime_seconds","simulation_runtime_seconds"]]] for p in ORDER],
       ["Policy","Totale [min]","Warm-up/replay [min]","Simulazione [min]"])
table("validation_test", [[label(p),fmt(selected.loc[p,"ndcg@10"]),fmt(test.loc[p,"ndcg@10"]),fmt(test.loc[p,"ndcg@10"]-selected.loc[p,"ndcg@10"]),f'{100*(test.loc[p,"ndcg@10"]/selected.loc[p,"ndcg@10"]-1):.2f}'.replace(".",",")+r"\%",
      fmt(test.loc[p,"average_reward"]-selected.loc[p,"average_reward"])] for p in ORDER],
      ["Policy","Val.","Test",r"$\Delta$ NDCG",r"$\Delta$ rel.",r"$\Delta$ Reward"])
rows=[]
for n in (50,100,200):
    for p in ORDER:
        d=val[(val.candidate_pool_depth==n)&(val.policy==p)]
        if p=="linucb": d=d[d.alpha==2]
        if p=="epsilon_greedy": d=d[d.epsilon==0.05]
        r=d.iloc[0]; rows.append([n,label(p),fmt(r["ndcg@10"]),fmt(r["average_reward"]),fmt(r["artist_novelty@10"]),fmt(r["artist_ild@10"])])
table("sensitivity", rows, ["N","Policy","NDCG@10","Reward","AN@10","ILD@10"])
rows=[]
for _,r in val.sort_values(["candidate_pool_depth","policy","alpha","epsilon"]).iterrows():
    par = f'{r.alpha:g}' if r.policy=="linucb" else (f'{r.epsilon:g}' if r.policy=="epsilon_greedy" else "--")
    rows.append(" & ".join([str(r.candidate_pool_depth),label(r.policy),par,fmt(r["ndcg@10"]),fmt(r.average_reward)])+r" \\")
(OUT/"tables/grid.tex").write_text("\\begin{longtable}{rlrrr}\n\\caption{Griglia ufficiale di validation: 36 run.}\\\\\n\\toprule N & Policy & Parametro & NDCG@10 & Reward \\\\\n\\midrule\\endfirsthead\n\\toprule N & Policy & Parametro & NDCG@10 & Reward \\\\\n\\midrule\\endhead\n"+"\n".join(rows)+"\n\\bottomrule\\end{longtable}",encoding="utf-8")
# Preserve genuine EDA figures without re-running their expensive source cells.
nb=json.loads((ROOT/"notebooks/01_dataset_analysis.ipynb").read_text(encoding="utf-8"))
for idx,name in [(21,"eda_activity"),(41,"eda_concentration"),(45,"eda_repetition")]:
    image=next(o["data"]["image/png"] for o in nb["cells"][idx]["outputs"] if "image/png" in o.get("data",{}))
    (OUT/"figures"/f"{name}.png").write_bytes(base64.b64decode(image))
nbstat=json.loads((ROOT/"notebooks/03_statistical_analysis.ipynb").read_text(encoding="utf-8"))
class Tables(HTMLParser):
    def __init__(self): super().__init__(); self.rows=[]; self.row=None; self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=="tr": self.row=[]
        if tag in ("td","th"): self.cell=""
    def handle_data(self,data):
        if self.cell is not None: self.cell+=data
    def handle_endtag(self,tag):
        if tag in ("td","th") and self.cell is not None:
            self.row.append(self.cell.strip()); self.cell=None
        if tag=="tr" and self.row: self.rows.append(self.row); self.row=None
stat_tables=[]
for c in nbstat["cells"]:
    for o in c.get("outputs",[]):
        if "text/html" in o.get("data",{}):
            p=Tables(); p.feed("".join(o["data"]["text/html"]))
            if p.rows: stat_tables.append(p.rows)
(OUT/"data/statistical_tables.json").write_text(json.dumps(stat_tables,indent=2,ensure_ascii=False),encoding="utf-8")
verification = next(t for t in stat_tables if "equivalenza entro 1e-12" in t[0])
assert len(verification) == 66 and all(r[-1] == "True" for r in verification[1:])
# Redraw only five EDA concentration points already saved in the notebook.
# The histogram stays the original raster: no bins are inferred from pixels.
eda_concentration = []
for idx in (39, 40):
    output = next(o["data"]["text/html"] for o in nb["cells"][idx]["outputs"]
                  if "text/html" in o.get("data", {}))
    parsed = Tables()
    parsed.feed("".join(output))
    points = [(float(row[-3]), float(row[-1]) * 100) for row in parsed.rows[1:]]
    assert len(points) == 5 and [x for x, _ in points] == [1, 5, 10, 20, 50]
    eda_concentration.append(points)
fig, ax = plt.subplots(figsize=(6.6, 2.8))
for points, name, color in zip(eda_concentration, ["Artisti", "Tracce"], ["#39766b", "#775395"]):
    ax.plot(*zip(*points), "o-", label=name, color=color, lw=1.8)
ax.set(xlabel="Quota delle entità più popolari (%)", ylabel="Quota degli ascolti (%)",
       xticks=[1, 5, 10, 20, 50], ylim=(0, 103))
ax.grid(alpha=.2)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "figures/eda_concentration.pdf")
plt.close(fig)
# Descriptive charts directly from official aggregates.
fig,axs=plt.subplots(1,2,figsize=(10,3.5))
for ax,m,title in zip(axs,["ndcg@10","average_reward"],["NDCG@10 (macro utente)","Reward Top-1: media di log(1+ascolti)"]):
    x=np.arange(5); ax.bar(x-.18,selected[m],.36,label="Validation",color="#799baa")
    ax.bar(x+.18,test[m],.36,label="Test",color="#755985")
    ax.set_xticks(x,["Random","Pop.","Greedy",r"$\varepsilon$-G.","LinUCB"]); ax.set_title(title); ax.grid(axis="y",alpha=.2)
axs[0].legend();fig.tight_layout();fig.savefig(OUT/"figures/validation_test.pdf");plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(6.8,2.9))
for ax,p,col in zip(axs,["linucb","epsilon_greedy"],["alpha","epsilon"]):
    for n,color in zip((50,100,200),COLORS[2:]):
        d=val[(val.policy==p)&(val.candidate_pool_depth==n)].sort_values(col)
        ax.plot(d[col],d["ndcg@10"],"o-",color=color,label=f"N={n}")
    ax.set_xlabel(r"$\alpha$" if col=="alpha" else r"$\varepsilon$")
    ax.set_ylabel("NDCG@10");ax.grid(alpha=.2);ax.legend();ax.set_title("LinUCB" if p=="linucb" else r"$\varepsilon$-Greedy")
fig.tight_layout();fig.savefig(OUT/"figures/sweeps.pdf");plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(6.8,3.2))
for p,color in zip(ORDER,COLORS):
    r=test.loc[p]
    for ax,y in zip(axs,["artist_novelty@10","artist_ild@10"]):
        ax.scatter(r["ndcg@10"],r[y],s=70,color=color,label=LABEL[p])
        ax.set_xlabel("NDCG@10");ax.grid(alpha=.2)
axs[0].set_ylabel("Novelty artista@10");axs[1].set_ylabel("Diversità fra artisti (ILD@10)")
handles,labels=axs[0].get_legend_handles_labels()
labels = ["Random", "Popolarità globale", "Greedy personale", r"$\varepsilon$-Greedy", "LinUCB"]
fig.legend(handles,labels,loc="lower center",ncol=3,bbox_to_anchor=(.5,-.13),frameon=False,fontsize=9)
fig.tight_layout();fig.savefig(OUT/"figures/tradeoff.pdf");plt.close(fig)
cache_path=ROOT/"notebooks/data/qualitative_examples.json"
cache=json.loads(cache_path.read_text(encoding="utf-8"))
assert hashlib.sha256(cache_path.read_bytes()).hexdigest()=="ed582fc9a26189785bf440be58b0a4ee9a419c77e20c8131bf0ea6ed7840341b"
# One fixed, previously verified example; no new selection on held-out logs.
example = next(e for e in cache["examples"] if e["alias"] == "U04")
fig, axs = plt.subplots(1, 3, figsize=(6.8, 7.1))
for ax, policy, title in zip(axs, ["greedy", "linucb", "epsilon_greedy"],
                            ["Personalized Greedy", r"LinUCB $\alpha=2$", r"$\varepsilon$-Greedy $\varepsilon=0{,}05$"]):
    data = example["policies"][policy]
    rows = data["recommendations_top10"]
    assert len(rows) == 10 and [r["rank"] for r in rows] == list(range(1, 11))
    ax.set(xlim=(0, 1), ylim=(0, 10))
    ax.axis("off")
    ax.set_title(title, fontsize=9.2, fontweight="bold", pad=28)
    ax.text(.5, 1.015, f'Reward Top-1: {data["reward_top1_offline_proxy"]:.6f}',
            transform=ax.transAxes, ha="center", fontsize=8.2)
    for row in rows:
        y = 10 - row["rank"]
        new = row["artist_is_new_before_t"]
        assert isinstance(new, bool)
        ax.add_patch(Rectangle((0, y + .02), 1, .96,
                              facecolor="#e4f2eb" if new else "#f2f3f6",
                              edgecolor="#28394a" if row["rank"] == 1 else "white",
                              lw=1.5 if row["rank"] == 1 else .7))
        artist = textwrap.fill(row["artist_name"], width=27, break_long_words=False)
        track = textwrap.fill(row["track_title"], width=30, break_long_words=False)
        ax.text(.045, y + .86, f'{row["rank"]:02d}  {artist}', va="top",
                fontsize=8.2, fontweight="bold", linespacing=1.1)
        # Keep artist/title disjoint even when an artist wraps to two lines.
        title_y = y + .63 - .17 * artist.count("\n")
        ax.text(.045, title_y, track, va="top", fontsize=7.8, linespacing=1.12)
fig.subplots_adjust(left=.01, right=.99, top=.93, bottom=.035, wspace=.055)
fig.text(.5, .005, "U04 · 09/04/2009 15:08:15 · verde: artista nuovo · bordo scuro: Top-1",
         ha="center", fontsize=8)
fig.savefig(OUT / "figures/qualitative_U04.pdf")
plt.close(fig)
table("examples",[[e["alias"],e["decision_time"].replace("T"," "),label(p),
                   esc(e["policies"][p]["recommendations_top10"][0]["artist_name"]),
                   esc(e["policies"][p]["recommendations_top10"][0]["track_title"]),
                   fmt(e["policies"][p]["reward_top1_offline_proxy"]),
                   e["policies"][p]["artist_new_distinct_count_top10"]]
                  for e in cache["examples"][:2] for p in ["greedy","linucb","epsilon_greedy"]],
      ["Alias","Istante","Policy","Artista Top-1","Titolo Top-1","Reward","Nuovi"],"lll p{2cm}p{3cm}rr")
for e in [cache["examples"][0],cache["examples"][3]]:
    table("example_"+e["alias"],[[label(p),esc(e["policies"][p]["recommendations_top10"][0]["artist_name"]),
         esc(e["policies"][p]["recommendations_top10"][0]["track_title"]),
         fmt(e["policies"][p]["reward_top1_offline_proxy"]),e["policies"][p]["artist_new_distinct_count_top10"]]
         for p in ["greedy","linucb","epsilon_greedy"]],
         ["Policy","Artista Top-1","Titolo Top-1","Reward","Nuovi"],"lp{2.6cm}p{4.3cm}rr")
ci_rows=[r for r in stat_tables[-1][1:] if r[2]=="LinUCB − Personalized Greedy"]
assert len(ci_rows) == 5
for r in ci_rows:
    assert abs(float(r[3]) - (test.loc["linucb",r[1]]-test.loc["greedy",r[1]])) < 5.1e-7
table("paired_ci",[[esc(r[1]),r[3].replace(".",","),r[4].replace(".",","),r[5].replace(".",",")] for r in ci_rows],
      ["Metrica",r"$\Delta$","IC 95\\% inf.","IC 95\\% sup."])
fig,axs=plt.subplots(1,3,figsize=(6.8,2.65),gridspec_kw={"width_ratios":[1,1,1.7]})
groups = [ci_rows[:1], ci_rows[1:2], ci_rows[2:]]
short_names = {"average_reward":"Reward", "ndcg@10":"NDCG@10", "track_novelty@10":"TN@10",
               "artist_novelty@10":"AN@10", "artist_ild@10":"ILD@10"}
for ax, group, title in zip(axs, groups, ["Reward Top-1", "NDCG@10", "Composizione @10"]):
    for i,r in enumerate(group):
        d,lo,hi=map(float,r[3:6])
        ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt="o",capsize=4,color="#775395")
        ax.annotate(short_names[r[1]],(d,i),xytext=(0,11),textcoords="offset points",
                    ha="center",fontsize=8.5)
    ax.set_ylim(-.55,len(group)-.1)
    ax.axvline(0,color="#555555",ls="--",lw=1)
    ax.set_yticks([])
    ax.set_title(title,fontsize=10)
    ax.set_xlabel("Differenza",fontsize=9)
    ax.tick_params(axis="x",labelsize=8)
    ax.grid(axis="x",alpha=.2)
fig.suptitle("LinUCB − Personalized Greedy · IC bootstrap al 95%",fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT/"figures/paired_ci.pdf")
plt.close(fig)
# Store exact evidence and hashes, not large official logs.
evidence={"source_commit": __import__("subprocess").check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
          "sources":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [vpath,tpath,ROOT/"experiments/configs/final_selection_manifest.json",cache_path]},
          "validation_run_ids":list(val.run_id),"test_run_ids":list(test.run_id),
          "test_users":list(test.n_users.astype(int)),"test_decisions":list(test.n_decisions.astype(int)),
          "notes":["13 standardized numeric features + 2 unchanged binary flags = dimension 15",
                   "EDA identity counts are historical exploratory entities, not canonical item catalog",
                   "No TSV scans or policy executions performed"]}
evidence["checkpoints"] = []
for frame, directory in [(val, "ea8616724509-5b076e7a1cb3"),
                         (test.reset_index(), "final-test-044f9269d4e3-5b076e7a1cb3")]:
    checkpoint_dir = ROOT / "experiments/results/checkpoints" / directory
    markers = list(checkpoint_dir.glob("*.checkpoint.json"))
    assert len(markers) == len(frame)
    assert {p.name.removesuffix(".checkpoint.json") for p in markers} == set(frame.run_id)
    for _, row in frame.iterrows():
        marker_path = checkpoint_dir / f"{row.run_id}.checkpoint.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert marker["completion_status"] == "completed"
        identity = marker["identity"]
        for field in ["run_id","policy","configuration_hash","dataset_content_hash","git_commit","scaler_digest"]:
            assert identity[field] == row[field], (row.run_id,field)
        for metric,value in marker["metrics"].items():
            assert abs(float(row[metric])-value) < 1e-12, (row.run_id,metric)
        assert (checkpoint_dir / marker["result_filename"]).is_file()
        evidence["checkpoints"].append({"run_id":row.run_id,"marker_sha256":hashlib.sha256(marker_path.read_bytes()).hexdigest(),
                                       "recorded_detail_sha256":marker["result_sha256"]})
(OUT/"data/provenance.json").write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding="utf-8")
val.to_csv(OUT/"data/validation_summary.csv",index=False)
test.reset_index().to_csv(OUT/"data/test_summary.csv",index=False)
print("Report assets built; official inputs unchanged.")
print("Statistical reconstruction: 65 verified macro-user aggregates; five paired effects checked.")
print("Verified 36 validation and 5 test checkpoints, with unchanged source configuration hashes.")
