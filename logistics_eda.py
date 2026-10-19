"""Week 3 logistics EDA on a hypothetical dataset.
Install: pip install pandas numpy matplotlib
Run: python logistics_eda.py
All records are simulated; they are not actual company or DataCo results.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT=Path("."); VIZ=OUT/"visualizations"; VIZ.mkdir(exist_ok=True)
rng=np.random.default_rng(23); n=120
distance=rng.integers(40,1001,n); volume=rng.integers(5,81,n)
mode=rng.choice(["Road","Air","Rail"],n,p=[.62,.18,.20])
days=np.maximum(1,np.round(distance/260+rng.normal(1.1,.9,n)+np.select([mode=="Air",mode=="Rail"],[-1,.3],default=0),1))
rate=np.select([mode=="Air",mode=="Rail"],[8.,2.1],default=3.2)
cost=np.maximum(100,np.round(180+distance*rate+volume*5+rng.normal(0,120,n),2))
threshold=np.where(mode=="Air",3.5,np.where(mode=="Rail",4.5,4.0))
status=np.where(days>threshold,"Delayed","On-time")
df=pd.DataFrame({"Shipment_ID":[f"SH{i:04d}" for i in range(1,n+1)],"Shipment_Distance_km":distance,"Shipment_Volume_units":volume,"Transportation_Mode":mode,"Delivery_Time_Days":days,"Transportation_Cost_INR":cost,"Delivery_Status":status})
df.to_csv(OUT/"logistics_hypothetical_dataset.csv",index=False)
numeric=["Shipment_Distance_km","Shipment_Volume_units","Delivery_Time_Days","Transportation_Cost_INR"]
print("Dataset shape:",df.shape)
print("\\nMissing values:\\n",df.isna().sum())
print("\\nDescriptive statistics:\\n",df[numeric].describe().round(2))
print("\\nCorrelation matrix:\\n",df[numeric].corr().round(3))
summary=df.groupby("Transportation_Mode").agg(Shipments=("Shipment_ID","count"),Avg_Delivery_Days=("Delivery_Time_Days","mean"),Avg_Cost_INR=("Transportation_Cost_INR","mean"),On_Time_Rate=("Delivery_Status",lambda s:(s=="On-time").mean()*100)).round(2)
print("\\nBy mode:\\n",summary)
on_time=(df.Delivery_Status=="On-time").mean()*100
print(f"\\nOn-time rate: {on_time:.2f}% | Delay rate: {100-on_time:.2f}%")
plt.figure(figsize=(8,5)); plt.hist(df.Delivery_Time_Days,bins=12,edgecolor="black"); plt.title("Distribution of Delivery Times"); plt.xlabel("Delivery time (days)"); plt.ylabel("Shipments"); plt.tight_layout(); plt.savefig(VIZ/"delivery_time_distribution.png",dpi=150); plt.close()
ms=df.groupby("Transportation_Mode").Transportation_Cost_INR.mean().reindex(["Road","Rail","Air"])
plt.figure(figsize=(8,5)); plt.bar(ms.index,ms.values); plt.title("Average Transportation Cost by Mode"); plt.xlabel("Mode"); plt.ylabel("Average cost (INR)"); plt.tight_layout(); plt.savefig(VIZ/"transportation_cost_by_mode.png",dpi=150); plt.close()
plt.figure(figsize=(8,5))
for m in ["Road","Air","Rail"]:
 d=df[df.Transportation_Mode==m]; plt.scatter(d.Shipment_Distance_km,d.Transportation_Cost_INR,label=m,alpha=.75)
plt.title("Shipment Distance vs Transportation Cost"); plt.xlabel("Distance (km)"); plt.ylabel("Cost (INR)"); plt.legend(); plt.tight_layout(); plt.savefig(VIZ/"distance_vs_cost.png",dpi=150); plt.close()
corr=df[numeric].corr(); plt.figure(figsize=(7,5)); im=plt.imshow(corr.values,vmin=-1,vmax=1)
labels=["Distance km","Volume units","Delivery days","Cost INR"]; plt.xticks(range(4),labels,rotation=25,ha="right"); plt.yticks(range(4),labels)
for i in range(4):
 for j in range(4): plt.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center")
plt.title("Correlation Heatmap"); plt.colorbar(im); plt.tight_layout(); plt.savefig(VIZ/"correlation_heatmap.png",dpi=150); plt.close()
plt.figure(figsize=(8,5)); plt.boxplot([df.loc[df.Transportation_Mode==m,"Delivery_Time_Days"] for m in ["Road","Rail","Air"]],labels=["Road","Rail","Air"]); plt.title("Delivery Time by Transportation Mode"); plt.xlabel("Mode"); plt.ylabel("Days"); plt.tight_layout(); plt.savefig(VIZ/"delivery_time_by_mode_boxplot.png",dpi=150); plt.close()
(OUT/"eda_summary.txt").write_text(f"HYPOTHETICAL DATA ONLY\\nRows: {len(df)}\\nMean delivery: {df.Delivery_Time_Days.mean():.2f} days\\nMedian delivery: {df.Delivery_Time_Days.median():.2f} days\\nMean cost: INR {df.Transportation_Cost_INR.mean():.2f}\\nMean distance: {df.Shipment_Distance_km.mean():.2f} km\\nOn-time rate: {on_time:.2f}%\\nDelay rate: {100-on_time:.2f}%\\n\\n{summary.to_string()}\\n",encoding="utf-8")
print("Saved CSV, summary, and five charts in visualizations/.")
