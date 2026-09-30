import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="GameProb Lab", page_icon="🏟️", layout="wide")
st.title("GameProb Lab — calibrated sports probabilities")
st.caption("The question is not “who is guaranteed to win?” but “how well do historical inputs become calibrated probabilities?”")


@st.cache_data
def demo_games(n=4000):
    rng=np.random.default_rng(51)
    date=pd.date_range("2014-01-01",periods=n,freq="D")
    rating=rng.normal(0,1,n); form=.65*rating+rng.normal(0,.65,n)
    home=rng.binomial(1,.5,n); rest=rng.integers(-3,4,n); injuries=rng.poisson(.7,n)-rng.poisson(.7,n)
    logit=1.05*rating+.55*form+.38*home+.07*rest-.23*injuries+rng.normal(0,.25,n)
    p=1/(1+np.exp(-logit)); win=rng.binomial(1,p)
    return pd.DataFrame({"date":date,"rating_difference":rating,"recent_form_difference":form,
                         "team_a_home":home,"rest_day_difference":rest,"injury_difference":injuries,"team_a_win":win})


def load(upload):
    if upload is None: return demo_games(),"Synthetic demonstration league (not real teams or odds)"
    df=pd.read_csv(upload)
    target=next((c for c in df if c.lower() in {"team_a_win","home_win","outcome","target"}),None)
    date=next((c for c in df if c.lower() in {"date","game_date","timestamp"}),None)
    if not target or not date: raise ValueError("CSV needs a date column and a binary outcome/team_a_win column.")
    df[date]=pd.to_datetime(df[date],errors="coerce")
    return df.rename(columns={target:"team_a_win",date:"date"}).dropna(subset=["date","team_a_win"]),f"Uploaded historical data: {upload.name}"


def fit(df):
    features=[c for c in df.select_dtypes(include=np.number) if c!="team_a_win"]
    if len(features)<2: raise ValueError("At least two numeric predictive columns are required.")
    ordered=df.sort_values("date").dropna(subset=features)
    cut=int(len(ordered)*.75); train,test=ordered.iloc[:cut],ordered.iloc[cut:]
    models={"Logistic baseline":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
            "Gradient boosting":GradientBoostingClassifier(n_estimators=140,max_depth=2,random_state=2)}
    fitted={}; rows=[]
    for name,m in models.items():
        m.fit(train[features],train.team_a_win); p=m.predict_proba(test[features])[:,1]
        rows.append({"Model":name,"Accuracy":accuracy_score(test.team_a_win,p>=.5),"ROC-AUC":roc_auc_score(test.team_a_win,p),
                     "Brier score ↓":brier_score_loss(test.team_a_win,p),"Log loss ↓":log_loss(test.team_a_win,p)})
        fitted[name]=(m,p)
    return features,train,test,fitted,pd.DataFrame(rows)


upload=st.sidebar.file_uploader("Optional historical game CSV",type="csv")
try:
    games,source=load(upload)
    if len(games)<200: raise ValueError("At least 200 dated games are required.")
    features,train,test,models,scores=fit(games)
except Exception as exc: st.error(str(exc)); st.stop()
choice=st.sidebar.selectbox("Model",scores.Model)
model,prob=models[choice]
tabs=st.tabs(["Evaluation","Calibration","Matchup simulator","Data & limitations"])
with tabs[0]:
    st.info(f"{source}. Train: earliest 75%; test: latest 25%.")
    st.dataframe(scores.style.format({c:"{:.3f}" for c in scores if c!="Model"}),width="stretch")
    view=pd.DataFrame({"Predicted probability":prob,"Actual outcome":test.team_a_win.astype(str)})
    st.plotly_chart(px.histogram(view,x="Predicted probability",color="Actual outcome",barmode="overlay",nbins=25,title="Prediction distributions"),width="stretch")
with tabs[1]:
    observed,predicted=calibration_curve(test.team_a_win,prob,n_bins=10,strategy="quantile")
    cal=pd.DataFrame({"Predicted":predicted,"Observed":observed})
    fig=px.line(cal,x="Predicted",y="Observed",markers=True,title="Reliability diagram")
    fig.add_scatter(x=[0,1],y=[0,1],name="Perfect calibration",line_dash="dash")
    st.plotly_chart(fig,width="stretch")
    binned=pd.qcut(prob,10,duplicates="drop"); check=pd.DataFrame({"bin":binned.astype(str),"actual":test.team_a_win,"p":prob}).groupby("bin",observed=True).agg(games=("actual","size"),predicted=("p","mean"),observed=("actual","mean"))
    st.dataframe(check.style.format({"predicted":"{:.1%}","observed":"{:.1%}"}),width="stretch")
    st.caption("A calibrated 70% forecast should win about 70% of the time across many comparable cases—not every time.")
with tabs[2]:
    values={}; cols=st.columns(2)
    for i,f in enumerate(features):
        s=train[f]; lo,hi=float(s.quantile(.01)),float(s.quantile(.99))
        values[f]=cols[i%2].slider(f,lo,hi,float(s.median()))
    p=float(model.predict_proba(pd.DataFrame([values]))[0,1])
    st.metric("Estimated Team A win probability",f"{p:.1%}")
    st.progress(p)
    st.caption("A model estimate under hypothetical inputs—not a betting recommendation or guaranteed outcome.")
with tabs[3]:
    st.dataframe(games.head(100),width="stretch")
    st.markdown("""### Limitations
Historical models can overfit, leak future information, miss injuries and lineup changes, and degrade as teams and rules change. Chronological splitting reduces one form of leakage but does not solve selection bias or distribution shift. The included demo has simulated relationships and exists only to make the workflow runnable; upload a properly licensed historical dataset for substantive study.""")

