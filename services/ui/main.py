import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# API_URL = "http://localhost:8000"
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Music Recommendation MLOps",
    page_icon="🎵",
    layout="wide",
)


def api_get(endpoint: str, params: dict | None = None) -> dict | list | None:
    try:
        response = requests.get(f"{API_URL}{endpoint}", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        st.error(f"API error {response.status_code}: {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_URL}")
        return None
    except Exception as e:
        st.error(f"Request error: {e}")
        return None


def api_post(endpoint: str, params: dict | None = None) -> dict | None:
    try:
        response = requests.post(f"{API_URL}{endpoint}", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        st.error(f"API error {response.status_code}: {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_URL}")
        return None


st.sidebar.title("🎵 Music Rec MLOps")
page = st.sidebar.radio(
    "Navigation",
    ["🔍 Inference", "📋 Predictions", "📊 Experiments", "🚨 Drift & Monitoring"],
)

health = api_get("/health")
if health:
    status_color = "🟢" if health.get("model_ready") else "🔴"
    st.sidebar.markdown(f"{status_color} API: **{health.get('status', 'unknown')}**")
    st.sidebar.markdown(f"Model: **{health.get('model_version', 'unknown')}**")
else:
    st.sidebar.markdown("🔴 API: **offline**")


if page == "🔍 Inference":
    st.title("🔍 Inference — Get Recommendations")

    col1, col2 = st.columns([3, 1])
    with col1:
        user_id = st.text_input("User ID", placeholder="Enter user_id...")
    with col2:
        n_items = st.number_input(
            "N recommendations", min_value=1, max_value=100, value=10
        )

    if st.button("Get Recommendations", type="primary"):
        if not user_id:
            st.warning("Please enter a User ID")
        else:
            with st.spinner("Getting recommendations..."):
                result = api_get(
                    f"/recommendations/{user_id}",
                    params={"n_items": n_items},
                )

            if result:
                st.success(f"✅ Got {len(result['recommendations'])} recommendations")
                st.caption(
                    f"Timestamp: {result['timestamp']} |"
                    f"Model: {result['model_version']}"
                )

                rec_df = pd.DataFrame(
                    {
                        "Rank": range(1, len(result["recommendations"]) + 1),
                        "Song ID": result["recommendations"],
                        "Score": [round(s, 4) for s in result["scores"]],
                    }
                )
                st.dataframe(rec_df, use_container_width=True, hide_index=True)

                fig = px.bar(
                    rec_df,
                    x="Rank",
                    y="Score",
                    hover_data=["Song ID"],
                    title="Recommendation Scores",
                    color="Score",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig, use_container_width=True)


elif page == "📋 Predictions":
    st.title("📋 Latest Predictions")

    col1, col2 = st.columns([1, 4])
    with col1:
        limit = st.number_input("Show last N", min_value=5, max_value=500, value=50)

    if st.button("Refresh", type="secondary"):
        st.rerun()

    predictions = api_get("/recommendations/history/latest", params={"limit": limit})

    if predictions:
        df = pd.DataFrame(predictions)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total shown", len(df))
        m2.metric("Anomalies", df["is_anomaly"].sum())
        m3.metric(
            "Anomaly rate",
            f"{df['is_anomaly'].mean() * 100:.1f}%",
        )
        m4.metric("Unique users", df["user_id"].nunique())

        st.markdown("---")

        display_df = df[
            [
                "id",
                "timestamp",
                "user_id",
                "model_version",
                "n_recommendations",
                "is_anomaly",
                "anomaly_reason",
            ]
        ].copy()

        def highlight_anomalies(row):
            if row["is_anomaly"]:
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        styled = display_df.style.apply(highlight_anomalies, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.floor("h")
        hourly = df.groupby("hour")["is_anomaly"].agg(["sum", "count"]).reset_index()
        hourly.columns = ["hour", "anomalies", "total"]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=hourly["hour"],
                y=hourly["total"],
                name="Total",
                marker_color="lightblue",
            )
        )
        fig.add_trace(
            go.Bar(
                x=hourly["hour"],
                y=hourly["anomalies"],
                name="Anomalies",
                marker_color="red",
            )
        )
        fig.update_layout(title="Predictions & Anomalies by Hour", barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)


elif page == "📊 Experiments":
    st.title("📊 MLflow Experiments")

    metrics = api_get("/recommendations/metrics/current")
    if metrics:
        st.subheader("Current Model Metrics")

        ks = [10, 20, 50, 100]
        metric_names = ["ndcg", "hit_rate", "mrr", "precision", "recall", "map"]

        tabs = st.tabs([f"K={k}" for k in ks])
        for tab, k in zip(tabs, ks):
            with tab:
                cols = st.columns(len(metric_names))
                for col, metric_name in zip(cols, metric_names):
                    key = f"als_test_{metric_name}@{k}"
                    val = metrics.get(key, None)
                    if val is not None:
                        col.metric(metric_name.upper(), f"{val:.4f}")

        st.markdown("---")

        st.subheader("ALS vs Baselines (Test set, K=10)")
        comparison_data = []
        for model_prefix in ["als_test", "popular_test", "random_test"]:
            for metric in metric_names:
                key = f"{model_prefix}_{metric}@10"
                val = metrics.get(key)
                if val is not None:
                    comparison_data.append(
                        {
                            "Model": model_prefix.replace("_test", "").upper(),
                            "Metric": metric.upper(),
                            "Value": val,
                        }
                    )

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            fig = px.bar(
                comp_df,
                x="Metric",
                y="Value",
                color="Model",
                barmode="group",
                title="Model Comparison @ K=10",
            )
            st.plotly_chart(fig, use_container_width=True)

    # История переобучений
    st.subheader("Retraining History")
    retrain_history = api_get("/retrain/history", params={"limit": 10})
    if retrain_history:
        retrain_df = pd.DataFrame(retrain_history)
        st.dataframe(retrain_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("🔗 [Open MLflow UI](http://localhost:5000)")


elif page == "🚨 Drift & Monitoring":
    st.title("🚨 Drift & Monitoring")

    st.subheader("🔄 Model Retraining")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Trigger Retraining", type="primary"):
            with st.spinner("Triggering retraining..."):
                result = api_post("/retrain/", params={"triggered_by": "manual_ui"})
            if result:
                st.success(f"✅ Retraining started! Job ID: {result['job_id']}")
                st.info(result["message"])

    with col2:
        retrain_status = api_get("/retrain/status/latest")
        if retrain_status:
            status = retrain_status.get("status", "unknown")
            color_map = {
                "success": "🟢",
                "running": "🟡",
                "failed": "🔴",
                "pending": "🟡",
                "no_jobs": "⚪",
            }
            icon = color_map.get(status, "⚪")
            st.markdown(f"**Last job status:** {icon} {status.upper()}")
            if retrain_status.get("error_message"):
                st.error(f"Error: {retrain_status['error_message']}")

    st.markdown("---")

    st.subheader("📡 Drift Status")
    drift_status = api_get("/drift/status")

    if drift_status:
        if drift_status.get("status") == "no_data":
            st.info("ℹ️ No drift reports yet. Make some predictions first.")
        else:
            col1, col2, col3 = st.columns(3)
            is_drift = drift_status.get("is_drift_detected", False)
            drift_score = drift_status.get("drift_score", 0.0)

            col1.metric(
                "Drift Detected",
                "🚨 YES" if is_drift else "✅ NO",
            )
            col2.metric("Drift Score", f"{drift_score:.3f}")
            col3.metric("Drift Type", drift_status.get("drift_type", "unknown"))

            if is_drift:
                st.warning("⚠️ Drift detected! Consider retraining the model.")

    st.markdown("---")

    st.subheader("📜 Drift Reports History")
    reports = api_get("/drift/reports", params={"limit": 20})
    if reports:
        reports_df = pd.DataFrame(reports)
        if not reports_df.empty:
            # График дрейфа по времени
            reports_df["timestamp"] = pd.to_datetime(reports_df["timestamp"])
            fig = px.line(
                reports_df,
                x="timestamp",
                y="drift_score",
                color="drift_type",
                title="Drift Score Over Time",
                markers=True,
            )
            fig.add_hline(
                y=0.3, line_dash="dash", line_color="red", annotation_text="Threshold"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Таблица
            def highlight_drift(row):
                if row.get("is_drift_detected"):
                    return ["background-color: #ffcccc"] * len(row)
                return [""] * len(row)

            styled_reports = reports_df.style.apply(highlight_drift, axis=1)
            st.dataframe(styled_reports, use_container_width=True, hide_index=True)
    else:
        st.info("No drift reports available yet.")

    st.markdown("---")
    st.markdown(
        "[Open Grafana](http://localhost:3000) | "
        "🔗 [Open Prometheus](http://localhost:9090)"
    )
