import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage, AIMessage
from multi_agent import metadata_agent_executor, analytics_executor, routing_decision

# ========== Helpers ==========

def analytics_to_dataframe(tool_data: dict) -> pd.DataFrame:
    if "data" not in tool_data:
        return pd.DataFrame()
    data = tool_data["data"]
    headers = data.get("headers", [])
    rows = data.get("rows", [])
    columns = [header["name"] for header in headers]
    return pd.DataFrame(rows, columns=columns)


def render_chart_chartjs(df: pd.DataFrame, selected_indicators: list, chart_type: str):
    if df.empty or not selected_indicators:
        st.info("No data available or indicators selected.")
        return

    df = df.rename(columns={
        "Data": "dx",
        "Period": "pe",
        "Organisation unit": "ou",
        "Value": "value"
    })
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    grouped = df[df["dx"].isin(selected_indicators)]

    labels = sorted(grouped["pe"].unique().tolist())
    colors = ["#4285F4", "#DB4437", "#F4B400", "#0F9D58", "#AB47BC"]
    datasets = []

    for idx, indicator in enumerate(selected_indicators):
        subset = grouped[grouped["dx"] == indicator]
        subset_grouped = subset.groupby("pe")["value"].sum().reindex(labels).fillna(0)
        values = subset_grouped.tolist()

        datasets.append({
            "label": indicator,
            "data": values,
            "backgroundColor": colors[idx % len(colors)],
            "borderColor": colors[idx % len(colors)],
            "fill": chart_type != "line",
            "tension": 0.3
        })

    chart_config = {
        "type": chart_type,
        "data": {
            "labels": labels,
            "datasets": datasets
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {"position": "top"},
                "title": {"display": True, "text": "DHIS2 Analytics Chart"}
            }
        }
    }

    chart_json = json.dumps(chart_config)
    html_code = f"""
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <canvas id="myChart" height="400"></canvas>
    <script>
    const ctx = document.getElementById('myChart').getContext('2d');
    new Chart(ctx, {chart_json});
    </script>
    """
    st.subheader("📊 Chart Preview (Chart.js)")
    components.html(html_code, height=500)


def render_chart_matplotlib(df: pd.DataFrame, selected_indicators: list, chart_type: str):
    if df.empty or not selected_indicators:
        st.info("No data available or indicators selected.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    grouped = df[df["dx"].isin(selected_indicators)]

    if chart_type == "bar":
        for indicator in selected_indicators:
            subset = grouped[grouped["dx"] == indicator]
            ax.bar(subset["pe"], subset["value"], label=indicator)
    elif chart_type == "line":
        for indicator in selected_indicators:
            subset = grouped[grouped["dx"] == indicator]
            ax.plot(subset["pe"], subset["value"], label=indicator, marker='o')
    elif chart_type == "pie" and len(selected_indicators) == 1:
        subset = grouped[grouped["dx"] == selected_indicators[0]]
        subset = subset.groupby("pe")["value"].sum()
        ax.pie(subset.values, labels=subset.index, autopct='%1.1f%%')
    else:
        st.warning("Pie chart only supports one indicator.")
        return

    if chart_type != "pie":
        ax.set_xlabel("Period")
        ax.set_ylabel("Value")
        ax.set_title("DHIS2 Analytics Chart (Matplotlib)")
        ax.legend()

    st.subheader("📊 Chart Preview (Matplotlib)")
    st.pyplot(fig)


def render_chart_plotly(df: pd.DataFrame, selected_indicators: list, chart_type: str):
    if df.empty or not selected_indicators:
        st.info("No data available or indicators selected.")
        return

    filtered_df = df[df["dx"].isin(selected_indicators)]

    if chart_type == "bar":
        fig = px.bar(filtered_df, x="pe", y="value", color="dx", barmode="group", title="DHIS2 Analytics Chart (Plotly)")
    elif chart_type == "line":
        fig = px.line(filtered_df, x="pe", y="value", color="dx", markers=True, title="DHIS2 Analytics Chart (Plotly)")
    elif chart_type == "pie" and len(selected_indicators) == 1:
        subset = filtered_df[filtered_df["dx"] == selected_indicators[0]]
        pie_data = subset.groupby("pe")["value"].sum().reset_index()
        fig = px.pie(pie_data, names="pe", values="value", title=f"{selected_indicators[0]} (Pie Chart)")
    else:
        st.warning("Pie chart only supports one indicator.")
        return

    st.subheader("📊 Chart Preview (Plotly)")
    st.plotly_chart(fig, use_container_width=True)


def chart_data(result_, chart_backend):
    raw_data = result_.get("raw_data", {})

    # Always overwrite with real data
    df_chart_data = analytics_to_dataframe(raw_data)
    st.session_state.raw_data_df = df_chart_data
    raw_data_df = df_chart_data

    required_cols = {"pe", "value", "dx"}
    if not raw_data_df.empty and required_cols.issubset(raw_data_df.columns):
        with st.container():
            indicators = raw_data_df["dx"].dropna().unique().tolist()
            selected_indicators = st.multiselect("Select indicator(s)", indicators, default=indicators[:1])
            chart_type = st.selectbox("Select chart type", ["bar", "line", "pie"])

        if selected_indicators:
            if chart_backend == "chartjs":
                render_chart_chartjs(raw_data_df, selected_indicators, chart_type)
            elif chart_backend == "matplotlib":
                render_chart_matplotlib(raw_data_df, selected_indicators, chart_type)
            elif chart_backend == "plotly":
                render_chart_plotly(raw_data_df, selected_indicators, chart_type)
        else:
            st.info("Please select at least one indicator to display chart.")
    else:
        st.warning("No valid data or required columns missing.")
# ========== App UI ==========

st.set_page_config(page_title="DHIS2 Assistant", layout="wide")

col1, col2 = st.columns([1, 5])
with col1:
    st.image("fhi360.png")
with col2:
    st.title("🗨️ DHIS2 Chat Assistant")

upload_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls"])
if upload_file is not None:
    if upload_file.name.endswith(".csv"):
        df = pd.read_csv(upload_file)
    else:
        df = pd.read_excel(upload_file)
    st.session_state.raw_data_df = df
    st.write("✅ File uploaded:", upload_file.name)
    st.write(df)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_chart" not in st.session_state:
    st.session_state.show_chart = False

for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Ask DHIS2 Assistant..."):
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    state = {"messages": st.session_state.messages}
    route = routing_decision(state)

    if route == "metadata":
        result = metadata_agent_executor.invoke(state)
    elif route == "analytics":
        result = analytics_executor.invoke(state)
        st.session_state.result = result  # Store result in session state
        st.session_state.show_chart = True
    else:
        error_msg = AIMessage(content="❌ Unknown routing decision.")
        st.session_state.messages.append(error_msg)
        st.chat_message("assistant").markdown(error_msg.content)
        st.stop()

    output = result.get("output")
    assistant_msg = output if isinstance(output, AIMessage) else AIMessage(content=str(output))
    st.session_state.messages.append(assistant_msg)
    with st.chat_message("assistant"):
        st.markdown(assistant_msg.content)

if st.session_state.get("show_chart", False):
    chart_data(st.session_state.result, "chartjs")