import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage, AIMessage
from multi_agent import (
    metadata_agent_executor,
    analytics_executor,
    data_entry_executor,
    event_data_executor,
    tracker_data_executor,
    routing_decision
)

# ========== Helpers ==========
def generate_distinct_colors(n):
    return [f"hsl({i * 360 / n}, 70%, 50%)" for i in range(n)]

def analytics_to_dataframe(tool_data: dict) -> pd.DataFrame:
    if "data" not in tool_data:
        return pd.DataFrame()
    data = tool_data["data"]
    headers = data.get("headers", [])
    rows = data.get("rows", [])
    columns = [header["name"] for header in headers]
    return pd.DataFrame(rows, columns=columns)


import base64


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

        if chart_type == "pie":
            dynamic_colors = generate_distinct_colors(len(values))
            datasets.append({
                "label": indicator,
                "data": values,
                "backgroundColor": dynamic_colors,
                "borderColor": "#fff",
                "borderWidth": 1
            })
        else:
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
                "title": {"display": True, "text": "DHIS2 Analytics Chart"},
                "datalabels": {
                    "anchor": "end",
                    "align": "top",
                    "color": "#000",
                    "font": {"weight": "bold"},
                    "formatter": "Math.round"
                }
            },
            "scales": {
                "y": {
                    "beginAtZero": True
                }
            }
        },
        "plugins": ["ChartDataLabels"]
    }

    chart_json = json.dumps(chart_config)

    # Encode CSV data as base64 for safe injection
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    csv_b64 = base64.b64encode(csv_bytes).decode("utf-8")

    html_code = f"""
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>

    <div style="position: relative; height: 400px;">
        <canvas id="myChart"></canvas>
    </div>

    <div style="margin-top: 10px;">
        <button onclick="downloadChart()" style="padding: 8px 16px; font-size: 14px; margin-right: 10px;">
            📥 Download Chart as PNG
        </button>
        <button onclick="downloadCSV()" style="padding: 8px 16px; font-size: 14px;">
            📥 Download Data as CSV
        </button>
    </div>

    <script>
    const ctx = document.getElementById('myChart').getContext('2d');
    Chart.register(ChartDataLabels);
    new Chart(ctx, {chart_json});

    function downloadChart() {{
        const link = document.createElement('a');
        link.download = 'chart.png';
        link.href = document.getElementById('myChart').toDataURL('image/png');
        link.click();
    }}

    function downloadCSV() {{
        const csvData = atob("{csv_b64}");
        const blob = new Blob([csvData], {{ type: 'text/csv;charset=utf-8;' }});
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', 'filtered_data.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }}
    </script>
    """

    st.subheader("📊 Chart Preview")
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
    if raw_data is None:
        st.warning("No visualization data returned from the analytics executor.")
        return
    df_chart_data = analytics_to_dataframe(raw_data)
    st.session_state.raw_data_df = df_chart_data
    raw_data_df = df_chart_data

    required_cols = {"pe", "value", "dx"}

    # ✅ Check if required columns are present
    if not raw_data_df.empty and required_cols.issubset(raw_data_df.columns):
        with st.container():
            # ✅ Build filter options
            indicators = raw_data_df["dx"].dropna().unique().tolist()
            periods = sorted(raw_data_df["pe"].dropna().unique().tolist())
            org_units = raw_data_df["ou"].dropna().unique().tolist() if "ou" in raw_data_df.columns else raw_data_df["Organisation unit"].dropna().unique().tolist()

            # ✅ UI filters
            selected_indicators = st.multiselect("Select indicator(s)", indicators, default=indicators[:1])
            selected_org_units = st.multiselect("Select org unit(s)", org_units, default=org_units)
            selected_periods = st.multiselect("Select period(s)", periods, default=periods)

            chart_type = st.selectbox("Select chart type", ["bar", "line", "pie"])

        # ✅ Apply filters to the dataframe
        filtered_df = raw_data_df[
            raw_data_df["dx"].isin(selected_indicators) &
            raw_data_df[raw_data_df.columns[2]].isin(selected_org_units) &  # Assuming "Organisation unit" is 3rd column
            raw_data_df["pe"].isin(selected_periods)
        ]


        if selected_indicators and not filtered_df.empty:
            if chart_backend == "chartjs":
                render_chart_chartjs(filtered_df, selected_indicators, chart_type)
            elif chart_backend == "matplotlib":
                render_chart_matplotlib(filtered_df, selected_indicators, chart_type)
            elif chart_backend == "plotly":
                render_chart_plotly(filtered_df, selected_indicators, chart_type)
        else:
            st.info("No data for the selected filters.")
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

if "show_suggestion_options" not in st.session_state:
    st.session_state.show_suggestion_options = False

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
        metadata_result = result.get("metadata_result", "")


        if metadata_result and metadata_result.get("status") == "multiple_matches":
            suggestions = metadata_result["suggestions"]
            st.session_state.suggestions = suggestions
            st.session_state.show_suggestion_options = True


        else:
            st.session_state.result = result
            st.session_state.show_chart = True


    elif route == "data_entry":
        result = data_entry_executor.invoke(state)
    elif route == "event_data":
        result = event_data_executor.invoke(state)
    elif route == "tracker_data":
        result = tracker_data_executor.invoke(state)
    else:
        error_msg = AIMessage(content="❌ Unknown routing decision.")
        st.session_state.messages.append(error_msg)
        st.chat_message("assistant").markdown(error_msg.content)
        st.stop()

    if len(st.session_state.suggestions) < 1:
        output = result.get("output")
        assistant_msg = output if isinstance(output, AIMessage) else AIMessage(content=str(output))
        st.session_state.messages.append(assistant_msg)
        with st.chat_message("assistant"):
            st.markdown(assistant_msg.content)


if st.session_state.get("show_chart", False):
    chart_data(st.session_state.result, "chartjs")

# Show metadata options after user prompt
if st.session_state.get("show_suggestion_options", False):
    with st.chat_message("assistant"):
        st.markdown("There are multiple metadata matches. Please choose one:")

        for i, option in enumerate(st.session_state.suggestions):
            label = f"{option['name']} ({option['doc_type']})"
            if st.button(label, key=f"metadata_option_{i}"):
                st.session_state.selected_metadata = option  # Store full dict (id, name, doc_type, score)
                st.session_state.trigger_metadata_retry = True  # 🔁 trigger new processing
                st.session_state.show_suggestion_options = False
                st.rerun()

if st.session_state.get("trigger_metadata_retry"):
    selected = st.session_state.selected_metadata

    # Reconstruct the last actual user question
    original_user_msg = ""
    for msg in reversed(st.session_state.messages):
        if isinstance(msg, HumanMessage):
            original_user_msg = msg.content
            break

    # Build a clearer retry message with metadata reference
    augmented_msg = HumanMessage(
        # content=f"{original_user_msg} (selected: {selected['id']} - {selected['name']})"
        content = f"{selected['doc_type']} selected:  {selected['name']} -  {selected['id']})"

    )

    st.session_state.messages.append(augmented_msg)

    with st.chat_message("user"):
        st.markdown(original_user_msg)  # ✅ Show only original question in UI

    # Re-invoke agent
    state = {"messages": st.session_state.messages}
    result = analytics_executor.invoke(state)

    st.session_state.result = result
    st.session_state.show_chart = True
    st.session_state.trigger_metadata_retry = False

    output = result.get("output")
    assistant_msg = output if isinstance(output, AIMessage) else AIMessage(content=str(output))
    st.session_state.messages.append(assistant_msg)

    with st.chat_message("assistant"):
        st.markdown(assistant_msg.content)

    st.rerun()  # ✅ Force chart to render


