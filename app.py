import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage, AIMessage
import os
import base64
from fuzzysearch import find_near_matches
import requests
from dotenv import load_dotenv
from multi_agent import (
    metadata_agent_executor,
    analytics_executor,
    data_entry_executor,
    event_data_executor,
    tracker_data_executor,
    routing_decision
)
from agents.tools.faiss_search.embed_dhis2_faiss_metadata import run_embedding
from agents.tools.analytics_tools import get_all

load_dotenv()

# Load environment variables

DHIS2_BASE_URL = os.getenv("DHIS2_BASE_URL")
DHIS2_USERNAME = os.getenv("DHIS2_USERNAME")
DHIS2_PASSWORD = os.getenv("DHIS2_PASSWORD")

# ========== Helpers ==========
def generate_distinct_colors(n):
    return [f"hsl({i * 360 / n}, 70%, 50%)" for i in range(n)]



def get_data(url, doc_type, disaggregations, indicators, periods, org_units):
    flattened = []
    org_units_lookup = []
    indicator_string = ";".join(indicators)
    period_string = ";".join(periods)
    org_unit_string = ";".join(org_units)

    if len(org_units) > 0:
        get_the_org = get_all(
            endpoint="organisationUnits.json",
            key="organisationUnits",
            fields="id,name",
            # fields="id,code,name,parent,children,level,path,ancestors",
            filters={"id": f"in:[{','.join(org_units)}]"}
        )
        print("+++++ get_the_org ++++")
        print(get_the_org)

        for org in get_the_org:
            org_units_lookup.append({
                "org": org["name"],
                "org_id": org["id"],
            })
    print("+++ org_units +++")
    print(org_units)
    print("+++ org_units_lookup +++")
    print(org_units_lookup)


    dimensions = [
        f"dx:{indicator_string}",
        f"pe:{period_string}",
        f"ou:{org_unit_string}"
    ]
    if doc_type == "indicator":
        include_num_den = True
        include_coc_dimension = False
    elif doc_type == "dataElement":
        get_the_de = get_all(
            endpoint="dataElements.json",
            key="dataElements",
            fields="id,name,categoryCombo[id,name,categories[id,name,categoryOptions[id,name]]]",
            # fields="id,code,name,parent,children,level,path,ancestors",
            filters={"id": f"in:[{','.join(indicators)}]"}
        )

        for de in get_the_de:
            categories = de.get("categoryCombo", {}).get("categories", {})
            # Flatten the structure
            for category in categories:
                for option_ in category["categoryOptions"]:
                    flattened.append({
                        "dataElement": de["name"],
                        "dataElement_id": de["id"],
                        "category": category["name"],
                        "category_id": category["id"],
                        "option": option_["name"],
                        "option_id": option_["id"]
                    })
        print("+++++ flattened +++++")
        print(flattened)


        # After flattening is done
        unique_categories = sorted({item["category"] for item in flattened})
        print("✅ Unique categories found:")
        for cat in unique_categories:
            print(f"- {cat}")

        # Filter flattened items whose category name matches any disaggregation
        if disaggregations is not None:
            filtered_flattened= []
            for item in flattened:
                category_name = item["category"].lower().strip()
                for disaggregation in disaggregations:
                    matches = find_near_matches(disaggregation.lower().strip(), category_name, max_l_dist=2)
                    if matches:
                        filtered_flattened.append(item)
                        break  # Stop after the first match for this item
            # Build dictionary of {category_id: [option_id, ...]}
            disaggregations_dict = {}
            for item in filtered_flattened:
                cat_id = item["category_id"]
                opt_id = item["option_id"]
                disaggregations_dict.setdefault(cat_id, []).append(opt_id)


            # Construct dimensions list
            dimensions = [
                f"dx:{indicator_string}",
                f"pe:{period_string}",
                f"ou:{org_unit_string}"
            ]

            # Add disaggregation dimensions
            for cat_id, option_ids in disaggregations_dict.items():
                option_string = ";".join(sorted(set(option_ids)))  # Optional: deduplicate + sort
                dimensions.append(f"{cat_id}:{option_string}")

            dimensions.append("co")  # ⚠️ Only add this if dx supports category option combos


        include_num_den = False
        include_coc_dimension = True  # Often useful for DEs

        # Add co dimension explicitly if requested
        # if include_coc_dimension:
        #     dimensions.append("co")  # ⚠️ Only add this if dx supports category option combos


    elif doc_type == "programIndicator":
        include_num_den = False
        include_coc_dimension = True  # Often useful for DEs
        pass
    else:
        include_num_den = False
        include_coc_dimension = True  # Often useful for DEs
        pass
    params = {
        "dimension": dimensions,
        # "displayProperty": display_property,
        "includeNumDen": str(include_num_den).lower(),
        "skipMeta": "false",
        "skipData": "false"
    }
    url = f"{DHIS2_BASE_URL.rstrip('/')}/api/analytics"
    try:
        response = requests.get(
            url=url,
            auth=(DHIS2_USERNAME, DHIS2_PASSWORD),
            params=params
        )
        response.raise_for_status()
        return {
            "meta_data": response.json(),
            "org_units_lookup": org_units_lookup,
            "flattened": flattened,
        }
    except Exception as e:
        return {"error": str(e)}

def analytics_to_dataframe(tool_data: dict):
    if "data" not in tool_data:
        return pd.DataFrame()
    url= tool_data["url"]
    doc_type = tool_data["doc_type"]
    disaggregations = tool_data["disaggregations"]
    indicators = tool_data["indicators"]
    periods = tool_data["periods"]
    org_units = tool_data["org_units"]
    url = url.replace("skipMeta=true", "skipMeta=false")
    meta_data = get_data(url, doc_type, disaggregations, indicators, periods, org_units)
    if isinstance(meta_data, str):
        meta_data = json.loads(meta_data)

    # Now access it safely
    data = meta_data.get("meta_data", {})
    org_units_lookup = meta_data.get("org_units_lookup", {}),
    flattened = meta_data.get("flattened", {}),


    headers = data.get("headers", [])
    rows = data.get("rows", [])
    columns = [header["name"] for header in headers]
    return org_units_lookup, flattened, pd.DataFrame(rows, columns=columns)


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
    # # Handle optional category option combo column
    # if "co" in df.columns:
    #     df = df.rename(columns={"co": "Category option combo"})

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    grouped = df[df["dx"].isin(selected_indicators)]
    # Define available label columns
    available_label_columns = ["pe", "ou"] + [col for col in df.columns if col.startswith("co_")]

    # Ensure "pe" is the default (index=0)
    label_column = st.selectbox("Choose x-axis (label) dimension:", options=available_label_columns,
                                index=available_label_columns.index("pe"))

    labels = sorted(grouped[label_column].dropna().unique().tolist())
    colors = ["#4285F4", "#DB4437", "#F4B400", "#0F9D58", "#AB47BC"]
    datasets = []

    for idx, indicator in enumerate(selected_indicators):
        subset = grouped[grouped["dx"] == indicator]
        subset_grouped = subset.groupby(label_column)["value"].sum().reindex(labels).fillna(0)
        values = subset_grouped.tolist()
        dynamic_colors = generate_distinct_colors(len(values))
        if chart_type == "pie":
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
                "backgroundColor": dynamic_colors,
                "borderColor": dynamic_colors,
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
    print('+++++ raw_data ++++++')
    print(raw_data)

    if raw_data is None:
        st.warning("No visualization data returned from the analytics executor.")
        return
    org_units_lookup, flattened, df_chart_data = analytics_to_dataframe(raw_data)

    st.session_state.raw_data_df = df_chart_data

    raw_data_df = df_chart_data

    # Base required columns
    base_required_cols = {"dx", "pe", "value"}
    # Columns to exclude (known standard fields)
    excluded_cols = {"dx", "pe", "ou", "value", "numerator", "denominator", "factor", "multiplier", "divisor", "co"}
    # Detect up to 3 disaggregation columns (e.g., category option combos)
    disagg_cols = [col for col in raw_data_df.columns if col not in excluded_cols][:3]

    # Add disaggregation columns to required cols
    required_cols = base_required_cols.union(disagg_cols)

    if not raw_data_df.empty and required_cols.issubset(raw_data_df.columns):
        with st.container():
            # If org_units_lookup or flattened are strings, convert them to Python objects
            if isinstance(org_units_lookup, str):
                org_units_lookup = json.loads(org_units_lookup)

            if isinstance(flattened, str):
                flattened = json.loads(flattened)

            # If they are a tuple (e.g., result of a return statement like `return x,`), unwrap them
            if isinstance(org_units_lookup, tuple):
                org_units_lookup = org_units_lookup[0]

            if isinstance(flattened, tuple):
                flattened = flattened[0]

            org_map = {item['org_id']: item['org'] for item in org_units_lookup}
            dx_map = {item['dataElement_id']: item['dataElement'] for item in flattened}
            option_map = {item['option_id']: item['option'] for item in flattened}
            category_map = {item['category_id']: item['category'] for item in flattened}

            # --- Step 2: Replace IDs with names in DataFrame (only if columns exist) ---
            def replace_if_exists(df: pd.DataFrame, column: str, mapping: dict):
                if column in df.columns:
                    df[column] = df[column].map(mapping).fillna(df[column])

            # Assume filtered_df is already defined
            # Replace `ou` using org_map
            replace_if_exists(raw_data_df, 'ou', org_map)

            # Replace `dx` using dx_map
            replace_if_exists(raw_data_df, 'dx', dx_map)

            # Replace `co` using option_map
            replace_if_exists(raw_data_df, 'co', option_map)


            # Replace `co_1`, `co_2`, ..., using category_map
            print("++++ disagg_cols +++++")
            if len(disagg_cols)>0:
                for col in disagg_cols:
                    replace_if_exists(raw_data_df, col, option_map)

            print(raw_data_df)



            indicators = raw_data_df["dx"].dropna().unique().tolist()
            periods = sorted(raw_data_df["pe"].dropna().unique().tolist())
            try:
                org_units = (
                    raw_data_df["ou"].dropna().unique().tolist()
                    if "ou" in raw_data_df.columns else raw_data_df["Organisation unit"].dropna().unique().tolist()

                )
            except KeyError:
                org_units = []

            # Rename disaggregation columns to co_1, co_2, etc.
            co_col_map = {}  # e.g. {'co_1': 'cX5k9anHEHd'}
            for idx, col in enumerate(disagg_cols):
                readable_name = category_map.get(col, col)  # Fallback to col if not found
                co_name = f"co_{idx + 1} {readable_name}"
                raw_data_df.rename(columns={col: co_name}, inplace=True)
                co_col_map[co_name] = col

            # Filters


            selected_indicators = st.multiselect("Select indicator(s)", indicators, default=indicators[:1])
            if len(org_units) > 0:
                selected_org_units = st.multiselect("Select org unit(s)", org_units, default=org_units)
            selected_periods = st.multiselect("Select period(s)", periods, default=periods)

            # Multiselect filters for each co_*
            selected_cos = {}
            for co_col in co_col_map:
                co_values = raw_data_df[co_col].dropna().unique().tolist()
                selected_ = st.multiselect(f"Filter by category option: {co_col}", co_values, default=co_values)
                selected_cos[co_col] = selected_

            chart_type = st.selectbox("Select chart type", ["bar", "line", "pie"])

        # Apply all filters
        if len(org_units) > 0:
            raw_data_df.to_csv("filtered_df_6.csv")
            filtered_df = raw_data_df[
                raw_data_df["dx"].isin(selected_indicators) &
                raw_data_df["pe"].isin(selected_periods)
            ]
            # & raw_data_df[raw_data_df.columns[2]].isin(selected_org_units)

        else:
            raw_data_df.to_csv("filtered_df_7.csv")
            filtered_df = raw_data_df[
                raw_data_df["dx"].isin(selected_indicators) &
                raw_data_df["pe"].isin(selected_periods)
            ]
        filtered_df.to_csv("filtered_df_8.csv")
        # Apply co_* filters
        for co_col, selected_vals in selected_cos.items():
            if selected_vals:
                filtered_df = filtered_df[filtered_df[co_col].isin(selected_vals)]

        filtered_df.to_csv("filtered_df_9.csv")

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
# ========== Side Bar ==========
# ===== Initialize session state =====
if "side_bar_open" not in st.session_state:
    st.session_state.side_bar_open = False
if "side_bar_open_button" not in st.session_state:
    st.session_state.side_bar_open_button = True

if st.session_state.get("side_bar_open_button"):
    if st.button("🛠️"):
        st.session_state.side_bar_open = True

if st.session_state.get("side_bar_open"):
    # Simulate a menu using the sidebar
    st.session_state.side_bar_open_button = False
    with st.sidebar:
        st.header("Settings")
        # if st.button("▶ Run Script"):
        #     # result = run_my_code()
        #     st.success("Button pressed")
        with st.expander("⚙️ Advanced Options"):
            if st.button("Run Metadata Embedder"):
                with st.spinner("Embedding and indexing metadata..."):
                    run_embedding()
                st.success("Metadata embedded.")



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

# ========== Side bar =============

# # Simulate a menu using the sidebar
# with st.sidebar:
#     st.header("Settings")
#     # if st.button("▶ Run Script"):
#     #     # result = run_my_code()
#     #     st.success("Button pressed")
#     with st.expander("⚙️ Advanced Options"):
#         if st.button("Run Metadata Embedder"):
#             with st.spinner("Embedding and indexing metadata..."):
#                 run_embedding()
#             st.success("Metadata embedded.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
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


