"""
RCA Automation - Streamlit UI
Step 1: Hub-wise Root Cause Analysis from RL% drivers.
Step 2: Sub-reasons analysis for Pro Discarded driver.
"""

from io import StringIO
from typing import Optional

import pandas as pd
import streamlit as st

from rca_step1 import add_rca_column
from rca_step2 import analyze_hubs, format_sub_reasons_output, detect_data_format
from rca_step3 import analyze_hubs_services, format_services_output
from rca_step4 import bucketize_statements, bucketize_statement

st.set_page_config(
    page_title="RCA Automation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced styling for a clean, modern, data-focused UI
st.markdown("""
<style>
    /* Main app container (back to simple default background) */
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Sidebar - keep Streamlit default styling */
    [data-testid="stSidebar"] {
    }
    
    [data-testid="stSidebar"] [data-baseweb="select"] {
    }
    
    /* Headers */
    h1 {
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #334155;
        font-weight: 600;
    }
    
    /* Cards and blocks */
    .rca-report-block { 
        background: linear-gradient(90deg, #f8fafc 0%, #ffffff 100%);
        border-left: 4px solid #0ea5e9; 
        padding: 1rem 1.25rem; 
        margin: 0.75rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    
    .rca-report-block:hover {
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transform: translateX(2px);
    }
    
    .rca-hub { 
        font-weight: 600; 
        color: #0f172a;
        font-size: 1rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .rca-text { 
        color: #475569; 
        margin-top: 0.25rem;
        line-height: 1.6;
    }
    
    .sub-reason-block {
        background: linear-gradient(90deg, #f0f9ff 0%, #ffffff 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    
    .sub-reason-block:hover {
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transform: translateX(2px);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #0ea5e9;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1);
    }
    
    /* Select boxes */
    [data-baseweb="select"] {
        border-radius: 8px;
    }
    
    /* Success/Info messages */
    .stSuccess {
        border-radius: 8px;
        border-left: 4px solid #10b981;
        padding: 1rem;
    }
    
    .stInfo {
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
    }
    
    .stWarning {
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
    }
    
    .stError {
        border-radius: 8px;
        border-left: 4px solid #ef4444;
    }
    
    /* Data tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Download button */
    div[data-testid="stDownloadButton"] { 
        margin-top: 1.5rem; 
    }
    
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(90deg, #0ea5e9 0%, #0284c7 100%);
        border: none;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Radio buttons */
    [data-baseweb="radio"] {
        margin-bottom: 0.5rem;
    }
    
    /* Sliders */
    [data-baseweb="slider"] {
        margin: 1rem 0;
    }
    
    /* Tabs */
    [data-baseweb="tabs"] {
        margin-top: 1rem;
    }
    
    /* Step indicator */
    .step-indicator {
        background: #f1f5f9;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-size: 0.875rem;
        color: #64748b;
    }
    
    /* Caption styling */
    .stCaption {
        color: #64748b;
        font-size: 0.875rem;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 8px;
        border: 2px dashed #cbd5e1;
        padding: 1rem;
        transition: all 0.2s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #0ea5e9;
        background: #f0f9ff;
    }
    
    /* Sidebar sections */
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_df_from_upload(file) -> Optional[pd.DataFrame]:
    if file is None:
        return None
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return None


def get_df_from_paste(text: str) -> Optional[pd.DataFrame]:
    if not (text and text.strip()):
        return None
    try:
        sep = "\t" if "\t" in text.split("\n")[0] else ","
        return pd.read_csv(StringIO(text.strip()), sep=sep)
    except Exception as e:
        st.error(f"Could not parse pasted data: {e}")
        return None


def format_rca_report(df: pd.DataFrame, hub_col: str) -> str:
    """Format RCA results as hub + RCA line per row."""
    lines = []
    for _, row in df.iterrows():
        lines.append(str(row[hub_col]))
        lines.append(str(row["RCA"]))
        lines.append("")
    return "\n".join(lines)


# --- UI ---

# Header with better visual hierarchy
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("### 📊")
with col2:
    st.title("RCA Automation")
    st.caption("Root Cause Analysis Tool for Request Loss")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Navigation")

# Step selector with icons
step = st.sidebar.radio(
    "Select Analysis Step",
    [
        "Step 1: Hub-wise RCA",
        "Step 2: Sub-reasons Analysis",
        "Step 3: Service Tags Analysis",
        "Step 4: Summary / Bucketization",
    ],
    index=0,
    help="Choose which RCA analysis step to perform",
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# --- STEP 1 UI ---
if step == "Step 1: Hub-wise RCA":
    st.markdown("### 🔍 Step 1: Hub-wise Root Cause Analysis")
    st.caption("Identify top 2 drivers per hub with rounded RL%")
    st.markdown("---")
    
    st.sidebar.markdown("### 📥 Data Input")
    input_mode = st.sidebar.radio(
        "How do you want to provide data?",
        ["Upload CSV file", "Paste data (from Excel / Google Sheets)"],
        label_visibility="collapsed",
    )
    
    df: Optional[pd.DataFrame] = None
    
    if input_mode == "Upload CSV file":
        uploaded = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a file with hub names and 3 driver columns (e.g. Pro Discarded, Supply Shortage, supply_shortage)",
        )
        df = get_df_from_upload(uploaded)
    else:
        st.sidebar.caption("Paste tab or comma-separated data below (e.g. copy from Excel or Google Sheets).")
        pasted = st.sidebar.text_area(
            "Paste your data",
            height=180,
            placeholder="hub_name_2\tPro Discarded\tSupply Shortage (Leave)\tsupply_shortage\nhub1\t1.6\t1.3\t6.7\n...",
            label_visibility="collapsed",
        )
        df = get_df_from_paste(pasted) if pasted else None
    
    if df is not None:
        df = _normalize_columns(df)
        cols = list(df.columns)
        
        st.sidebar.success(f"✅ Data loaded: {len(df)} rows, {len(cols)} columns")
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Column Mapping")
        
        hub_col = st.sidebar.selectbox(
            "Hub name column",
            options=cols,
            index=0,
            help="Column that contains hub / location names",
        )
        
        # Default driver order if these columns exist
        default_drivers = ["Pro Discarded", "Supply Shortage (Leave)", "supply_shortage"]
        driver_options = [c for c in cols if c != hub_col]
        idx1 = driver_options.index(default_drivers[0]) if default_drivers[0] in driver_options else 0
        d1 = st.sidebar.selectbox("Driver 1 (metric)", options=driver_options, index=idx1)
        remaining = [c for c in driver_options if c != d1]
        idx2 = remaining.index(default_drivers[1]) if default_drivers[1] in remaining else 0
        d2 = st.sidebar.selectbox("Driver 2 (metric)", options=remaining, index=min(idx2, len(remaining) - 1))
        remaining2 = [c for c in remaining if c != d2]
        idx3 = remaining2.index(default_drivers[2]) if default_drivers[2] in remaining2 else 0
        d3 = st.sidebar.selectbox("Driver 3 (metric)", options=remaining2, index=min(idx3, len(remaining2) - 1))
        
        driver_cols = [d1, d2, d3]
        
        st.sidebar.markdown("---")
        run_clicked = st.sidebar.button("🚀 Run RCA Analysis", type="primary", use_container_width=True)
        
        if run_clicked:
            if len(driver_cols) != 3:
                st.error("Please select exactly 3 driver columns.")
            else:
                try:
                    result = add_rca_column(df, hub_col, driver_cols, None)
                    st.session_state["step1_result"] = result
                    st.session_state["step1_hub_col"] = hub_col
                except Exception as e:
                    st.error(str(e))
    
    # Show Step 1 results
    if st.session_state.get("step1_result") is not None:
        result = st.session_state["step1_result"]
        hub_col = st.session_state.get("step1_hub_col", result.columns[0])
        
        st.success("✅ RCA analysis completed successfully! Review the results below.")
        
        tab_table, tab_report = st.tabs(["Results table", "Report view"])
        
        with tab_table:
            st.dataframe(result, use_container_width=True, hide_index=True)
        
        with tab_report:
            report_text = format_rca_report(result, hub_col)
            st.text_area("RCA report (copy-friendly)", report_text, height=320, disabled=True)
            for _, row in result.iterrows():
                st.markdown(
                    f'<div class="rca-report-block"><span class="rca-hub">{row[hub_col]}</span><br><span class="rca-text">{row["RCA"]}</span></div>',
                    unsafe_allow_html=True,
                )
        
        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download result as CSV",
            data=csv_bytes,
            file_name="rca_step1_result.csv",
            mime="text/csv",
        )
    else:
        if df is not None:
            st.info("💡 **Next step:** Configure column mapping in the sidebar and click **🚀 Run RCA Analysis**.")
            st.markdown("#### 📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            st.caption(f"Showing first 10 rows of {len(df)} total rows")
        else:
            st.info("📤 **Get started:** Upload a CSV file or paste your data in the sidebar.")
            with st.expander("Expected format"):
                st.markdown("""
                - **Hub column**: one column with hub/location names (e.g. `hub_name_2`).
                - **3 driver columns**: metrics causing Request Loss %, e.g.:
                  - `Pro Discarded`
                  - `Supply Shortage (Leave)` → shown as *supply shortage leave (activity)*
                  - `supply_shortage` → shown as *supply shortage*
                
                Example (tab-separated):
                
                | hub_name_2 | Pro Discarded | Supply Shortage (Leave) | supply_shortage |
                |------------|---------------|-------------------------|-----------------|
                | hub_a      | 1.6           | 1.3                     | 6.7             |
                | hub_b      | 3.3           | 1.2                     | 6.4             |
                """)

#! --- STEP 2 UI ---
elif step == "Step 2: Sub-reasons Analysis":
    st.caption("Sub-reasons Analysis for Pro Discarded — find dominating sub-reasons for given hub(s)")
    
    st.sidebar.header("Input data")
    input_mode = st.sidebar.radio(
        "How do you want to provide data?",
        ["Upload CSV file", "Paste data (from Excel / Google Sheets)"],
        label_visibility="collapsed",
    )
    
    df_step2: Optional[pd.DataFrame] = None
    
    if input_mode == "Upload CSV file":
        uploaded = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a file with hub names and sub-reason columns (weightage percentages)",
        )
        df_step2 = get_df_from_upload(uploaded)
    else:
        st.sidebar.caption("Paste tab or comma-separated data below (e.g. copy from Excel or Google Sheets).")
        pasted = st.sidebar.text_area(
            "Paste your data",
            height=180,
            placeholder="hub_name\thasAllRequestSkillsV2\thasNoSuitableProfessional\t...\nhub1\t67.9\t9.2\t...",
            label_visibility="collapsed",
        )
        df_step2 = get_df_from_paste(pasted) if pasted else None
    
    if df_step2 is not None:
        df_step2 = _normalize_columns(df_step2)
        cols = list(df_step2.columns)
        
        st.sidebar.success(f"✅ Data loaded: {len(df_step2)} rows, {len(cols)} columns")
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Configuration")
        
        hub_col_step2 = st.sidebar.selectbox(
            "Hub name column",
            options=cols,
            index=0,
            help="Column that contains hub / location names",
        )
        
        # Detect data format and show appropriate column selectors
        data_format = detect_data_format(df_step2, hub_col_step2)
        
        sub_reason_col_step2 = None
        weightage_col_step2 = None
        
        if data_format == 'long':
            st.sidebar.info("📋 Long format detected (multiple rows per hub)")
            remaining_cols = [c for c in cols if c != hub_col_step2]
            
            # Auto-detect sub_reason_column
            sub_reason_candidates = [c for c in remaining_cols if c.lower() not in ['rl_absolute', 'weightage', 'value', 'percentage', '%']]
            sub_reason_default_idx = 0
            if sub_reason_candidates:
                sub_reason_default_idx = remaining_cols.index(sub_reason_candidates[0])
            
            sub_reason_col_step2 = st.sidebar.selectbox(
                "Sub-reason column",
                options=remaining_cols,
                index=sub_reason_default_idx,
                help="Column that contains sub-reason names (e.g., hasAllRequestSkillsV2)",
            )
            
            # Auto-detect weightage_column
            weightage_candidates = [c for c in remaining_cols if c.lower() in ['rl_absolute', 'weightage', 'value', 'percentage', '%'] or 'absolute' in c.lower()]
            weightage_default_idx = 0
            if weightage_candidates:
                weightage_default_idx = remaining_cols.index(weightage_candidates[0])
            else:
                # Try numeric columns
                numeric_cols = df_step2.select_dtypes(include=['number']).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != hub_col_step2 and c != sub_reason_col_step2]
                if numeric_cols:
                    weightage_default_idx = remaining_cols.index(numeric_cols[0])
            
            weightage_col_step2 = st.sidebar.selectbox(
                "Weightage column",
                options=remaining_cols,
                index=weightage_default_idx,
                help="Column that contains weightage/percentage values (e.g., RL_ABSOLUTE)",
            )
        else:
            st.sidebar.info("📊 Wide format detected (one row per hub)")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎛️ Analysis Settings")
        
        show_all_reasons = st.sidebar.checkbox(
            "Show all sub-reasons",
            value=False,
            help="If checked, shows all sub-reasons above threshold. If unchecked, shows only clearly dominating ones.",
        )
        
        min_threshold = st.sidebar.slider(
            "Minimum weightage threshold (%)",
            min_value=0.0,
            max_value=20.0,
            value=1.0,
            step=0.5,
            help="Sub-reasons below this threshold will be ignored (unless they're significant relative to top reason)",
        )
        
        dominance_ratio = st.sidebar.slider(
            "Dominance ratio",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Ratio threshold for determining if a reason is clearly dominating (only used if 'Show all' is unchecked)",
        )
        
        st.sidebar.markdown("---")
    
    # Hub name input (main area)
    st.markdown("### 📝 Enter Hub Name(s)")
    st.caption("Enter hub names separated by commas OR paste a column (one hub per line)")
    
    hub_input_mode = st.radio(
        "Input format",
        ["Single/Comma-separated", "Paste column (one per line)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    
    if hub_input_mode == "Single/Comma-separated":
        hub_input = st.text_input(
            "Hub name(s)",
            placeholder="aitchowkhl_city_delhi_v2_salon_at_home, hub2, hub3",
            help="Enter a single hub name, or multiple hub names separated by commas",
            label_visibility="collapsed",
        )
    else:
        hub_input = st.text_area(
            "Paste hub names (one per line)",
            placeholder="aitchowkhl_city_delhi_v2_salon_at_home\nhub2\nhub3",
            help="Paste a column of hub names from Excel - one hub name per line",
            height=150,
            label_visibility="collapsed",
        )
    
    analyze_clicked = st.button("🔍 Analyze Hub(s)", type="primary", use_container_width=True)
    
    if analyze_clicked and df_step2 is not None and hub_input:
        # Parse hub names: support both comma-separated and newline-separated
        hub_names = []
        if hub_input_mode == "Single/Comma-separated":
            # Split by comma
            hub_names = [h.strip() for h in hub_input.split(",") if h.strip()]
        else:
            # Split by newline (column format)
            hub_names = [h.strip() for h in hub_input.split("\n") if h.strip()]
        
        if not hub_names:
            st.error("Please enter at least one hub name.")
        else:
            try:
                results = analyze_hubs(
                    df_step2,
                    hub_names,
                    hub_col_step2,
                    sub_reason_col_step2,
                    weightage_col_step2,
                    min_threshold,
                    dominance_ratio,
                    show_all_reasons,
                )
                st.session_state["step2_results"] = results
                st.session_state["step2_hub_col"] = hub_col_step2
            except Exception as e:
                st.error(f"Error analyzing hubs: {e}")
    
    # Show Step 2 results
    if st.session_state.get("step2_results") is not None:
        results = st.session_state["step2_results"]
        hub_col = st.session_state.get("step2_hub_col", "hub_name")
        
        st.success(f"✅ Analysis completed for {len(results)} hub(s).")
        
        # Display results
        tab_summary, tab_detailed = st.tabs(["Summary", "Detailed view"])
        
        with tab_summary:
            for hub_name, sub_reasons in results.items():
                if sub_reasons:
                    formatted = format_sub_reasons_output(sub_reasons)
                    st.markdown(
                        f'<div class="sub-reason-block"><span class="rca-hub">{hub_name}</span><br><span class="rca-text">{formatted}</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(f"No sub-reasons found for hub: {hub_name}")
        
        with tab_detailed:
            # Show detailed table with all sub-reasons
            for hub_name, sub_reasons in results.items():
                st.markdown(f"#### {hub_name}")
                if sub_reasons:
                    detail_df = pd.DataFrame(sub_reasons, columns=["Sub-reason", "Weightage (%)"])
                    detail_df = detail_df.sort_values("Weightage (%)", ascending=False)
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"No sub-reasons found for hub: {hub_name}")
                st.divider()
        
        # Create summary text
        summary_lines = []
        for hub_name, sub_reasons in results.items():
            if sub_reasons:
                formatted = format_sub_reasons_output(sub_reasons)
                summary_lines.append(f"{hub_name}: {formatted}")
        
        if summary_lines:
            summary_text = "\n".join(summary_lines)
            st.text_area("Summary (copy-friendly)", summary_text, height=200, disabled=True)
    else:
        if df_step2 is not None:
            st.info("💡 **Next step:** Enter hub name(s) above and click **🔍 Analyze Hub(s)**.")
            st.markdown("#### 📊 Data Preview")
            st.dataframe(df_step2.head(10), use_container_width=True, hide_index=True)
            if hub_col_step2 in df_step2.columns:
                unique_hubs = df_step2[hub_col_step2].astype(str).unique()
                st.caption(f"📌 Available hubs: {', '.join(unique_hubs[:10])}")
                if len(unique_hubs) > 10:
                    st.caption(f"... and {len(unique_hubs) - 10} more")
        else:
            st.info("📤 **Get started:** Upload a CSV file or paste your data in the sidebar.")
            with st.expander("Expected format"):
                st.markdown("""
                - **Hub column**: one column with hub/location names (e.g. `hub_name`).
                - **Sub-reason columns**: multiple columns, each representing a sub-reason for "Pro Discarded".
                - **Values**: weightage percentages (e.g., 67.9, 9.2, 7.6).
                
                Example structure:
                
                | hub_name | hasAllRequestSkillsV2 | hasNoSuitableProfessional | lastStatusReason-No-Show (Pro) | ... |
                |----------|----------------------|---------------------------|-------------------------------|-----|
                | hub_a    | 67.9                 | 9.2                       | 7.6                            | ... |
                | hub_b    | 59.3                 | 10.8                      | 8.1                            | ... |
                
                The system will identify dominating sub-reasons based on weightage thresholds.
                """)

# --- STEP 3 UI ---
elif step == "Step 3: Service Tags Analysis":
    st.markdown("### 🏷️ Step 3: Service Tags Analysis")
    st.caption("Find dominant services causing RL for given hub(s)")
    st.markdown("---")

    st.sidebar.markdown("### 📥 Data Input")
    input_mode = st.sidebar.radio(
        "How do you want to provide data?",
        ["Upload CSV file", "Paste data (from Excel / Google Sheets)"],
        label_visibility="collapsed",
    )

    df_step3: Optional[pd.DataFrame] = None

    if input_mode == "Upload CSV file":
        uploaded = st.sidebar.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a file with hub names, category, tags, and RL (long format)",
        )
        df_step3 = get_df_from_upload(uploaded)
    else:
        st.sidebar.caption("Paste tab or comma-separated data below (e.g. copy from Excel or Google Sheets).")
        pasted = st.sidebar.text_area(
            "Paste your data",
            height=220,
            placeholder="Hub Name\tCategory\tTags\tRL\nisro layout_hl_city_bangalore_v2_salon_at_home\tSalon Prime\tpedicure::gender:female\t7.8\n...",
            label_visibility="collapsed",
        )
        df_step3 = get_df_from_paste(pasted) if pasted else None

    tags_col_step3 = None
    rl_col_step3 = None
    hub_col_step3 = None

    if df_step3 is not None:
        df_step3 = _normalize_columns(df_step3)
        cols = list(df_step3.columns)
        
        st.sidebar.success(f"✅ Data loaded: {len(df_step3)} rows, {len(cols)} columns")
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Configuration")

        hub_col_step3 = st.sidebar.selectbox(
            "Hub name column",
            options=cols,
            index=0,
            help="Column that contains hub / location names",
        )

        # Default guesses for tags and RL columns
        remaining_cols = [c for c in cols if c != hub_col_step3]

        # Tags column
        tag_candidates = [c for c in remaining_cols if "tag" in c.lower()]
        tags_default_idx = remaining_cols.index(tag_candidates[0]) if tag_candidates else 1 if len(remaining_cols) > 1 else 0
        tags_col_step3 = st.sidebar.selectbox(
            "Tags column",
            options=remaining_cols,
            index=tags_default_idx,
            help="Column that contains service tags (e.g., pedicure::gender:female::product_used_skill_ind:CrystalRosePedicure)",
        )

        # RL column
        rl_candidates = [c for c in remaining_cols if c.lower() in ["rl", "rl_absolute", "weightage", "value", "percentage"] or "rl" in c.lower()]
        rl_default_idx = remaining_cols.index(rl_candidates[0]) if rl_candidates else (remaining_cols.index("RL") if "RL" in remaining_cols else 0)
        rl_col_step3 = st.sidebar.selectbox(
            "RL column",
            options=remaining_cols,
            index=rl_default_idx,
            help="Column that contains RL / weightage values",
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎛️ Analysis Settings")

        show_all_services = st.sidebar.checkbox(
            "Show all service tags",
            value=False,
            help="If checked, shows all tags above threshold. If unchecked, shows only clearly dominating ones.",
        )

        min_rl_threshold = st.sidebar.slider(
            "Minimum RL threshold",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="Display threshold: tags below this RL are hidden, but ALL rows are still used for aggregation.",
        )

        dominance_ratio_step3 = st.sidebar.slider(
            "Dominance ratio",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Ratio threshold for determining if a tag is clearly dominating (only used if 'Show all' is unchecked)",
        )

        st.sidebar.divider()

    # Hub name input (main area) for Step 3
    st.markdown("### 📝 Enter Hub Name(s)")
    st.caption("Enter hub names separated by commas OR paste a column (one hub per line)")

    hub_input_mode_3 = st.radio(
        "Input format (Step 3)",
        ["Single/Comma-separated", "Paste column (one per line)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if hub_input_mode_3 == "Single/Comma-separated":
        hub_input_3 = st.text_input(
            "Hub name(s)",
            placeholder="isro layout_hl_city_bangalore_v2_salon_at_home, girinagar_city_bangalore_v2_salon_at_home",
            help="Enter a single hub name, or multiple hub names separated by commas",
            label_visibility="collapsed",
        )
    else:
        hub_input_3 = st.text_area(
            "Paste hub names (one per line)",
            placeholder="isro layout_hl_city_bangalore_v2_salon_at_home\ngirinagar_city_bangalore_v2_salon_at_home\n...",
            help="Paste a column of hub names from Excel - one hub name per line",
            height=150,
            label_visibility="collapsed",
        )

    analyze_clicked_3 = st.button("🔍 Analyze Hub(s)", type="primary", use_container_width=True)

    if analyze_clicked_3 and df_step3 is not None and hub_input_3:
        hub_names_3: list[str] = []
        if hub_input_mode_3 == "Single/Comma-separated":
            hub_names_3 = [h.strip() for h in hub_input_3.split(",") if h.strip()]
        else:
            hub_names_3 = [h.strip() for h in hub_input_3.split("\n") if h.strip()]

        if not hub_names_3:
            st.error("Please enter at least one hub name.")
        elif hub_col_step3 is None or tags_col_step3 is None or rl_col_step3 is None:
            st.error("Please ensure hub, tags, and RL columns are selected in the sidebar.")
        else:
            try:
                results_3 = analyze_hubs_services(
                    df_step3,
                    hub_names_3,
                    hub_column=hub_col_step3,
                    tags_column=tags_col_step3,
                    rl_column=rl_col_step3,
                    min_rl_threshold=min_rl_threshold,
                    dominance_ratio=dominance_ratio_step3,
                    show_all=show_all_services,
                )
                st.session_state["step3_results"] = results_3
                st.session_state["step3_hub_col"] = hub_col_step3
            except Exception as e:
                st.error(f"Error analyzing hubs (Step 3): {e}")

    # Show Step 3 results
    if st.session_state.get("step3_results") is not None:
        results_3 = st.session_state["step3_results"]

        st.success(f"✅ Service tag analysis completed for {len(results_3)} hub(s).")

        for hub_name, services in results_3.items():
            formatted_plain = format_services_output(services)
            formatted_html = formatted_plain.replace("\n", "<br>")
            st.markdown(
                f'<div class="sub-reason-block"><span class="rca-hub">{hub_name}</span><br><span class="rca-text">{formatted_html}</span></div>',
                unsafe_allow_html=True,
            )

        # Copy-friendly summary
        lines_3: list[str] = []
        for hub_name, services in results_3.items():
            formatted_plain = format_services_output(services)
            # Put services on new lines under each hub for easy copy
            lines_3.append(f"{hub_name}:\n{formatted_plain}")
        if lines_3:
            st.text_area(
                "Summary (copy-friendly)",
                "\n".join(lines_3),
                height=200,
                disabled=True,
            )
    else:
        if df_step3 is not None:
            st.info("💡 **Next step:** Enter hub name(s) above and click **🔍 Analyze Hub(s)**.")
            st.markdown("#### 📊 Data Preview")
            st.dataframe(df_step3.head(15), use_container_width=True, hide_index=True)
            st.caption(f"Showing first 15 rows of {len(df_step3)} total rows")
        else:
            st.info("📤 **Get started:** Upload a CSV file or paste your data in the sidebar.")
            with st.expander("Expected format"):
                st.markdown("""
                - **Hub column**: one column with hub/location names (e.g. `Hub Name`).
                - **Category** (optional): e.g. `Salon Prime`, `Salon Luxe`.
                - **Tags column**: service tags, e.g.:
                  - `pedicure::gender:female`
                  - `pedicure::gender:female::product_used_skill_ind:CrystalRosePedicure`
                  - `waxing::gender:female::product_used_skill_ind:RicaWhiteChocolateWax`
                - **RL column**: RL weightage for that hub-tag row.

                The system will aggregate RL by service tag per hub and return dominating tags, e.g.:
                `Pedi : CrystalRosePedicure Pedi : heel_peel Waxing : RicaWhiteChocolateWax`.
                """)

# --- STEP 4 UI ---
elif step == "Step 4: Summary / Bucketization":
    st.markdown("### 📋 Step 4: Summary / Bucketization")
    st.caption("Categorize RCA statements into predefined buckets")
    st.markdown("---")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    
    threshold_ratio = st.sidebar.slider(
        "Threshold ratio for ignoring smaller reasons",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="If smaller reason < (larger reason * ratio), it will be ignored. Default: 0.5 (50%)",
    )
    
    st.sidebar.markdown("---")
    
    st.markdown("### 📝 Enter RCA Statements")
    st.caption("Enter one or more RCA statements (one per line or separated by commas)")
    
    input_mode_step4 = st.radio(
        "Input format",
        ["Single/Comma-separated", "Paste column (one per line)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    
    if input_mode_step4 == "Single/Comma-separated":
        statements_input = st.text_input(
            "RCA statements",
            placeholder='8% RL due to pro discarded (hasAllRequestSkillsV2) and 6% due to supply shortage',
            help="Enter RCA statements separated by commas",
            label_visibility="collapsed",
        )
    else:
        statements_input = st.text_area(
            "Paste RCA statements (one per line)",
            placeholder='8% RL due to pro discarded (hasAllRequestSkillsV2) and 6% due to supply shortage\n7% RL due to supply shortage',
            help="Paste RCA statements from Excel or text - one statement per line",
            height=200,
            label_visibility="collapsed",
        )
    
    bucketize_clicked = st.button("📦 Bucketize Statements", type="primary", use_container_width=True)
    
    if bucketize_clicked and statements_input:
        # Parse statements
        statements_list = []
        if input_mode_step4 == "Single/Comma-separated":
            statements_list = [s.strip() for s in statements_input.split(",") if s.strip()]
        else:
            statements_list = [s.strip() for s in statements_input.split("\n") if s.strip()]
        
        if not statements_list:
            st.error("Please enter at least one RCA statement.")
        else:
            try:
                buckets = bucketize_statements(statements_list, threshold_ratio)
                st.session_state["step4_results"] = list(zip(statements_list, buckets))
            except Exception as e:
                st.error(f"Error bucketizing statements: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show Step 4 results
    if st.session_state.get("step4_results") is not None:
        results_4 = st.session_state["step4_results"]
        
        st.success(f"✅ Bucketization completed for {len(results_4)} statement(s).")
        
        # Create results dataframe
        results_df = pd.DataFrame(results_4, columns=["RCA Statement", "Bucket"])
        
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Copy-friendly summary
        summary_lines_4 = []
        for stmt, bucket in results_4:
            summary_lines_4.append(f"{stmt}\t{bucket}")
        
        if summary_lines_4:
            summary_text_4 = "\n".join(summary_lines_4)
            st.text_area(
                "Summary (copy-friendly, tab-separated)",
                summary_text_4,
                height=200,
                disabled=True,
            )
        
        # Download CSV
        csv_bytes_4 = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results as CSV",
            data=csv_bytes_4,
            file_name="rca_step4_buckets.csv",
            mime="text/csv",
        )
    else:
        st.info("💡 **Next step:** Enter RCA statement(s) above and click **📦 Bucketize Statements**.")
        with st.expander("Expected format and examples"):
            st.markdown("""
            **Input format:**
            - `X% RL due to [reason1] and Y% due to [reason2]`
            - Parentheses content (like sub-reasons) are automatically ignored
            
            **Examples:**
            - `8% RL due to pro discarded (hasAllRequestSkillsV2) and 6% due to supply shortage`
              → Bucket: **Pro discarded + Supply shortage**
            
            - `16% RL due to supply shortage and 3% due to pro discarded`
              → Bucket: **Supply shortage** (3% is ignored as it's < 50% of 16%)
            
            - `5% RL due to supply shortage leave (activity) and 4% due to pro discarded`
              → Bucket: **Activity + Pro discarded**
            
            **Available buckets:**
            - Activity
            - Activity + Pro discarded
            - Activity + Supply shortage
            - Activity + Supply shortage + Pro discarded
            - Pro discarded
            - Pro discarded + Activity
            - Pro discarded + Supply shortage
            - Supply shortage
            - Supply shortage + Activity
            - Supply shortage + Pro discarded
            """)
