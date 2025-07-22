import streamlit as st
import pandas as pd
import plotly.express as px
from analytics_utils import get_numeric_summaries, predict_trend
import logging

def render_chat_response(response_json):
    """Renders a structured AI response, including text, charts, and tables. Logs errors for malformed parts."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Rendering response_json: {response_json}")
    # 1. CEO-level summary: show the first text part (if any) above everything else
    summary_shown = False
    table_totals = None
    grand_total = None
    for part in response_json.get("response_parts", []):
        if not summary_shown and isinstance(part, dict) and part.get("type") == "text":
            # Check if a table is present and get totals for summary
            for p2 in response_json.get("response_parts", []):
                if p2.get("type") == "table":
                    df2 = pd.DataFrame(p2["data"])
                    numeric_cols = df2.select_dtypes(include='number').columns
                    if len(numeric_cols) > 0:
                        table_totals = {col: df2[col].sum() for col in numeric_cols}
                        # Grand total: sum of all column totals
                        grand_total = sum(table_totals.values())
            summary = part['content']
            if table_totals:
                total_str = ", ".join([f"Total {col}: {table_totals[col]:,.2f}" for col in table_totals])
                summary += f"<br><span style='font-size:1em;color:#444;'>{total_str}" \
                           f"<br><b>Grand Total: {grand_total:,.2f}</b></span>"
            st.markdown(f"<div style='font-size:1.2em;font-weight:bold;margin-bottom:0.5em;'>{summary}</div>", unsafe_allow_html=True)
            summary_shown = True
    # 2. Show charts/dataframes in tabs, with improved chart appearance
    for part in response_json.get("response_parts", []):
        if not isinstance(part, dict) or "type" not in part:
            logger.error(f"Malformed response part: {part}")
            st.warning(f"⚠️ Malformed response part: {part}")
            continue
        part_type = part.get("type")
        if part_type == "text":
            if not summary_shown:  # If not already shown as summary
                st.markdown(part["content"])
        elif part_type == "table":
            df = pd.DataFrame(part["data"])
            numeric_cols = df.select_dtypes(include='number').columns
            # Add a 'Total' column for all numeric columns (row-wise sum)
            if not df.empty and len(numeric_cols) > 0:
                df['Total'] = df[numeric_cols].sum(axis=1)
                # Add a 'Total' row at the bottom for all numeric columns and the new 'Total' column
                total_row = {col: df[col].sum() for col in numeric_cols}
                total_row['Total'] = df['Total'].sum()
                total_row_full = {col: total_row.get(col, '') for col in df.columns}
                total_row_full[next(iter(df.columns))] = 'Total'  # Label the first column as 'Total'
                df = pd.concat([df, pd.DataFrame([total_row_full])], ignore_index=True)
            tab1, tab2 = st.tabs(["Chart", "Dataframe"])
            with tab1:
                if len(numeric_cols) > 0:
                    # Exclude total row from chart
                    plot_df = df.iloc[:-1] if 'Total' in str(df.iloc[-1,0]) else df
                    fig = px.bar(
                        plot_df,
                        x=plot_df.index if 'MHU' not in plot_df.columns else plot_df['MHU'],
                        y=numeric_cols,
                        barmode='group',
                        title="Bar Chart of Numeric Columns",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    fig.update_layout(
                        xaxis_title='MHU' if 'MHU' in df.columns else 'Index',
                        yaxis_title='Total',
                        font=dict(size=16),
                        legend_title_text='Metric',
                        bargap=0.2,
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No numeric columns to plot.")
            with tab2:
                st.dataframe(df, use_container_width=True)
        elif part_type == "chart":
            chart_spec = part["spec"]
            df = pd.DataFrame(chart_spec["data"])
            chart_type = chart_spec["chart_type"]
            fig = None
            value_col = None
            x_col = None
            if chart_type == "pie":
                value_col = chart_spec["values_column"]
                fig = px.pie(
                    df,
                    names=chart_spec["labels_column"],
                    values=value_col,
                    title=chart_spec["title"],
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.update_traces(textinfo='percent+label', textfont_size=16)
            elif chart_type == "bar":
                value_col = chart_spec["y_axis_column"]
                x_col = chart_spec["x_axis_column"]
                fig = px.bar(
                    df,
                    x=x_col,
                    y=value_col,
                    title=chart_spec["title"],
                    color=value_col,
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=value_col,
                    font=dict(size=16),
                    legend_title_text=value_col,
                    bargap=0.2,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
            elif chart_type == "line":
                value_col = chart_spec["y_axis_column"]
                x_col = chart_spec["x_axis_column"]
                fig = px.line(
                    df,
                    x=x_col,
                    y=value_col,
                    title=chart_spec["title"],
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=value_col,
                    font=dict(size=16),
                    legend_title_text=value_col,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
            else:
                st.warning(f"Unsupported chart type: {chart_spec['chart_type']}")
                continue
            tab1, tab2 = st.tabs(["Chart", "Dataframe"])
            with tab1:
                st.plotly_chart(fig, use_container_width=True)
                if value_col and value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
                    total = df[value_col].sum()
                    st.markdown(f"**Total {value_col}:** {total:,.2f}")
                if chart_type in ("line", "bar") and x_col and value_col and x_col in df.columns and value_col in df.columns:
                    pred = predict_trend(df, x_col, value_col, periods_ahead=1)
                    if pred is not None:
                        st.markdown(f"**Predicted next {value_col}:** {pred:,.2f}")
            with tab2:
                st.dataframe(df, use_container_width=True)
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Chart Data (CSV)",
                data=csv_data,
                file_name=f"{chart_spec["title"].replace(" ", "_").lower()}_data.csv",
                mime="text/csv",
                key=f"download_csv_{chart_spec['title'].replace(' ', '_')}_{hash(frozenset(chart_spec_for_hash.items()))}"
            )
