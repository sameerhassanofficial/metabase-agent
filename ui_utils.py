import streamlit as st
import pandas as pd
import plotly.express as px
from analytics_utils import get_numeric_summaries, predict_trend
import logging

def render_chat_response(response_json):
    """Renders a structured AI response, including text, charts, and tables. Logs errors for malformed parts."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Rendering response_json: {response_json}")
    for part in response_json.get("response_parts", []):
        if not isinstance(part, dict) or "type" not in part:
            logger.error(f"Malformed response part: {part}")
            st.warning(f"⚠️ Malformed response part: {part}")
            continue
        part_type = part.get("type")
        
        if part_type == "text":
            st.markdown(part["content"])
        
        elif part_type == "table":
            df = pd.DataFrame(part["data"])
            st.dataframe(df)
            # Show totals for numeric columns if available
            if not df.empty:
                numeric_cols = df.select_dtypes(include='number').columns
                if len(numeric_cols) > 0:
                    total_row = {col: df[col].sum() for col in numeric_cols}
                    total_str = ", ".join([f"**Total {col}:** {total_row[col]:,.2f}" for col in numeric_cols])
                    st.markdown(total_str)
                # Show numpy-based summaries
                summaries = get_numeric_summaries(df)
                if summaries:
                    for col, stats in summaries.items():
                        st.markdown(
                            f"**{col}**: Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, Std={stats['std']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}"
                        )

        elif part_type == "chart":
            chart_spec = part["spec"]
            df = pd.DataFrame(chart_spec["data"])
            chart_type = chart_spec["chart_type"]
            fig = None
            value_col = None
            x_col = None
            if chart_type == "pie":
                value_col = chart_spec["values_column"]
                fig = px.pie(df, names=chart_spec["labels_column"], values=value_col, title=chart_spec["title"])
            elif chart_type == "bar":
                value_col = chart_spec["y_axis_column"]
                x_col = chart_spec["x_axis_column"]
                fig = px.bar(df, x=x_col, y=value_col, title=chart_spec["title"])
            elif chart_type == "line":
                value_col = chart_spec["y_axis_column"]
                x_col = chart_spec["x_axis_column"]
                fig = px.line(df, x=x_col, y=value_col, title=chart_spec["title"])
            else:
                st.warning(f"Unsupported chart type: {chart_spec['chart_type']}")
                continue
            chart_spec_for_hash = {k: v for k, v in chart_spec.items() if k != 'data'}
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{chart_spec['title'].replace(' ', '_')}_{hash(frozenset(chart_spec_for_hash.items()))}")
            # Show total for value column if available
            if value_col and value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
                total = df[value_col].sum()
                st.markdown(f"**Total {value_col}:** {total:,.2f}")
            # Show trend prediction for time series charts
            if chart_type in ("line", "bar") and x_col and value_col and x_col in df.columns and value_col in df.columns:
                pred = predict_trend(df, x_col, value_col, periods_ahead=1)
                if pred is not None:
                    st.markdown(f"**Predicted next {value_col}:** {pred:,.2f}")
            # Add download button for chart data (CSV)
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Chart Data (CSV)",
                data=csv_data,
                file_name=f"{chart_spec["title"].replace(" ", "_").lower()}_data.csv",
                mime="text/csv",
                key=f"download_csv_{chart_spec['title'].replace(' ', '_')}_{hash(frozenset(chart_spec_for_hash.items()))}"
            )
