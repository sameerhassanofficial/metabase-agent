import streamlit as st
import pandas as pd
import plotly.express as px
from analytics_utils import get_numeric_summaries, predict_trend
import logging
import uuid

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
            # Use column_names for DataFrame columns, and do not include header row as data
            if "column_names" in part and part["column_names"]:
                # Remove header row if present
                if part["data"] and part["data"][0] == part["column_names"]:
                    data_rows = part["data"][1:]
                else:
                    data_rows = part["data"]
                df = pd.DataFrame(data_rows, columns=part["column_names"])
            else:
                df = pd.DataFrame(part["data"])
            
            # Map generic column names to meaningful ones if needed
            if "column_names" not in part or not part["column_names"]:
                column_mapping = {
                    0: "Patient Name",
                    1: "Age", 
                    2: "Gender",
                    3: "Date of Birth",
                    4: "Location",
                    5: "Visit Date",
                    6: "CNIC",
                    7: "Weight",
                    8: "Temperature",
                    9: "Mobile Health Unit",
                    10: "Medical Record Number",
                    11: "Referral Details"
                }
                new_columns = []
                for i, col in enumerate(df.columns):
                    if isinstance(col, int) and col in column_mapping:
                        new_columns.append(column_mapping[col])
                    else:
                        new_columns.append(str(col))
                df.columns = new_columns
            
            # Heuristic: columns that should NOT be coerced to numeric
            id_like_keywords = ["code", "id", "name", "unit", "location"]
            def is_id_like(col):
                col_lower = str(col).lower()
                return any(kw in col_lower for kw in id_like_keywords)
            # Ensure all data is properly typed to avoid ArrowTypeError
            for col in df.columns:
                if not is_id_like(col):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception:
                        df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(str)
            numeric_cols = df.select_dtypes(include='number').columns
            # Add a 'Total' column for all numeric columns (row-wise sum) only if it makes sense
            if not df.empty and len(numeric_cols) > 0:
                if len(numeric_cols) > 1 and all(col in ['Age', 'Weight', 'Temperature'] for col in numeric_cols):
                    df['Total'] = df[numeric_cols].sum(axis=1)
                    total_row = {col: df[col].sum() for col in numeric_cols}
                    total_row['Total'] = df['Total'].sum()
                    total_row_full = {col: total_row.get(col, '') for col in df.columns}
                    total_row_full[next(iter(df.columns))] = 'Total'
                    df = pd.concat([df, pd.DataFrame([total_row_full])], ignore_index=True)
            tab1, tab2 = st.tabs(["Chart", "Dataframe"])
            with tab1:
                if len(numeric_cols) > 0:
                    plot_df = df.iloc[:-1] if 'Total' in str(df.iloc[-1,0]) else df
                    logger = logging.getLogger(__name__)
                    logger.info(f"Chart data columns: {list(plot_df.columns)}")
                    logger.info(f"Numeric columns: {list(numeric_cols)}")
                    logger.info(f"First few rows: {plot_df.head().to_dict()}")
                    # Robust x_col selection
                    x_col = None
                    x_title = 'Index'
                    healthcare_keywords = ['mhu', 'unit', 'gender', 'location', 'name']
                    for keyword in healthcare_keywords:
                        for col in plot_df.columns:
                            col_str = str(col)
                            if keyword in col_str.lower() and col not in numeric_cols:
                                x_col = col
                                x_title = col_str
                                logger.info(f"Found healthcare categorical column: {col}")
                                break
                        if x_col:
                            break
                    if not x_col:
                        # Use first non-numeric column
                        for col in plot_df.columns:
                            if col not in numeric_cols:
                                x_col = col
                                x_title = str(col)
                                logger.info(f"Using first non-numeric column: {col}")
                                break
                    if not x_col:
                        # If still not found, create an Index column
                        plot_df = plot_df.copy()
                        plot_df['Index'] = range(1, len(plot_df)+1)
                        x_col = 'Index'
                        x_title = 'Index'
                        logger.info("No suitable x_col found, using DataFrame index as column.")
                    logger.info(f"Final x-axis selection: {x_col} (title: {x_title})")
                    fig = px.bar(
                        plot_df,
                        x=x_col,
                        y=numeric_cols,
                        barmode='group',
                        title=part.get("title", "Healthcare Data Analysis"),
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    fig.update_layout(
                        xaxis_title=x_title,
                        yaxis_title='Value',
                        font=dict(size=16),
                        legend_title_text='Metric',
                        bargap=0.2,
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    unique_id = uuid.uuid4().hex
                    chart_key = f"bar_chart_{unique_id}"
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                    # Only run trend prediction if x_col is a valid column, numeric, and not an ID/code/categorical column
                    if (
                        isinstance(x_col, str)
                        and x_col in plot_df.columns
                        and pd.api.types.is_numeric_dtype(plot_df[x_col])
                        and not is_id_like(x_col)
                    ):
                        try:
                            for value_col in numeric_cols:
                                pred = predict_trend(plot_df, x_col, value_col, periods_ahead=1)
                                if pred is not None:
                                    st.write(f"Predicted next value for {value_col}: {pred}")
                        except Exception as e:
                            logger.warning(f"Trend prediction failed: {e}")
                    else:
                        logger.info(f"Skipping trend prediction: x_col '{x_col}' is not suitable (must be numeric and not ID/code/categorical).")
                else:
                    st.info("No numeric columns to plot.")
            with tab2:
                st.dataframe(df, use_container_width=True)
            # --- End robust chart/table logic ---
        elif part_type == "chart":
            chart_spec = part["spec"]
            df = pd.DataFrame(chart_spec["data"])
            chart_type = chart_spec["chart_type"]
            fig = None
            value_col = None
            x_col = None
            
            # Handle different column naming conventions
            if chart_type == "pie":
                # Try different possible column names for pie charts
                if "values_column" in chart_spec:
                    value_col = chart_spec["values_column"]
                elif "Count" in df.columns:
                    value_col = "Count"
                elif "Value" in df.columns:
                    value_col = "Value"
                else:
                    # Find first numeric column
                    numeric_cols = df.select_dtypes(include='number').columns
                    value_col = numeric_cols[0] if len(numeric_cols) > 0 else None
                
                if "labels_column" in chart_spec:
                    labels_col = chart_spec["labels_column"]
                elif "Category" in df.columns:
                    labels_col = "Category"
                elif "Name" in df.columns:
                    labels_col = "Name"
                else:
                    # Find first non-numeric column
                    non_numeric_cols = df.select_dtypes(exclude='number').columns
                    labels_col = non_numeric_cols[0] if len(non_numeric_cols) > 0 else None
                
                if value_col and labels_col and value_col in df.columns and labels_col in df.columns:
                    fig = px.pie(
                        df,
                        names=labels_col,
                        values=value_col,
                        title=chart_spec.get("title", "Pie Chart"),
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    fig.update_traces(textinfo='percent+label', textfont_size=16)
                else:
                    st.warning(f"Could not find required columns for pie chart. Available columns: {list(df.columns)}")
                    continue
                    
            elif chart_type == "bar":
                # Try different possible column names for bar charts
                if "y_axis_column" in chart_spec:
                    value_col = chart_spec["y_axis_column"]
                elif "Count" in df.columns:
                    value_col = "Count"
                elif "Value" in df.columns:
                    value_col = "Value"
                else:
                    # Find first numeric column
                    numeric_cols = df.select_dtypes(include='number').columns
                    value_col = numeric_cols[0] if len(numeric_cols) > 0 else None
                
                if "x_axis_column" in chart_spec:
                    x_col = chart_spec["x_axis_column"]
                elif "Category" in df.columns:
                    x_col = "Category"
                elif "Name" in df.columns:
                    x_col = "Name"
                else:
                    # Find first non-numeric column
                    non_numeric_cols = df.select_dtypes(exclude='number').columns
                    x_col = non_numeric_cols[0] if len(non_numeric_cols) > 0 else None
                
                if value_col and x_col and value_col in df.columns and x_col in df.columns:
                    fig = px.bar(
                        df,
                        x=x_col,
                        y=value_col,
                        title=chart_spec.get("title", "Bar Chart"),
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
                else:
                    st.warning(f"Could not find required columns for bar chart. Available columns: {list(df.columns)}")
                    continue
                    
            elif chart_type == "line":
                # Try different possible column names for line charts
                if "y_axis_column" in chart_spec:
                    value_col = chart_spec["y_axis_column"]
                elif "Count" in df.columns:
                    value_col = "Count"
                elif "Value" in df.columns:
                    value_col = "Value"
                else:
                    # Find first numeric column
                    numeric_cols = df.select_dtypes(include='number').columns
                    value_col = numeric_cols[0] if len(numeric_cols) > 0 else None
                
                if "x_axis_column" in chart_spec:
                    x_col = chart_spec["x_axis_column"]
                elif "Category" in df.columns:
                    x_col = "Category"
                elif "Name" in df.columns:
                    x_col = "Name"
                else:
                    # Find first non-numeric column
                    non_numeric_cols = df.select_dtypes(exclude='number').columns
                    x_col = non_numeric_cols[0] if len(non_numeric_cols) > 0 else None
                
                if value_col and x_col and value_col in df.columns and x_col in df.columns:
                    fig = px.line(
                        df,
                        x=x_col,
                        y=value_col,
                        title=chart_spec.get("title", "Line Chart"),
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
                    st.warning(f"Could not find required columns for line chart. Available columns: {list(df.columns)}")
                    continue
            else:
                st.warning(f"Unsupported chart type: {chart_spec['chart_type']}")
                continue
                
            tab1, tab2 = st.tabs(["Chart", "Dataframe"])
            with tab1:
                if fig:
                    # Add unique key for plotly chart
                    unique_id = uuid.uuid4().hex
                    chart_key = f"{chart_type}_chart_{unique_id}"
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                    if value_col and value_col in df.columns and pd.api.types.is_numeric_dtype(df[value_col]):
                        total = df[value_col].sum()
                        st.markdown(f"**Total {value_col}:** {total:,.2f}")
                    if chart_type in ("line", "bar") and x_col and value_col and x_col in df.columns and value_col in df.columns:
                        # Only run trend prediction if x_col is numeric and not an ID/code/categorical column
                        if (
                            isinstance(x_col, str)
                            and x_col in df.columns
                            and pd.api.types.is_numeric_dtype(df[x_col])
                            and not is_id_like(x_col)
                        ):
                            try:
                                pred = predict_trend(df, x_col, value_col, periods_ahead=1)
                                if pred is not None:
                                    st.markdown(f"**Predicted next {value_col}:** {pred:,.2f}")
                            except Exception as e:
                                logger.warning(f"Trend prediction failed: {e}")
                        else:
                            logger.info(f"Skipping trend prediction: x_col '{x_col}' is not suitable (must be numeric and not ID/code/categorical).")
            with tab2:
                st.dataframe(df, use_container_width=True)
            
            # Download button
            try:
                csv_data = df.to_csv(index=False).encode('utf-8')
                chart_title = chart_spec.get("title", "chart_data")
                st.download_button(
                    label="Download Chart Data (CSV)",
                    data=csv_data,
                    file_name=f"{chart_title.replace(' ', '_').lower()}_data.csv",
                    mime="text/csv",
                    key=f"download_csv_{chart_title.replace(' ', '_')}_{hash(str(chart_spec))}"
                )
            except Exception as e:
                logger.warning(f"Could not create download button: {e}")
