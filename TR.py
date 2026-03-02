import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="TR Project – Full Rule Validator", layout="wide")
st.title("TR Project – Survey Logic Checker")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    # =====================================================
    # GLOBAL CLEANING
    # =====================================================

    df = df.replace(["#NULL!", "NULL", "null", ""], np.nan)
    df.columns = df.columns.str.strip()

    # Strip hidden spaces
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    st.success("Excel uploaded and cleaned successfully")

    validation_errors = []

    def add_error(respid, rule_id, variable, actual_value, expected):
        validation_errors.append({
            "respid": respid,
            "RuleID": rule_id,
            "Variable": variable,
            "Actual_Value": actual_value,
            "Expected": expected
        })

    brands = range(1,18)

    body13_cols = [f"body_type_13_{i}" for i in range(1,12) if f"body_type_13_{i}" in df.columns]
    body22_cols = [f"body_type_22_{i}" for i in range(1,12) if f"body_type_22_{i}" in df.columns]

    # =====================================================
    # LOOP THROUGH RESPONDENTS
    # =====================================================

    for _, row in df.iterrows():

        respid = row.get("respid")

        # -----------------------------
        # SCREENING
        # -----------------------------

        def check_range(var, rule_id, valid_range):
            raw_val = row.get(var)
            val = pd.to_numeric(raw_val, errors="coerce")

            if pd.isna(val):
                return

            if val not in valid_range:
                add_error(
                    respid,
                    rule_id,
                    var,
                    raw_val,
                    f"Must be between {min(valid_range)}–{max(valid_range)}"
                )

        # Screening checks
        check_range("industry", "R1", range(1,11))

        if pd.notna(row.get("intro1")) and row.get("intro1") != 1:
            add_error(respid, "R2", "intro1", row.get("intro1"), "Must equal 1")

        if pd.notna(row.get("decision_maker")) and row.get("decision_maker") != 1:
            add_error(respid, "R3", "decision_maker", row.get("decision_maker"), "Must equal 1")

        if pd.notna(row.get("decision_maker_4axle")) and row.get("decision_maker_4axle") != 1:
            add_error(respid, "R4", "decision_maker_4axle", row.get("decision_maker_4axle"), "Must equal 1")

        if pd.notna(row.get("target_group_3")) and row.get("target_group_3") != 1:
            add_error(respid, "R5", "target_group_3", row.get("target_group_3"), "Must equal 1")

        # -----------------------------
        # TRUCK QUANTITY
        # -----------------------------

        truck_q = pd.to_numeric(row.get("truck_quantity"), errors="coerce")
        tq1 = pd.to_numeric(row.get("type_quantity_1"), errors="coerce")
        tq2 = pd.to_numeric(row.get("type_quantity_2"), errors="coerce")

        truck_q = 0 if pd.isna(truck_q) else truck_q
        tq1 = 0 if pd.isna(tq1) else tq1
        tq2 = 0 if pd.isna(tq2) else tq2

        if truck_q > 0:

            if tq1 <= 0 and tq2 <= 0:
                add_error(respid, "R6", "type_quantity_1/type_quantity_2",
                          f"{tq1}/{tq2}", "At least one must be > 0")

            if tq1 + tq2 != truck_q:
                add_error(
                    respid,
                    "R7",
                    "type_quantity_sum",
                    tq1 + tq2,
                    f"Must equal truck_quantity ({truck_q})"
                )

        # -----------------------------
        # BODY VALIDATION
        # -----------------------------

        total_body13 = 0
        for col in body13_cols:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if not pd.isna(val):
                total_body13 += val

        if tq2 > 0 and total_body13 != tq2:
            add_error(
                respid,
                "R8",
                "body_type_13_total",
                total_body13,
                f"Must equal type_quantity_2 ({tq2})"
            )

        total_body22 = 0
        for col in body22_cols:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if not pd.isna(val):
                total_body22 += val

        if tq1 > 0 and total_body22 != tq1:
            add_error(
                respid,
                "R9",
                "body_type_22_total",
                total_body22,
                f"Must equal type_quantity_1 ({tq1})"
            )

        # -----------------------------
        # CRANE VALIDATION
        # -----------------------------

        check_range("crane_13", "R10", range(1,5))
        check_range("crane_22", "R11", range(1,5))

        # -----------------------------
        # SCALE CHECKS
        # -----------------------------

        scale_rules = {
            "overall_satisfaction": ("R12", range(1,6)),
            "rear": ("R13", range(1,4)),
            "preparation": ("R14", range(1,7)),
            "bev": ("R15", range(1,7)),
            "bev_2": ("R16", range(1,6)),
            "bev_3": ("R17", range(1,6)),
        }

        for var, (rule_id, valid_range) in scale_rules.items():
            check_range(var, rule_id, valid_range)

    # =====================================================
    # BUILD REPORT
    # =====================================================

    report_df = pd.DataFrame(validation_errors)

    st.subheader("Validation Report")

    if report_df.empty:
        st.success("No validation errors found!")
    else:
        st.dataframe(report_df)

    # =====================================================
    # EXPORT
    # =====================================================

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        report_df.to_excel(writer, index=False, sheet_name="Validation_Report")

    st.download_button(
        "Download Validation Report",
        output.getvalue(),
        "Validation_Report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )