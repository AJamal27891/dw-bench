"""Phase 2: Compute enriched node features for Schema Track graphs.

For each table node, computes:
  1. DDL text embedding — synthesized CREATE TABLE statement embedded via
     sentence-transformers (all-MiniLM-L6-v2, 384-dim)
  2. Structural features — degree, betweenness, pagerank, silo ID
     (already computed in Phase 1)

Final node.x = Concat([DDL_embedding(384), structural(6)]) = 390-dim

Usage:
    python compute_features.py --dataset adventureworks
    python compute_features.py --dataset tpc-ds
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def synthesize_ddl(table_name: str, columns: pd.DataFrame) -> str:
    """Synthesize a CREATE TABLE DDL string from column metadata.

    Example output:
        CREATE TABLE Sales.SalesOrderHeader (
            SalesOrderID int NOT NULL,
            RevisionNumber tinyint NOT NULL,
            OrderDate datetime NOT NULL,
            ...
        );
    """
    lines = [f"CREATE TABLE {table_name} ("]
    for _, col in columns.iterrows():
        col_name = col['COLUMN_NAME']
        data_type = col['DATA_TYPE']
        max_len = col.get('CHARACTER_MAXIMUM_LENGTH', '')
        nullable = col.get('IS_NULLABLE', 'YES')

        type_str = data_type
        if pd.notna(max_len) and max_len != '' and int(float(max_len)) > 0:
            type_str = f"{data_type}({int(float(max_len))})"

        null_str = "NOT NULL" if nullable == 'NO' else "NULL"
        lines.append(f"    {col_name} {type_str} {null_str},")

    # Remove trailing comma from last column
    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(',')
    lines.append(");")
    return "\n".join(lines)


def build_adventureworks_ddls(data_dir: Path) -> dict:
    """Build DDL strings for all AdventureWorks tables."""
    oltp_cols = pd.read_csv(data_dir / 'AdventureWorks_Columns.csv')
    dw_cols = pd.read_csv(data_dir / 'AdventureWorksDW_Columns.csv')

    oltp_tables = pd.read_csv(data_dir / 'AdventureWorks_Tables.csv')
    dw_tables = pd.read_csv(data_dir / 'AdventureWorksDW_Tables.csv')

    ddls = {}

    # OLTP tables
    base_oltp = oltp_tables[oltp_tables['TABLE_TYPE'] == 'BASE TABLE']
    for _, tbl in base_oltp.iterrows():
        schema, name = tbl['TABLE_SCHEMA'], tbl['TABLE_NAME']
        full_name = f"OLTP.{schema}.{name}"
        cols = oltp_cols[
            (oltp_cols['TABLE_SCHEMA'] == schema) &
            (oltp_cols['TABLE_NAME'] == name)
        ]
        if not cols.empty:
            ddls[full_name] = synthesize_ddl(f"{schema}.{name}", cols)

    # DW tables
    base_dw = dw_tables[dw_tables['TABLE_TYPE'] == 'BASE TABLE']
    for _, tbl in base_dw.iterrows():
        name = tbl['TABLE_NAME']
        full_name = f"DW.dbo.{name}"
        cols = dw_cols[dw_cols['TABLE_NAME'] == name]
        if not cols.empty:
            ddls[full_name] = synthesize_ddl(f"dbo.{name}", cols)

    return ddls


def build_tpc_ds_ddls() -> dict:
    """Build DDL strings for TPC-DS tables from official TPC-DS v4.0.0 spec.

    Source: github.com/gregrahn/tpcds-kit/blob/master/tools/tpcds.sql
    Every column is included -- no truncation.
    """
    tables = {
        'store_sales': [
            ('ss_sold_date_sk','integer',True),('ss_sold_time_sk','integer',True),
            ('ss_item_sk','integer',False),('ss_customer_sk','integer',True),
            ('ss_cdemo_sk','integer',True),('ss_hdemo_sk','integer',True),
            ('ss_addr_sk','integer',True),('ss_store_sk','integer',True),
            ('ss_promo_sk','integer',True),('ss_ticket_number','integer',False),
            ('ss_quantity','integer',True),('ss_wholesale_cost','decimal(7,2)',True),
            ('ss_list_price','decimal(7,2)',True),('ss_sales_price','decimal(7,2)',True),
            ('ss_ext_discount_amt','decimal(7,2)',True),('ss_ext_sales_price','decimal(7,2)',True),
            ('ss_ext_wholesale_cost','decimal(7,2)',True),('ss_ext_list_price','decimal(7,2)',True),
            ('ss_ext_tax','decimal(7,2)',True),('ss_coupon_amt','decimal(7,2)',True),
            ('ss_net_paid','decimal(7,2)',True),('ss_net_paid_inc_tax','decimal(7,2)',True),
            ('ss_net_profit','decimal(7,2)',True),
        ],
        'store_returns': [
            ('sr_returned_date_sk','integer',True),('sr_return_time_sk','integer',True),
            ('sr_item_sk','integer',False),('sr_customer_sk','integer',True),
            ('sr_cdemo_sk','integer',True),('sr_hdemo_sk','integer',True),
            ('sr_addr_sk','integer',True),('sr_store_sk','integer',True),
            ('sr_reason_sk','integer',True),('sr_ticket_number','integer',False),
            ('sr_return_quantity','integer',True),('sr_return_amt','decimal(7,2)',True),
            ('sr_return_tax','decimal(7,2)',True),('sr_return_amt_inc_tax','decimal(7,2)',True),
            ('sr_fee','decimal(7,2)',True),('sr_return_ship_cost','decimal(7,2)',True),
            ('sr_refunded_cash','decimal(7,2)',True),('sr_reversed_charge','decimal(7,2)',True),
            ('sr_store_credit','decimal(7,2)',True),('sr_net_loss','decimal(7,2)',True),
        ],
        'catalog_sales': [
            ('cs_sold_date_sk','integer',True),('cs_sold_time_sk','integer',True),
            ('cs_ship_date_sk','integer',True),('cs_bill_customer_sk','integer',True),
            ('cs_bill_cdemo_sk','integer',True),('cs_bill_hdemo_sk','integer',True),
            ('cs_bill_addr_sk','integer',True),('cs_ship_customer_sk','integer',True),
            ('cs_ship_cdemo_sk','integer',True),('cs_ship_hdemo_sk','integer',True),
            ('cs_ship_addr_sk','integer',True),('cs_call_center_sk','integer',True),
            ('cs_catalog_page_sk','integer',True),('cs_ship_mode_sk','integer',True),
            ('cs_warehouse_sk','integer',True),('cs_item_sk','integer',False),
            ('cs_promo_sk','integer',True),('cs_order_number','integer',False),
            ('cs_quantity','integer',True),('cs_wholesale_cost','decimal(7,2)',True),
            ('cs_list_price','decimal(7,2)',True),('cs_sales_price','decimal(7,2)',True),
            ('cs_ext_discount_amt','decimal(7,2)',True),('cs_ext_sales_price','decimal(7,2)',True),
            ('cs_ext_wholesale_cost','decimal(7,2)',True),('cs_ext_list_price','decimal(7,2)',True),
            ('cs_ext_tax','decimal(7,2)',True),('cs_coupon_amt','decimal(7,2)',True),
            ('cs_ext_ship_cost','decimal(7,2)',True),('cs_net_paid','decimal(7,2)',True),
            ('cs_net_paid_inc_tax','decimal(7,2)',True),('cs_net_paid_inc_ship','decimal(7,2)',True),
            ('cs_net_paid_inc_ship_tax','decimal(7,2)',True),('cs_net_profit','decimal(7,2)',True),
        ],
        'catalog_returns': [
            ('cr_returned_date_sk','integer',True),('cr_returned_time_sk','integer',True),
            ('cr_item_sk','integer',False),('cr_refunded_customer_sk','integer',True),
            ('cr_refunded_cdemo_sk','integer',True),('cr_refunded_hdemo_sk','integer',True),
            ('cr_refunded_addr_sk','integer',True),('cr_returning_customer_sk','integer',True),
            ('cr_returning_cdemo_sk','integer',True),('cr_returning_hdemo_sk','integer',True),
            ('cr_returning_addr_sk','integer',True),('cr_call_center_sk','integer',True),
            ('cr_catalog_page_sk','integer',True),('cr_ship_mode_sk','integer',True),
            ('cr_warehouse_sk','integer',True),('cr_reason_sk','integer',True),
            ('cr_order_number','integer',False),('cr_return_quantity','integer',True),
            ('cr_return_amount','decimal(7,2)',True),('cr_return_tax','decimal(7,2)',True),
            ('cr_return_amt_inc_tax','decimal(7,2)',True),('cr_fee','decimal(7,2)',True),
            ('cr_return_ship_cost','decimal(7,2)',True),('cr_refunded_cash','decimal(7,2)',True),
            ('cr_reversed_charge','decimal(7,2)',True),('cr_store_credit','decimal(7,2)',True),
            ('cr_net_loss','decimal(7,2)',True),
        ],
        'web_sales': [
            ('ws_sold_date_sk','integer',True),('ws_sold_time_sk','integer',True),
            ('ws_ship_date_sk','integer',True),('ws_item_sk','integer',False),
            ('ws_bill_customer_sk','integer',True),('ws_bill_cdemo_sk','integer',True),
            ('ws_bill_hdemo_sk','integer',True),('ws_bill_addr_sk','integer',True),
            ('ws_ship_customer_sk','integer',True),('ws_ship_cdemo_sk','integer',True),
            ('ws_ship_hdemo_sk','integer',True),('ws_ship_addr_sk','integer',True),
            ('ws_web_page_sk','integer',True),('ws_web_site_sk','integer',True),
            ('ws_ship_mode_sk','integer',True),('ws_warehouse_sk','integer',True),
            ('ws_promo_sk','integer',True),('ws_order_number','integer',False),
            ('ws_quantity','integer',True),('ws_wholesale_cost','decimal(7,2)',True),
            ('ws_list_price','decimal(7,2)',True),('ws_sales_price','decimal(7,2)',True),
            ('ws_ext_discount_amt','decimal(7,2)',True),('ws_ext_sales_price','decimal(7,2)',True),
            ('ws_ext_wholesale_cost','decimal(7,2)',True),('ws_ext_list_price','decimal(7,2)',True),
            ('ws_ext_tax','decimal(7,2)',True),('ws_coupon_amt','decimal(7,2)',True),
            ('ws_ext_ship_cost','decimal(7,2)',True),('ws_net_paid','decimal(7,2)',True),
            ('ws_net_paid_inc_tax','decimal(7,2)',True),('ws_net_paid_inc_ship','decimal(7,2)',True),
            ('ws_net_paid_inc_ship_tax','decimal(7,2)',True),('ws_net_profit','decimal(7,2)',True),
        ],
        'web_returns': [
            ('wr_returned_date_sk','integer',True),('wr_returned_time_sk','integer',True),
            ('wr_item_sk','integer',False),('wr_refunded_customer_sk','integer',True),
            ('wr_refunded_cdemo_sk','integer',True),('wr_refunded_hdemo_sk','integer',True),
            ('wr_refunded_addr_sk','integer',True),('wr_returning_customer_sk','integer',True),
            ('wr_returning_cdemo_sk','integer',True),('wr_returning_hdemo_sk','integer',True),
            ('wr_returning_addr_sk','integer',True),('wr_web_page_sk','integer',True),
            ('wr_reason_sk','integer',True),('wr_order_number','integer',False),
            ('wr_return_quantity','integer',True),('wr_return_amt','decimal(7,2)',True),
            ('wr_return_tax','decimal(7,2)',True),('wr_return_amt_inc_tax','decimal(7,2)',True),
            ('wr_fee','decimal(7,2)',True),('wr_return_ship_cost','decimal(7,2)',True),
            ('wr_refunded_cash','decimal(7,2)',True),('wr_reversed_charge','decimal(7,2)',True),
            ('wr_account_credit','decimal(7,2)',True),('wr_net_loss','decimal(7,2)',True),
        ],
        'inventory': [
            ('inv_date_sk','integer',False),('inv_item_sk','integer',False),
            ('inv_warehouse_sk','integer',False),('inv_quantity_on_hand','integer',True),
        ],
        'date_dim': [
            ('d_date_sk','integer',False),('d_date_id','char(16)',False),
            ('d_date','date',True),('d_month_seq','integer',True),
            ('d_week_seq','integer',True),('d_quarter_seq','integer',True),
            ('d_year','integer',True),('d_dow','integer',True),
            ('d_moy','integer',True),('d_dom','integer',True),
            ('d_qoy','integer',True),('d_fy_year','integer',True),
            ('d_fy_quarter_seq','integer',True),('d_fy_week_seq','integer',True),
            ('d_day_name','char(9)',True),('d_quarter_name','char(6)',True),
            ('d_holiday','char(1)',True),('d_weekend','char(1)',True),
            ('d_following_holiday','char(1)',True),('d_first_dom','integer',True),
            ('d_last_dom','integer',True),('d_same_day_ly','integer',True),
            ('d_same_day_lq','integer',True),('d_current_day','char(1)',True),
            ('d_current_week','char(1)',True),('d_current_month','char(1)',True),
            ('d_current_quarter','char(1)',True),('d_current_year','char(1)',True),
        ],
        'time_dim': [
            ('t_time_sk','integer',False),('t_time_id','char(16)',False),
            ('t_time','integer',True),('t_hour','integer',True),
            ('t_minute','integer',True),('t_second','integer',True),
            ('t_am_pm','char(2)',True),('t_shift','char(20)',True),
            ('t_sub_shift','char(20)',True),('t_meal_time','char(20)',True),
        ],
        'item': [
            ('i_item_sk','integer',False),('i_item_id','char(16)',False),
            ('i_rec_start_date','date',True),('i_rec_end_date','date',True),
            ('i_item_desc','varchar(200)',True),('i_current_price','decimal(7,2)',True),
            ('i_wholesale_cost','decimal(7,2)',True),('i_brand_id','integer',True),
            ('i_brand','char(50)',True),('i_class_id','integer',True),
            ('i_class','char(50)',True),('i_category_id','integer',True),
            ('i_category','char(50)',True),('i_manufact_id','integer',True),
            ('i_manufact','char(50)',True),('i_size','char(20)',True),
            ('i_formulation','char(20)',True),('i_color','char(20)',True),
            ('i_units','char(10)',True),('i_container','char(10)',True),
            ('i_manager_id','integer',True),('i_product_name','char(50)',True),
        ],
        'customer': [
            ('c_customer_sk','integer',False),('c_customer_id','char(16)',False),
            ('c_current_cdemo_sk','integer',True),('c_current_hdemo_sk','integer',True),
            ('c_current_addr_sk','integer',True),('c_first_shipto_date_sk','integer',True),
            ('c_first_sales_date_sk','integer',True),('c_salutation','char(10)',True),
            ('c_first_name','char(20)',True),('c_last_name','char(30)',True),
            ('c_preferred_cust_flag','char(1)',True),('c_birth_day','integer',True),
            ('c_birth_month','integer',True),('c_birth_year','integer',True),
            ('c_birth_country','varchar(20)',True),('c_login','char(13)',True),
            ('c_email_address','char(50)',True),('c_last_review_date_sk','integer',True),
        ],
        'customer_address': [
            ('ca_address_sk','integer',False),('ca_address_id','char(16)',False),
            ('ca_street_number','char(10)',True),('ca_street_name','varchar(60)',True),
            ('ca_street_type','char(15)',True),('ca_suite_number','char(10)',True),
            ('ca_city','varchar(60)',True),('ca_county','varchar(30)',True),
            ('ca_state','char(2)',True),('ca_zip','char(10)',True),
            ('ca_country','varchar(20)',True),('ca_gmt_offset','decimal(5,2)',True),
            ('ca_location_type','char(20)',True),
        ],
        'customer_demographics': [
            ('cd_demo_sk','integer',False),('cd_gender','char(1)',True),
            ('cd_marital_status','char(1)',True),('cd_education_status','char(20)',True),
            ('cd_purchase_estimate','integer',True),('cd_credit_rating','char(10)',True),
            ('cd_dep_count','integer',True),('cd_dep_employed_count','integer',True),
            ('cd_dep_college_count','integer',True),
        ],
        'household_demographics': [
            ('hd_demo_sk','integer',False),('hd_income_band_sk','integer',True),
            ('hd_buy_potential','char(15)',True),('hd_dep_count','integer',True),
            ('hd_vehicle_count','integer',True),
        ],
        'store': [
            ('s_store_sk','integer',False),('s_store_id','char(16)',False),
            ('s_rec_start_date','date',True),('s_rec_end_date','date',True),
            ('s_closed_date_sk','integer',True),('s_store_name','varchar(50)',True),
            ('s_number_employees','integer',True),('s_floor_space','integer',True),
            ('s_hours','char(20)',True),('s_manager','varchar(40)',True),
            ('s_market_id','integer',True),('s_geography_class','varchar(100)',True),
            ('s_market_desc','varchar(100)',True),('s_market_manager','varchar(40)',True),
            ('s_division_id','integer',True),('s_division_name','varchar(50)',True),
            ('s_company_id','integer',True),('s_company_name','varchar(50)',True),
            ('s_street_number','varchar(10)',True),('s_street_name','varchar(60)',True),
            ('s_street_type','char(15)',True),('s_suite_number','char(10)',True),
            ('s_city','varchar(60)',True),('s_county','varchar(30)',True),
            ('s_state','char(2)',True),('s_zip','char(10)',True),
            ('s_country','varchar(20)',True),('s_gmt_offset','decimal(5,2)',True),
            ('s_tax_precentage','decimal(5,2)',True),
        ],
        'promotion': [
            ('p_promo_sk','integer',False),('p_promo_id','char(16)',False),
            ('p_start_date_sk','integer',True),('p_end_date_sk','integer',True),
            ('p_item_sk','integer',True),('p_cost','decimal(15,2)',True),
            ('p_response_target','integer',True),('p_promo_name','char(50)',True),
            ('p_channel_dmail','char(1)',True),('p_channel_email','char(1)',True),
            ('p_channel_catalog','char(1)',True),('p_channel_tv','char(1)',True),
            ('p_channel_radio','char(1)',True),('p_channel_press','char(1)',True),
            ('p_channel_event','char(1)',True),('p_channel_demo','char(1)',True),
            ('p_channel_details','varchar(100)',True),('p_purpose','char(15)',True),
            ('p_discount_active','char(1)',True),
        ],
        'warehouse': [
            ('w_warehouse_sk','integer',False),('w_warehouse_id','char(16)',False),
            ('w_warehouse_name','varchar(20)',True),('w_warehouse_sq_ft','integer',True),
            ('w_street_number','char(10)',True),('w_street_name','varchar(60)',True),
            ('w_street_type','char(15)',True),('w_suite_number','char(10)',True),
            ('w_city','varchar(60)',True),('w_county','varchar(30)',True),
            ('w_state','char(2)',True),('w_zip','char(10)',True),
            ('w_country','varchar(20)',True),('w_gmt_offset','decimal(5,2)',True),
        ],
        'ship_mode': [
            ('sm_ship_mode_sk','integer',False),('sm_ship_mode_id','char(16)',False),
            ('sm_type','char(30)',True),('sm_code','char(10)',True),
            ('sm_carrier','char(20)',True),('sm_contract','char(20)',True),
        ],
        'reason': [
            ('r_reason_sk','integer',False),('r_reason_id','char(16)',False),
            ('r_reason_desc','char(100)',True),
        ],
        'income_band': [
            ('ib_income_band_sk','integer',False),('ib_lower_bound','integer',True),
            ('ib_upper_bound','integer',True),
        ],
        'call_center': [
            ('cc_call_center_sk','integer',False),('cc_call_center_id','char(16)',False),
            ('cc_rec_start_date','date',True),('cc_rec_end_date','date',True),
            ('cc_closed_date_sk','integer',True),('cc_open_date_sk','integer',True),
            ('cc_name','varchar(50)',True),('cc_class','varchar(50)',True),
            ('cc_employees','integer',True),('cc_sq_ft','integer',True),
            ('cc_hours','char(20)',True),('cc_manager','varchar(40)',True),
            ('cc_mkt_id','integer',True),('cc_mkt_class','char(50)',True),
            ('cc_mkt_desc','varchar(100)',True),('cc_market_manager','varchar(40)',True),
            ('cc_division','integer',True),('cc_division_name','varchar(50)',True),
            ('cc_company','integer',True),('cc_company_name','char(50)',True),
            ('cc_street_number','char(10)',True),('cc_street_name','varchar(60)',True),
            ('cc_street_type','char(15)',True),('cc_suite_number','char(10)',True),
            ('cc_city','varchar(60)',True),('cc_county','varchar(30)',True),
            ('cc_state','char(2)',True),('cc_zip','char(10)',True),
            ('cc_country','varchar(20)',True),('cc_gmt_offset','decimal(5,2)',True),
            ('cc_tax_percentage','decimal(5,2)',True),
        ],
        'catalog_page': [
            ('cp_catalog_page_sk','integer',False),('cp_catalog_page_id','char(16)',False),
            ('cp_start_date_sk','integer',True),('cp_end_date_sk','integer',True),
            ('cp_department','varchar(50)',True),('cp_catalog_number','integer',True),
            ('cp_catalog_page_number','integer',True),('cp_description','varchar(100)',True),
            ('cp_type','varchar(100)',True),
        ],
        'web_site': [
            ('web_site_sk','integer',False),('web_site_id','char(16)',False),
            ('web_rec_start_date','date',True),('web_rec_end_date','date',True),
            ('web_name','varchar(50)',True),('web_open_date_sk','integer',True),
            ('web_close_date_sk','integer',True),('web_class','varchar(50)',True),
            ('web_manager','varchar(40)',True),('web_mkt_id','integer',True),
            ('web_mkt_class','varchar(50)',True),('web_mkt_desc','varchar(100)',True),
            ('web_market_manager','varchar(40)',True),('web_company_id','integer',True),
            ('web_company_name','char(50)',True),('web_street_number','char(10)',True),
            ('web_street_name','varchar(60)',True),('web_street_type','char(15)',True),
            ('web_suite_number','char(10)',True),('web_city','varchar(60)',True),
            ('web_county','varchar(30)',True),('web_state','char(2)',True),
            ('web_zip','char(10)',True),('web_country','varchar(20)',True),
            ('web_gmt_offset','decimal(5,2)',True),('web_tax_percentage','decimal(5,2)',True),
        ],
        'web_page': [
            ('wp_web_page_sk','integer',False),('wp_web_page_id','char(16)',False),
            ('wp_rec_start_date','date',True),('wp_rec_end_date','date',True),
            ('wp_creation_date_sk','integer',True),('wp_access_date_sk','integer',True),
            ('wp_autogen_flag','char(1)',True),('wp_customer_sk','integer',True),
            ('wp_url','varchar(100)',True),('wp_type','char(50)',True),
            ('wp_char_count','integer',True),('wp_link_count','integer',True),
            ('wp_image_count','integer',True),('wp_max_ad_count','integer',True),
        ],
    }

    ddls = {}
    for table_name, cols in tables.items():
        lines = [f"CREATE TABLE {table_name} ("]
        for col_name, col_type, nullable in cols:
            null_str = "NULL" if nullable else "NOT NULL"
            lines.append(f"    {col_name} {col_type} {null_str},")
        lines[-1] = lines[-1].rstrip(',')
        lines.append(");")
        ddls[table_name] = "\n".join(lines)
    return ddls

def build_tpc_di_ddls(data_dir) -> dict:
    """Build DDL strings for TPC-DI tables from CSV columns + source nodes."""
    import json
    from pathlib import Path
    data_dir = Path(data_dir)
    cols_df = pd.read_csv(data_dir / 'TPCDI_Columns.csv')
    tables_df = pd.read_csv(data_dir / 'TPCDI_Tables.csv')

    ddls = {}
    for _, tbl in tables_df.iterrows():
        name = tbl['TABLE_NAME']
        tbl_cols = cols_df[cols_df['TABLE_NAME'] == name]
        if not tbl_cols.empty:
            ddls[name] = synthesize_ddl(name, tbl_cols)

    # Add source file descriptions (SRC.* nodes)
    lineage_file = data_dir / 'lineage_edges.json'
    if lineage_file.exists():
        lineage = json.load(open(lineage_file, 'r', encoding='utf-8'))
        for edge in lineage:
            src = edge.get('source', '')
            if src.startswith('SRC.') and src not in ddls:
                evidence = edge.get('evidence', '')
                ddls[src] = f"-- Source file: {src}\n-- {evidence}"

    return ddls

def embed_ddls(ddls: dict, model_name: str = 'all-MiniLM-L6-v2') -> dict:
    """Embed DDL strings using a sentence transformer.

    Returns dict of {table_name: embedding_tensor(384,)}
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    names = list(ddls.keys())
    texts = [ddls[n] for n in names]

    print(f"  Embedding {len(texts)} DDL strings with {model_name}...")
    embeddings = model.encode(texts, show_progress_bar=True,
                              convert_to_tensor=True)

    return {name: embeddings[i] for i, name in enumerate(names)}


def enrich_schema_graph(graph_path: Path, ddl_embeddings: dict) -> None:
    """Load a schema_graph.pt, concatenate DDL embeddings with structural
    features, and save as enriched_schema_graph.pt."""

    data = torch.load(graph_path, weights_only=False)
    table_names = data['table'].table_names
    structural_x = data['table'].x  # (N, 6)

    ddl_vecs = []
    missing = []
    for name in table_names:
        if name in ddl_embeddings:
            ddl_vecs.append(ddl_embeddings[name].cpu())
        else:
            missing.append(name)
            # Fallback: zero vector
            ddl_vecs.append(torch.zeros(384))

    if missing:
        print(f"  ⚠️  {len(missing)} tables missing DDL embeddings: "
              f"{missing[:5]}...")

    ddl_tensor = torch.stack(ddl_vecs)  # (N, 384)

    # Concatenate: [DDL_embedding(384) | structural(6)] = 390-dim
    enriched_x = torch.cat([ddl_tensor, structural_x], dim=1)
    data['table'].x = enriched_x

    # Save enriched version
    out_path = graph_path.parent / 'enriched_schema_graph.pt'
    torch.save(data, out_path)

    print(f"  ✅ Enriched features: {list(enriched_x.shape)}")
    print(f"  Saved to: {out_path}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Compute enriched node features',
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['adventureworks', 'tpc-ds', 'tpc-di', 'all'],
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    datasets_to_process = (
        ['adventureworks', 'tpc-ds', 'tpc-di']
        if args.dataset == 'all'
        else [args.dataset]
    )

    for ds in datasets_to_process:
        print(f"\n{'=' * 60}")
        print(f"Processing: {ds}")
        print(f"{'=' * 60}")

        ds_dir = repo_root / 'datasets' / ds
        graph_path = ds_dir / 'schema_graph.pt'

        if not graph_path.exists():
            print(f"  ❌ {graph_path} not found. Run convert_schema_graph.py "
                  f"first.")
            continue

        # Build DDLs
        print(f"  Synthesizing DDL strings...")
        if ds == 'adventureworks':
            ddls = build_adventureworks_ddls(ds_dir)
        elif ds == 'tpc-ds':
            ddls = build_tpc_ds_ddls()
        elif ds == 'tpc-di':
            ddls = build_tpc_di_ddls(ds_dir)
        else:
            print(f"  ❌ Unknown dataset: {ds}")
            continue

        print(f"  Generated {len(ddls)} DDL statements")

        # Embed DDLs
        print(f"  Computing sentence embeddings...")
        ddl_embeddings = embed_ddls(ddls)

        # Enrich graph
        print(f"  Enriching schema graph...")
        enrich_schema_graph(graph_path, ddl_embeddings)


if __name__ == '__main__':
    main()
