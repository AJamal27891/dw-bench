"""Generate synthetic row-level value data for Syn-Logistics.

Creates realistic CSV data for all 64 Syn-Logistics tables, with
domain-appropriate columns per silo (Logistics, HR, Healthcare,
E-commerce, Finance) and per layer (raw, staging, core, mart).

All data is deterministic (seeded RNG) for reproducibility.

Usage:
    python scripts/generate_value_data.py
    python scripts/generate_value_data.py --validate
"""
import argparse
import csv
import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

REPO_ROOT = Path(__file__).parent.parent
OUT_DIR = REPO_ROOT / 'datasets' / 'syn_logistics' / 'value_data'

# ──────────────────────────────────────────────────────────────────
# Row counts per layer
# ──────────────────────────────────────────────────────────────────
ROW_COUNTS = {
    'raw': (5000, 10000),
    'staging': (3000, 8000),
    'core_dim': (500, 2000),
    'core_fact': (1000, 5000),
    'mart': (200, 1000),
}


def row_count_for(table_name: str, layer: str) -> int:
    """Deterministic row count per table."""
    rng = random.Random(hash(table_name) & 0xFFFFFFFF)
    if layer == 'core':
        key = 'core_fact' if table_name.startswith('fact_') else 'core_dim'
    else:
        key = layer
    lo, hi = ROW_COUNTS[key]
    return rng.randint(lo, hi)


# ──────────────────────────────────────────────────────────────────
# Names / vocabulary pools (deterministic)
# ──────────────────────────────────────────────────────────────────
SUPPLIER_NAMES = [
    'Acme Corp', 'GlobalTrade Inc', 'FastFreight LLC', 'PrimeParts Co',
    'SteelWorks Ltd', 'TechSupply AG', 'BlueLine Services', 'Atlas Shipping',
    'OceanCargo GmbH', 'NorthStar Materials', 'RedHawk Logistics',
    'Summit Distributors', 'EaglePack Inc', 'Horizon Wholesale',
    'AnchorPoint Supply', 'SwiftMove Transport', 'IronBridge Metals',
    'Pacific Commodities', 'UrbanEdge Tech', 'QuantumSource Ltd',
]

PRODUCT_NAMES = [
    'Widget-A', 'Widget-B', 'Gizmo-X1', 'Gizmo-X2', 'Bracket-M5',
    'Cable-Cat6', 'Sensor-UV', 'Motor-DC12', 'Pump-HP3', 'Valve-SS316',
    'Filter-HEPA', 'Board-PCB7', 'Switch-RF', 'Relay-5V', 'Bushing-Nylon',
    'Bearing-6205', 'Gasket-FKM', 'Coupling-Flex', 'Actuator-Lin',
    'Thermostat-K', 'Capacitor-100uF', 'Resistor-10K', 'Diode-Zener',
    'Connector-USB', 'Housing-ABS',
]

WAREHOUSE_NAMES = [
    'WH-East', 'WH-West', 'WH-Central', 'WH-North', 'WH-South',
    'WH-Pacific', 'WH-Atlantic', 'WH-Metro', 'WH-Suburb', 'WH-Industrial',
]

CARRIER_NAMES = [
    'FedEx', 'UPS', 'DHL', 'USPS', 'Maersk', 'FedEx Freight',
    'XPO Logistics', 'J.B. Hunt', 'Ryder System', 'Old Dominion',
]

REGIONS = ['North', 'South', 'East', 'West', 'Central']

EMPLOYEE_NAMES = [
    'Alice Johnson', 'Bob Smith', 'Carol Williams', 'David Brown',
    'Eva Martinez', 'Frank Davis', 'Grace Wilson', 'Henry Miller',
    'Iris Taylor', 'Jack Anderson', 'Karen Thomas', 'Leo Jackson',
    'Mia White', 'Noah Harris', 'Olivia Martin', 'Paul Garcia',
    'Quinn Robinson', 'Rose Clark', 'Sam Lewis', 'Tina Walker',
]

DEPARTMENTS = [
    'Engineering', 'Sales', 'Marketing', 'Finance', 'HR',
    'Operations', 'Legal', 'R&D', 'Customer Support', 'IT',
]

JOB_TITLES = [
    'Analyst', 'Manager', 'Director', 'VP', 'Engineer',
    'Coordinator', 'Specialist', 'Lead', 'Associate', 'Consultant',
]

PATIENT_NAMES = [
    'John Doe', 'Jane Roe', 'Robert Patient', 'Mary Subject',
    'James Test', 'Sarah Sample', 'Michael Case', 'Emily Study',
    'William Trial', 'Jessica Record', 'Daniel Health', 'Amanda Care',
    'Christopher Med', 'Jennifer Nurse', 'Matthew Doctor', 'Ashley Clinic',
]

DIAGNOSES = [
    'Hypertension', 'Type-2 Diabetes', 'Asthma', 'COPD',
    'Heart Failure', 'Pneumonia', 'UTI', 'Migraine',
    'Back Pain', 'Anxiety', 'Depression', 'Arthritis',
]

MEDICATIONS = [
    'Lisinopril', 'Metformin', 'Albuterol', 'Tiotropium',
    'Furosemide', 'Amoxicillin', 'Ciprofloxacin', 'Sumatriptan',
    'Ibuprofen', 'Sertraline', 'Escitalopram', 'Methotrexate',
]

LAB_TESTS = [
    'CBC', 'BMP', 'CMP', 'Lipid Panel', 'HbA1c',
    'TSH', 'Urinalysis', 'Liver Panel', 'Coag Panel', 'CRP',
]

PRODUCT_CATEGORIES = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']

PAYMENT_METHODS = ['credit_card', 'debit_card', 'paypal', 'bank_transfer', 'crypto']

ACCOUNT_TYPES = ['Asset', 'Liability', 'Equity', 'Revenue', 'Expense']

COST_CENTERS = [
    'CC-1001', 'CC-1002', 'CC-1003', 'CC-1004', 'CC-1005',
    'CC-2001', 'CC-2002', 'CC-2003', 'CC-3001', 'CC-3002',
]

SOURCE_SYSTEMS = ['SAP', 'Oracle', 'Salesforce', 'Custom_ERP', 'Legacy_DB']


# ──────────────────────────────────────────────────────────────────
# Date helpers
# ──────────────────────────────────────────────────────────────────
BASE_DATE = datetime(2023, 1, 1)


def rand_date(rng, days_range=730):
    return (BASE_DATE + timedelta(days=rng.randint(0, days_range))).strftime('%Y-%m-%d')


def rand_timestamp(rng, days_range=730):
    dt = BASE_DATE + timedelta(
        days=rng.randint(0, days_range),
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59),
        seconds=rng.randint(0, 59),
    )
    return dt.strftime('%Y-%m-%d %H:%M:%S')


# ──────────────────────────────────────────────────────────────────
# Column generators per silo × layer
# ──────────────────────────────────────────────────────────────────

def gen_logistics_raw(table_name: str, n: int, rng: random.Random) -> tuple:
    """Generate raw logistics data."""
    if 'purchase_order' in table_name:
        headers = ['id', 'order_id', 'supplier_name', 'product_sku',
                    'quantity', 'unit_price', 'order_date', 'region',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'PO-{rng.randint(10000, 99999)}',
                rng.choice(SUPPLIER_NAMES), rng.choice(PRODUCT_NAMES),
                rng.randint(1, 500), round(rng.uniform(5, 5000), 2),
                rand_date(rng), rng.choice(REGIONS),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    elif 'shipment' in table_name:
        headers = ['id', 'shipment_id', 'carrier', 'origin_warehouse',
                    'destination', 'weight_kg', 'ship_date', 'delivery_date',
                    'status', 'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            ship_d = rng.randint(0, 700)
            rows.append([
                i, f'SH-{rng.randint(10000, 99999)}',
                rng.choice(CARRIER_NAMES), rng.choice(WAREHOUSE_NAMES),
                f'City-{rng.randint(1, 50)}', round(rng.uniform(0.5, 2000), 1),
                (BASE_DATE + timedelta(days=ship_d)).strftime('%Y-%m-%d'),
                (BASE_DATE + timedelta(days=ship_d + rng.randint(1, 14))).strftime('%Y-%m-%d'),
                rng.choice(['delivered', 'in_transit', 'delayed', 'returned']),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    elif 'supplier' in table_name:
        headers = ['id', 'supplier_code', 'supplier_name', 'country',
                    'category', 'rating', 'active',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'SUP-{rng.randint(1000, 9999)}',
                rng.choice(SUPPLIER_NAMES), rng.choice(['US', 'DE', 'CN', 'JP', 'IN', 'BR']),
                rng.choice(['raw_materials', 'components', 'packaging', 'services']),
                round(rng.uniform(1, 5), 1), rng.choice([True, True, True, False]),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    elif 'warehouse' in table_name:
        headers = ['id', 'warehouse_code', 'warehouse_name', 'region',
                    'capacity_sqft', 'current_util_pct',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'WH-{rng.randint(100, 999)}',
                rng.choice(WAREHOUSE_NAMES), rng.choice(REGIONS),
                rng.randint(10000, 500000), round(rng.uniform(20, 98), 1),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    elif 'freight' in table_name:
        headers = ['id', 'invoice_id', 'carrier', 'amount',
                    'currency', 'invoice_date', 'paid',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'FI-{rng.randint(10000, 99999)}',
                rng.choice(CARRIER_NAMES), round(rng.uniform(50, 25000), 2),
                rng.choice(['USD', 'EUR', 'GBP']),
                rand_date(rng), rng.choice([True, True, False]),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    return _gen_generic_raw(table_name, n, rng)


def gen_hr_raw(table_name: str, n: int, rng: random.Random) -> tuple:
    if 'employee' in table_name:
        headers = ['id', 'employee_id', 'full_name', 'hire_date',
                    'department', 'job_title', 'status',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'EMP-{rng.randint(1000, 9999)}',
                rng.choice(EMPLOYEE_NAMES), rand_date(rng),
                rng.choice(DEPARTMENTS), rng.choice(JOB_TITLES),
                rng.choice(['active', 'active', 'active', 'terminated', 'on_leave']),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    elif 'department' in table_name:
        headers = ['id', 'dept_code', 'dept_name', 'parent_dept',
                    'head_count', 'budget',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'DEPT-{rng.randint(100, 999)}',
                rng.choice(DEPARTMENTS),
                rng.choice(DEPARTMENTS + [None]),
                rng.randint(5, 200), round(rng.uniform(100000, 5000000), 2),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    elif 'compensation' in table_name:
        headers = ['id', 'employee_id', 'salary', 'bonus',
                    'pay_grade', 'effective_date',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'EMP-{rng.randint(1000, 9999)}',
                round(rng.uniform(40000, 200000), 2),
                round(rng.uniform(0, 50000), 2),
                rng.choice(['G1', 'G2', 'G3', 'G4', 'G5']),
                rand_date(rng),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    return _gen_generic_raw(table_name, n, rng)


def gen_healthcare_raw(table_name: str, n: int, rng: random.Random) -> tuple:
    if 'patient' in table_name:
        headers = ['id', 'patient_id', 'patient_name', 'dob',
                    'gender', 'blood_type', 'admission_date',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'PAT-{rng.randint(10000, 99999)}',
                rng.choice(PATIENT_NAMES),
                (BASE_DATE - timedelta(days=rng.randint(7300, 36500))).strftime('%Y-%m-%d'),
                rng.choice(['M', 'F']),
                rng.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']),
                rand_date(rng),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    elif 'lab' in table_name:
        headers = ['id', 'test_id', 'patient_id', 'test_name',
                    'result_value', 'unit', 'test_date', 'abnormal',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'LAB-{rng.randint(10000, 99999)}',
                f'PAT-{rng.randint(10000, 99999)}',
                rng.choice(LAB_TESTS),
                round(rng.uniform(0.1, 500), 2),
                rng.choice(['mg/dL', 'mmol/L', '%', 'g/dL', 'U/L']),
                rand_date(rng), rng.choice([True, False, False, False]),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    elif 'prescription' in table_name:
        headers = ['id', 'rx_id', 'patient_id', 'medication',
                    'dosage', 'frequency', 'prescriber', 'rx_date',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'RX-{rng.randint(10000, 99999)}',
                f'PAT-{rng.randint(10000, 99999)}',
                rng.choice(MEDICATIONS),
                f'{rng.choice([5, 10, 20, 50, 100, 250, 500])}mg',
                rng.choice(['daily', 'BID', 'TID', 'PRN', 'weekly']),
                f'Dr. {rng.choice(EMPLOYEE_NAMES).split()[1]}',
                rand_date(rng),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    return _gen_generic_raw(table_name, n, rng)


def gen_ecommerce_raw(table_name: str, n: int, rng: random.Random) -> tuple:
    if 'cart' in table_name or 'session' in table_name:
        headers = ['id', 'session_id', 'customer_id', 'product_id',
                    'quantity', 'session_start', 'session_end',
                    'device', 'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'SESS-{rng.randint(100000, 999999)}',
                f'CUST-{rng.randint(1000, 9999)}',
                f'PROD-{rng.randint(100, 999)}',
                rng.randint(1, 10),
                rand_timestamp(rng), rand_timestamp(rng),
                rng.choice(['desktop', 'mobile', 'tablet']),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    elif 'product' in table_name or 'catalog' in table_name:
        headers = ['id', 'product_code', 'product_name', 'category',
                    'price', 'in_stock', 'brand',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'PROD-{rng.randint(100, 999)}',
                f'{rng.choice(["Deluxe", "Basic", "Pro", "Ultra", "Mini"])} '
                f'{rng.choice(["Widget", "Gadget", "Tool", "Device"])}',
                rng.choice(PRODUCT_CATEGORIES),
                round(rng.uniform(5, 2000), 2),
                rng.choice([True, True, True, False]),
                rng.choice(['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    elif 'payment' in table_name:
        headers = ['id', 'transaction_id', 'customer_id', 'amount',
                    'payment_method', 'status', 'payment_date',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'TXN-{rng.randint(100000, 999999)}',
                f'CUST-{rng.randint(1000, 9999)}',
                round(rng.uniform(5, 5000), 2),
                rng.choice(PAYMENT_METHODS),
                rng.choice(['completed', 'completed', 'pending', 'failed', 'refunded']),
                rand_date(rng),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    return _gen_generic_raw(table_name, n, rng)


def gen_finance_raw(table_name: str, n: int, rng: random.Random) -> tuple:
    if 'ledger' in table_name:
        headers = ['id', 'entry_id', 'account_code', 'debit', 'credit',
                    'description', 'posting_date',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            amt = round(rng.uniform(10, 100000), 2)
            is_debit = rng.choice([True, False])
            rows.append([
                i, f'JE-{rng.randint(10000, 99999)}',
                f'{rng.randint(1000, 9999)}',
                amt if is_debit else 0,
                0 if is_debit else amt,
                rng.choice(['Sales revenue', 'COGS', 'Salary expense',
                            'Rent', 'Utilities', 'Depreciation', 'Interest']),
                rand_date(rng),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    elif 'bank' in table_name:
        headers = ['id', 'txn_ref', 'bank_account', 'amount',
                    'txn_type', 'txn_date', 'cleared',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            rows.append([
                i, f'BK-{rng.randint(100000, 999999)}',
                f'ACCT-{rng.randint(1000, 9999)}',
                round(rng.uniform(-50000, 50000), 2),
                rng.choice(['deposit', 'withdrawal', 'transfer', 'fee']),
                rand_date(rng), rng.choice([True, True, True, False]),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    elif 'tax' in table_name:
        headers = ['id', 'filing_id', 'tax_period', 'tax_type',
                    'taxable_amount', 'tax_due', 'status',
                    'source_system', 'load_timestamp']
        rows = []
        for i in range(n):
            taxable = round(rng.uniform(10000, 1000000), 2)
            rows.append([
                i, f'TAX-{rng.randint(10000, 99999)}',
                rng.choice(['Q1-2023', 'Q2-2023', 'Q3-2023', 'Q4-2023',
                            'Q1-2024', 'Q2-2024', 'Q3-2024', 'Q4-2024']),
                rng.choice(['income', 'sales', 'payroll', 'property']),
                taxable, round(taxable * rng.uniform(0.05, 0.35), 2),
                rng.choice(['filed', 'pending', 'audited']),
                rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            ])
        return headers, rows
    return _gen_generic_raw(table_name, n, rng)


def _gen_generic_raw(table_name: str, n: int, rng: random.Random) -> tuple:
    """Fallback raw generator."""
    headers = ['id', 'source_system', 'load_timestamp',
               'raw_value_1', 'raw_value_2', 'batch_id']
    rows = []
    for i in range(n):
        rows.append([
            i, rng.choice(SOURCE_SYSTEMS), rand_timestamp(rng),
            f'val_{rng.randint(1, 10000)}',
            round(rng.uniform(0, 1000), 2),
            rng.randint(1, 100),
        ])
    return headers, rows


# ──────────────────────────────────────────────────────────────────
# Staging, Core, Mart generators (generic — derived from raw)
# ──────────────────────────────────────────────────────────────────

def gen_staging(table_name: str, n: int, rng: random.Random) -> tuple:
    headers = ['id', 'source_id', 'validated_flag', 'cleaned_value',
               'quality_score', 'etl_timestamp', 'batch_id']
    rows = []
    for i in range(n):
        rows.append([
            i, rng.randint(0, 9999),
            rng.choice([True, True, True, True, False]),
            f'cleaned_{rng.randint(1, 10000)}',
            round(rng.uniform(0.5, 1.0), 3),
            rand_timestamp(rng), rng.randint(1, 200),
        ])
    return headers, rows


def gen_core_dim(table_name: str, n: int, rng: random.Random) -> tuple:
    headers = ['surrogate_key', 'business_key', 'name', 'category',
               'effective_date', 'expiry_date', 'is_current']
    rows = []

    # Domain-aware names
    if 'supplier' in table_name:
        pool = SUPPLIER_NAMES
    elif 'product' in table_name:
        pool = PRODUCT_NAMES
    elif 'warehouse' in table_name:
        pool = WAREHOUSE_NAMES
    elif 'carrier' in table_name:
        pool = CARRIER_NAMES
    elif 'employee' in table_name:
        pool = EMPLOYEE_NAMES
    elif 'department' in table_name:
        pool = DEPARTMENTS
    elif 'patient' in table_name:
        pool = PATIENT_NAMES
    elif 'provider' in table_name:
        pool = [f'Dr. {n.split()[1]}' for n in EMPLOYEE_NAMES]
    elif 'customer' in table_name:
        pool = [f'Customer_{i}' for i in range(200)]
    elif 'account' in table_name:
        pool = [f'Account-{t}-{i}' for t in ACCOUNT_TYPES for i in range(40)]
    elif 'cost_center' in table_name:
        pool = COST_CENTERS
    else:
        pool = [f'Entity_{i}' for i in range(200)]

    for i in range(n):
        eff = rng.randint(0, 500)
        rows.append([
            i, f'BK-{rng.randint(1000, 99999)}',
            pool[i % len(pool)],
            rng.choice(REGIONS + PRODUCT_CATEGORIES),
            (BASE_DATE + timedelta(days=eff)).strftime('%Y-%m-%d'),
            (BASE_DATE + timedelta(days=eff + rng.randint(30, 730))).strftime('%Y-%m-%d'),
            rng.choice([True, True, True, False]),
        ])
    return headers, rows


def gen_core_fact(table_name: str, n: int, rng: random.Random,
                  dim_counts: dict) -> tuple:
    """Generate fact table rows with FK references to dims."""
    # Find which dims this fact references (from the SILOS structure)
    silo_dims = {k: v for k, v in dim_counts.items() if k.startswith('dim_')}

    dim_names = sorted(silo_dims.keys())
    headers = ['fact_key'] + [f'{d}_key' for d in dim_names] + \
              ['measure_1', 'measure_2', 'measure_3', 'period_date']
    rows = []
    for i in range(n):
        row = [i]
        for d in dim_names:
            row.append(rng.randint(0, silo_dims[d] - 1))
        row.extend([
            round(rng.uniform(10, 50000), 2),
            round(rng.uniform(1, 1000), 2),
            rng.randint(1, 100),
            rand_date(rng),
        ])
        rows.append(row)
    return headers, rows


def gen_mart(table_name: str, n: int, rng: random.Random) -> tuple:
    headers = ['metric_key', 'dimension_key', 'kpi_value',
               'period', 'aggregation_level', 'refresh_timestamp']
    rows = []
    periods = ['2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4',
               '2024-Q1', '2024-Q2', '2024-Q3', '2024-Q4']
    agg_levels = ['daily', 'weekly', 'monthly', 'quarterly']
    for i in range(n):
        rows.append([
            i, rng.randint(0, 99),
            round(rng.uniform(100, 1000000), 2),
            rng.choice(periods),
            rng.choice(agg_levels),
            rand_timestamp(rng),
        ])
    return headers, rows


# ──────────────────────────────────────────────────────────────────
# SILOS definition (mirrors build_syn_logistics.py)
# ──────────────────────────────────────────────────────────────────
SILOS = {
    'logistics': {
        'raw': ['raw_purchase_orders', 'raw_shipment_tracking', 'raw_supplier_catalog',
                'raw_warehouse_inventory', 'raw_freight_invoices'],
        'staging': ['stg_orders_cleaned', 'stg_shipments_validated', 'stg_suppliers_deduped',
                    'stg_inventory_snapshot', 'stg_freight_reconciled'],
        'core': ['dim_supplier', 'dim_product', 'dim_warehouse', 'dim_carrier',
                 'fact_shipment', 'fact_purchase_order'],
        'mart': ['mart_delivery_performance', 'mart_procurement_cost',
                 'mart_supplier_scorecard', 'mart_inventory_turnover'],
    },
    'hr': {
        'raw': ['raw_employee_records', 'raw_department_hierarchy',
                'raw_compensation_data'],
        'staging': ['stg_employees_cleaned', 'stg_departments_mapped',
                    'stg_compensation_normalized'],
        'core': ['dim_employee', 'dim_department', 'fact_payroll'],
        'mart': ['mart_headcount_report', 'mart_attrition_analysis'],
    },
    'healthcare': {
        'raw': ['raw_patient_intake', 'raw_lab_results', 'raw_prescriptions'],
        'staging': ['stg_patients_validated', 'stg_labs_standardized',
                    'stg_prescriptions_coded'],
        'core': ['dim_patient', 'dim_provider', 'fact_encounter'],
        'mart': ['mart_readmission_risk', 'mart_treatment_outcomes'],
    },
    'ecommerce': {
        'raw': ['raw_cart_sessions', 'raw_product_catalog', 'raw_payment_events'],
        'staging': ['stg_sessions_enriched', 'stg_products_normalized',
                    'stg_payments_validated'],
        'core': ['dim_customer', 'dim_product_listing', 'fact_transaction'],
        'mart': ['mart_conversion_funnel', 'mart_revenue_dashboard'],
    },
    'finance': {
        'raw': ['raw_ledger_entries', 'raw_bank_feeds', 'raw_tax_filings'],
        'staging': ['stg_ledger_reconciled', 'stg_bank_matched',
                    'stg_tax_validated'],
        'core': ['dim_account', 'dim_cost_center', 'fact_journal_entry'],
        'mart': ['mart_trial_balance', 'mart_cash_flow_statement'],
    },
}

RAW_GENERATORS = {
    'logistics': gen_logistics_raw,
    'hr': gen_hr_raw,
    'healthcare': gen_healthcare_raw,
    'ecommerce': gen_ecommerce_raw,
    'finance': gen_finance_raw,
}


def generate_all():
    """Generate CSVs for all 64 tables."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_tables = 0
    total_rows = 0
    manifest = {}  # table_name -> {rows, columns, file}

    for silo_name, layers in SILOS.items():
        print(f'\n  Silo: {silo_name}')

        # First pass: compute dim row counts for fact table FK references
        dim_counts = {}
        for table_name in layers.get('core', []):
            if table_name.startswith('dim_'):
                n = row_count_for(table_name, 'core')
                dim_counts[table_name] = n

        for layer_name in ['raw', 'staging', 'core', 'mart']:
            for table_name in layers.get(layer_name, []):
                n = row_count_for(table_name, layer_name)
                rng = random.Random(hash(table_name) & 0xFFFFFFFF)

                # Generate table data
                if layer_name == 'raw':
                    gen_fn = RAW_GENERATORS.get(silo_name, _gen_generic_raw)
                    headers, rows = gen_fn(table_name, n, rng)
                elif layer_name == 'staging':
                    headers, rows = gen_staging(table_name, n, rng)
                elif layer_name == 'core':
                    if table_name.startswith('fact_'):
                        headers, rows = gen_core_fact(table_name, n, rng,
                                                     dim_counts)
                    else:
                        headers, rows = gen_core_dim(table_name, n, rng)
                elif layer_name == 'mart':
                    headers, rows = gen_mart(table_name, n, rng)

                # Write CSV
                csv_path = OUT_DIR / f'{table_name}.csv'
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(rows)

                manifest[table_name] = {
                    'rows': len(rows),
                    'columns': len(headers),
                    'layer': layer_name,
                    'silo': silo_name,
                    'file': f'{table_name}.csv',
                }
                total_tables += 1
                total_rows += len(rows)
                print(f'    {table_name:40s}: {len(rows):>6,} rows × {len(headers)} cols')

    # Save manifest
    manifest_path = OUT_DIR / '_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f'\n  Total: {total_tables} tables, {total_rows:,} rows')
    print(f'  Saved to: {OUT_DIR}')
    print(f'  Manifest: {manifest_path}')
    return manifest


def validate(manifest: dict = None):
    """Validate generated CSVs."""
    if manifest is None:
        manifest_path = OUT_DIR / '_manifest.json'
        if not manifest_path.exists():
            print('❌ No manifest found. Run generation first.')
            return False
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

    errors = []
    for table_name, info in manifest.items():
        csv_path = OUT_DIR / info['file']
        if not csv_path.exists():
            errors.append(f'Missing: {csv_path}')
            continue
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            row_count = sum(1 for _ in reader)

        if len(header) != info['columns']:
            errors.append(f'{table_name}: column count {len(header)} != {info["columns"]}')
        if row_count != info['rows']:
            errors.append(f'{table_name}: row count {row_count} != {info["rows"]}')

    if errors:
        print(f'❌ {len(errors)} validation errors:')
        for e in errors:
            print(f'  {e}')
        return False

    print(f'✅ All {len(manifest)} CSVs validated ({sum(m["rows"] for m in manifest.values()):,} total rows)')
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate Syn-Logistics value data')
    parser.add_argument('--validate', action='store_true',
                        help='Validate existing CSVs instead of generating')
    args = parser.parse_args()

    if args.validate:
        ok = validate()
        exit(0 if ok else 1)

    print('Generating Syn-Logistics value data...')
    manifest = generate_all()
    print('\nValidating...')
    validate(manifest)
    print('\n✅ Done!')


if __name__ == '__main__':
    main()
