"""Build TPC-DI dataset from the official TPC-DI v1.1 specification.

Source of truth: TPC-DI Specification v1.1.0 + detobel36/tpc-di DDL
Domain: Retail brokerage firm

Tables: 17 warehouse tables
  - 8 Dimension: DimBroker, DimCustomer, DimAccount, DimCompany, DimDate,
                  DimSecurity, DimTime, DimTrade
  - 4 Fact: FactCashBalances, FactHoldings, FactMarketHistory, FactWatches
  - 1 Financial: Financial
  - 4 Reference: Industry, StatusType, TaxRate, TradeType
  + 2 System: Prospect, DImessages (excluded from benchmark)

Source files (OLTP extracts):
  - CustomerMgmt.xml -> DimCustomer, DimAccount
  - HR.csv -> DimBroker
  - FINWIRE (fixed-width) -> DimCompany, DimSecurity, Financial
  - Trade.txt, TradeHistory.txt -> DimTrade, FactHoldings
  - WatchHistory.txt -> FactWatches
  - CashTransaction.txt -> FactCashBalances
  - DailyMarket.txt -> FactMarketHistory
  - Prospect.csv -> Prospect
  - Date.txt -> DimDate
  - Time.txt -> DimTime
  - StatusType.txt, TaxRate.txt, TradeType.txt, Industry.txt -> reference tables
"""
import csv
import json
from pathlib import Path


def build_tables_csv(out_dir: Path):
    """Create TPC-DI tables CSV in same format as AdventureWorks."""
    tables = [
        # Dimension tables
        ('DimBroker', 'dbo', 'BASE TABLE'),
        ('DimCustomer', 'dbo', 'BASE TABLE'),
        ('DimAccount', 'dbo', 'BASE TABLE'),
        ('DimCompany', 'dbo', 'BASE TABLE'),
        ('DimDate', 'dbo', 'BASE TABLE'),
        ('DimSecurity', 'dbo', 'BASE TABLE'),
        ('DimTime', 'dbo', 'BASE TABLE'),
        ('DimTrade', 'dbo', 'BASE TABLE'),
        # Fact tables
        ('FactCashBalances', 'dbo', 'BASE TABLE'),
        ('FactHoldings', 'dbo', 'BASE TABLE'),
        ('FactMarketHistory', 'dbo', 'BASE TABLE'),
        ('FactWatches', 'dbo', 'BASE TABLE'),
        # Financial
        ('Financial', 'dbo', 'BASE TABLE'),
        # Reference tables
        ('Industry', 'dbo', 'BASE TABLE'),
        ('StatusType', 'dbo', 'BASE TABLE'),
        ('TaxRate', 'dbo', 'BASE TABLE'),
        ('TradeType', 'dbo', 'BASE TABLE'),
    ]

    path = out_dir / 'TPCDI_Tables.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['TABLE_NAME', 'TABLE_SCHEMA', 'TABLE_TYPE'])
        writer.writeheader()
        for name, schema, ttype in tables:
            writer.writerow({'TABLE_NAME': name, 'TABLE_SCHEMA': schema, 'TABLE_TYPE': ttype})
    print(f"  Written {len(tables)} tables to {path}")
    return [t[0] for t in tables]


def build_relationships_csv(out_dir: Path):
    """Create FK relationships CSV from the DDL REFERENCES clauses."""
    # Extracted directly from the DDL CREATE TABLE statements
    fks = [
        # DimAccount references
        ('DimAccount', 'SK_BrokerID', 'DimBroker', 'SK_BrokerID', 'FK_DimAccount_DimBroker'),
        ('DimAccount', 'SK_CustomerID', 'DimCustomer', 'SK_CustomerID', 'FK_DimAccount_DimCustomer'),
        # DimSecurity references
        ('DimSecurity', 'SK_CompanyID', 'DimCompany', 'SK_CompanyID', 'FK_DimSecurity_DimCompany'),
        # DimTrade references
        ('DimTrade', 'SK_BrokerID', 'DimBroker', 'SK_BrokerID', 'FK_DimTrade_DimBroker'),
        ('DimTrade', 'SK_CreateDateID', 'DimDate', 'SK_DateID', 'FK_DimTrade_DimDate_Create'),
        ('DimTrade', 'SK_CreateTimeID', 'DimTime', 'SK_TimeID', 'FK_DimTrade_DimTime_Create'),
        ('DimTrade', 'SK_CloseDateID', 'DimDate', 'SK_DateID', 'FK_DimTrade_DimDate_Close'),
        ('DimTrade', 'SK_CloseTimeID', 'DimTime', 'SK_TimeID', 'FK_DimTrade_DimTime_Close'),
        ('DimTrade', 'SK_SecurityID', 'DimSecurity', 'SK_SecurityID', 'FK_DimTrade_DimSecurity'),
        ('DimTrade', 'SK_CompanyID', 'DimCompany', 'SK_CompanyID', 'FK_DimTrade_DimCompany'),
        ('DimTrade', 'SK_CustomerID', 'DimCustomer', 'SK_CustomerID', 'FK_DimTrade_DimCustomer'),
        ('DimTrade', 'SK_AccountID', 'DimAccount', 'SK_AccountID', 'FK_DimTrade_DimAccount'),
        # FactCashBalances references
        ('FactCashBalances', 'SK_CustomerID', 'DimCustomer', 'SK_CustomerID', 'FK_FactCashBal_DimCustomer'),
        ('FactCashBalances', 'SK_AccountID', 'DimAccount', 'SK_AccountID', 'FK_FactCashBal_DimAccount'),
        ('FactCashBalances', 'SK_DateID', 'DimDate', 'SK_DateID', 'FK_FactCashBal_DimDate'),
        # FactHoldings references
        ('FactHoldings', 'SK_CustomerID', 'DimCustomer', 'SK_CustomerID', 'FK_FactHoldings_DimCustomer'),
        ('FactHoldings', 'SK_AccountID', 'DimAccount', 'SK_AccountID', 'FK_FactHoldings_DimAccount'),
        ('FactHoldings', 'SK_SecurityID', 'DimSecurity', 'SK_SecurityID', 'FK_FactHoldings_DimSecurity'),
        ('FactHoldings', 'SK_CompanyID', 'DimCompany', 'SK_CompanyID', 'FK_FactHoldings_DimCompany'),
        ('FactHoldings', 'SK_DateID', 'DimDate', 'SK_DateID', 'FK_FactHoldings_DimDate'),
        ('FactHoldings', 'SK_TimeID', 'DimTime', 'SK_TimeID', 'FK_FactHoldings_DimTime'),
        # FactMarketHistory references
        ('FactMarketHistory', 'SK_SecurityID', 'DimSecurity', 'SK_SecurityID', 'FK_FactMktHist_DimSecurity'),
        ('FactMarketHistory', 'SK_CompanyID', 'DimCompany', 'SK_CompanyID', 'FK_FactMktHist_DimCompany'),
        ('FactMarketHistory', 'SK_DateID', 'DimDate', 'SK_DateID', 'FK_FactMktHist_DimDate'),
        # FactWatches references
        ('FactWatches', 'SK_CustomerID', 'DimCustomer', 'SK_CustomerID', 'FK_FactWatches_DimCustomer'),
        ('FactWatches', 'SK_SecurityID', 'DimSecurity', 'SK_SecurityID', 'FK_FactWatches_DimSecurity'),
        ('FactWatches', 'SK_DateID_DatePlaced', 'DimDate', 'SK_DateID', 'FK_FactWatches_DimDate_Placed'),
        ('FactWatches', 'SK_DateID_DateRemoved', 'DimDate', 'SK_DateID', 'FK_FactWatches_DimDate_Removed'),
        # Financial references
        ('Financial', 'SK_CompanyID', 'DimCompany', 'SK_CompanyID', 'FK_Financial_DimCompany'),
        # Prospect references
        ('Prospect', 'SK_UpdateDateID', 'DimDate', 'SK_DateID', 'FK_Prospect_DimDate'),
    ]

    path = out_dir / 'TPCDI_Relationships.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'ForeignKeyName', 'ChildTable', 'ChildColumn', 'ParentTable', 'ParentColumn'
        ])
        writer.writeheader()
        for child, child_col, parent, parent_col, fk_name in fks:
            writer.writerow({
                'ForeignKeyName': fk_name,
                'ChildTable': child,
                'ChildColumn': child_col,
                'ParentTable': parent,
                'ParentColumn': parent_col,
            })
    print(f"  Written {len(fks)} FK relationships to {path}")
    return fks


def build_columns_csv(out_dir: Path):
    """Create columns CSV from the official DDL."""
    # Complete columns from TPC-DI v1.1 DDL (detobel36/tpc-di)
    columns = {
        'DimBroker': [
            ('SK_BrokerID', 'INTEGER', 'NO'),
            ('BrokerID', 'INTEGER', 'NO'),
            ('ManagerID', 'INTEGER', 'YES'),
            ('FirstName', 'CHAR(50)', 'NO'),
            ('LastName', 'CHAR(50)', 'NO'),
            ('MiddleInitial', 'CHAR(1)', 'YES'),
            ('Branch', 'CHAR(50)', 'YES'),
            ('Office', 'CHAR(50)', 'YES'),
            ('Phone', 'CHAR(14)', 'YES'),
            ('IsCurrent', 'BIT', 'NO'),
            ('BatchID', 'INTEGER', 'NO'),
            ('EffectiveDate', 'DATE', 'NO'),
            ('EndDate', 'DATE', 'NO'),
        ],
        'DimCustomer': [
            ('SK_CustomerID', 'INTEGER', 'NO'),
            ('CustomerID', 'INTEGER', 'NO'),
            ('TaxID', 'CHAR(20)', 'NO'),
            ('Status', 'CHAR(10)', 'NO'),
            ('LastName', 'CHAR(30)', 'NO'),
            ('FirstName', 'CHAR(30)', 'NO'),
            ('MiddleInitial', 'CHAR(1)', 'YES'),
            ('Gender', 'CHAR(1)', 'YES'),
            ('Tier', 'INTEGER', 'YES'),
            ('DOB', 'DATE', 'NO'),
            ('AddressLine1', 'VARCHAR(80)', 'NO'),
            ('AddressLine2', 'VARCHAR(80)', 'YES'),
            ('PostalCode', 'CHAR(12)', 'NO'),
            ('City', 'CHAR(25)', 'NO'),
            ('StateProv', 'CHAR(20)', 'NO'),
            ('Country', 'CHAR(24)', 'YES'),
            ('Phone1', 'CHAR(30)', 'YES'),
            ('Phone2', 'CHAR(30)', 'YES'),
            ('Phone3', 'CHAR(30)', 'YES'),
            ('Email1', 'CHAR(50)', 'YES'),
            ('Email2', 'CHAR(50)', 'YES'),
            ('NationalTaxRateDesc', 'VARCHAR(50)', 'YES'),
            ('NationalTaxRate', 'NUMERIC(6,5)', 'YES'),
            ('LocalTaxRateDesc', 'VARCHAR(50)', 'YES'),
            ('LocalTaxRate', 'NUMERIC(6,5)', 'YES'),
            ('AgencyID', 'CHAR(30)', 'YES'),
            ('CreditRating', 'INTEGER', 'YES'),
            ('NetWorth', 'NUMERIC(10)', 'YES'),
            ('MarketingNameplate', 'VARCHAR(100)', 'YES'),
            ('IsCurrent', 'BIT', 'NO'),
            ('BatchID', 'INTEGER', 'NO'),
            ('EffectiveDate', 'DATE', 'NO'),
            ('EndDate', 'DATE', 'NO'),
        ],
        'DimAccount': [
            ('SK_AccountID', 'INTEGER', 'NO'),
            ('AccountID', 'INTEGER', 'NO'),
            ('SK_BrokerID', 'INTEGER', 'NO'),
            ('SK_CustomerID', 'INTEGER', 'NO'),
            ('Status', 'CHAR(10)', 'NO'),
            ('AccountDesc', 'VARCHAR(50)', 'YES'),
            ('TaxStatus', 'INTEGER', 'NO'),
            ('IsCurrent', 'BIT', 'NO'),
            ('BatchID', 'INTEGER', 'NO'),
            ('EffectiveDate', 'DATE', 'NO'),
            ('EndDate', 'DATE', 'NO'),
        ],
        'DimCompany': [
            ('SK_CompanyID', 'INTEGER', 'NO'),
            ('CompanyID', 'INTEGER', 'NO'),
            ('Status', 'CHAR(10)', 'NO'),
            ('Name', 'CHAR(60)', 'NO'),
            ('Industry', 'CHAR(50)', 'NO'),
            ('SPrating', 'CHAR(4)', 'YES'),
            ('isLowGrade', 'BIT', 'YES'),
            ('CEO', 'CHAR(100)', 'NO'),
            ('AddressLine1', 'CHAR(80)', 'YES'),
            ('AddressLine2', 'CHAR(80)', 'YES'),
            ('PostalCode', 'CHAR(12)', 'NO'),
            ('City', 'CHAR(25)', 'NO'),
            ('StateProv', 'CHAR(20)', 'NO'),
            ('Country', 'CHAR(24)', 'YES'),
            ('Description', 'CHAR(150)', 'NO'),
            ('FoundingDate', 'DATE', 'YES'),
            ('IsCurrent', 'BIT', 'NO'),
            ('BatchID', 'NUMERIC(5)', 'NO'),
            ('EffectiveDate', 'DATE', 'NO'),
            ('EndDate', 'DATE', 'NO'),
        ],
        'DimDate': [
            ('SK_DateID', 'INTEGER', 'NO'),
            ('DateValue', 'DATE', 'NO'),
            ('DateDesc', 'CHAR(20)', 'NO'),
            ('CalendarYearID', 'NUMERIC(4)', 'NO'),
            ('CalendarYearDesc', 'CHAR(20)', 'NO'),
            ('CalendarQtrID', 'NUMERIC(5)', 'NO'),
            ('CalendarQtrDesc', 'CHAR(20)', 'NO'),
            ('CalendarMonthID', 'NUMERIC(6)', 'NO'),
            ('CalendarMonthDesc', 'CHAR(20)', 'NO'),
            ('CalendarWeekID', 'NUMERIC(6)', 'NO'),
            ('CalendarWeekDesc', 'CHAR(20)', 'NO'),
            ('DayOfWeekNumeric', 'NUMERIC(1)', 'NO'),
            ('DayOfWeekDesc', 'CHAR(10)', 'NO'),
            ('FiscalYearID', 'NUMERIC(4)', 'NO'),
            ('FiscalYearDesc', 'CHAR(20)', 'NO'),
            ('FiscalQtrID', 'NUMERIC(5)', 'NO'),
            ('FiscalQtrDesc', 'CHAR(20)', 'NO'),
            ('HolidayFlag', 'BIT', 'YES'),
        ],
        'DimSecurity': [
            ('SK_SecurityID', 'INTEGER', 'NO'),
            ('Symbol', 'CHAR(15)', 'NO'),
            ('Issue', 'CHAR(6)', 'NO'),
            ('Status', 'CHAR(10)', 'NO'),
            ('Name', 'CHAR(70)', 'NO'),
            ('ExchangeID', 'CHAR(6)', 'NO'),
            ('SK_CompanyID', 'INTEGER', 'NO'),
            ('SharesOutstanding', 'INTEGER', 'NO'),
            ('FirstTrade', 'DATE', 'NO'),
            ('FirstTradeOnExchange', 'DATE', 'NO'),
            ('Dividend', 'INTEGER', 'NO'),
            ('IsCurrent', 'BIT', 'NO'),
            ('BatchID', 'NUMERIC(5)', 'NO'),
            ('EffectiveDate', 'DATE', 'NO'),
            ('EndDate', 'DATE', 'NO'),
        ],
        'DimTime': [
            ('SK_TimeID', 'INTEGER', 'NO'),
            ('TimeValue', 'TIME', 'NO'),
            ('HourID', 'NUMERIC(2)', 'NO'),
            ('HourDesc', 'CHAR(20)', 'NO'),
            ('MinuteID', 'NUMERIC(2)', 'NO'),
            ('MinuteDesc', 'CHAR(20)', 'NO'),
            ('SecondID', 'NUMERIC(2)', 'NO'),
            ('SecondDesc', 'CHAR(20)', 'NO'),
            ('MarketHoursFlag', 'BIT', 'YES'),
            ('OfficeHoursFlag', 'BIT', 'YES'),
        ],
        'DimTrade': [
            ('TradeID', 'INTEGER', 'NO'),
            ('SK_BrokerID', 'INTEGER', 'YES'),
            ('SK_CreateDateID', 'INTEGER', 'NO'),
            ('SK_CreateTimeID', 'INTEGER', 'NO'),
            ('SK_CloseDateID', 'INTEGER', 'YES'),
            ('SK_CloseTimeID', 'INTEGER', 'YES'),
            ('Status', 'CHAR(10)', 'NO'),
            ('DT_Type', 'CHAR(12)', 'NO'),
            ('CashFlag', 'BIT', 'NO'),
            ('SK_SecurityID', 'INTEGER', 'NO'),
            ('SK_CompanyID', 'INTEGER', 'NO'),
            ('Quantity', 'NUMERIC(6,0)', 'NO'),
            ('BidPrice', 'NUMERIC(8,2)', 'NO'),
            ('SK_CustomerID', 'INTEGER', 'NO'),
            ('SK_AccountID', 'INTEGER', 'NO'),
            ('ExecutedBy', 'CHAR(64)', 'NO'),
            ('TradePrice', 'NUMERIC(8,2)', 'YES'),
            ('Fee', 'NUMERIC(10,2)', 'YES'),
            ('Commission', 'NUMERIC(10,2)', 'YES'),
            ('Tax', 'NUMERIC(10,2)', 'YES'),
            ('BatchID', 'NUMERIC(5)', 'NO'),
        ],
        'FactCashBalances': [
            ('SK_CustomerID', 'INTEGER', 'NO'),
            ('SK_AccountID', 'INTEGER', 'NO'),
            ('SK_DateID', 'INTEGER', 'NO'),
            ('Cash', 'NUMERIC(15,2)', 'NO'),
            ('BatchID', 'NUMERIC(5)', 'YES'),
        ],
        'FactHoldings': [
            ('TradeID', 'INTEGER', 'NO'),
            ('CurrentTradeID', 'INTEGER', 'NO'),
            ('SK_CustomerID', 'INTEGER', 'NO'),
            ('SK_AccountID', 'INTEGER', 'NO'),
            ('SK_SecurityID', 'INTEGER', 'NO'),
            ('SK_CompanyID', 'INTEGER', 'NO'),
            ('SK_DateID', 'INTEGER', 'NO'),
            ('SK_TimeID', 'INTEGER', 'NO'),
            ('CurrentPrice', 'INTEGER', 'YES'),
            ('CurrentHolding', 'NUMERIC(6)', 'NO'),
            ('BatchID', 'NUMERIC(5)', 'YES'),
        ],
        'FactMarketHistory': [
            ('SK_SecurityID', 'INTEGER', 'NO'),
            ('SK_CompanyID', 'INTEGER', 'NO'),
            ('SK_DateID', 'INTEGER', 'NO'),
            ('PERatio', 'NUMERIC(10,2)', 'YES'),
            ('Yield', 'NUMERIC(5,2)', 'NO'),
            ('FiftyTwoWeekHigh', 'NUMERIC(8,2)', 'NO'),
            ('SK_FiftyTwoWeekHighDate', 'INTEGER', 'NO'),
            ('FiftyTwoWeekLow', 'NUMERIC(8,2)', 'NO'),
            ('SK_FiftyTwoWeekLowDate', 'INTEGER', 'NO'),
            ('ClosePrice', 'NUMERIC(8,2)', 'NO'),
            ('DayHigh', 'NUMERIC(8,2)', 'NO'),
            ('DayLow', 'NUMERIC(8,2)', 'NO'),
            ('Volume', 'NUMERIC(12)', 'NO'),
            ('BatchID', 'NUMERIC(5)', 'YES'),
        ],
        'FactWatches': [
            ('SK_CustomerID', 'INTEGER', 'NO'),
            ('SK_SecurityID', 'INTEGER', 'NO'),
            ('SK_DateID_DatePlaced', 'INTEGER', 'NO'),
            ('SK_DateID_DateRemoved', 'INTEGER', 'YES'),
            ('BatchID', 'NUMERIC(5)', 'NO'),
        ],
        'Financial': [
            ('SK_CompanyID', 'INTEGER', 'NO'),
            ('FI_YEAR', 'NUMERIC(4)', 'NO'),
            ('FI_QTR', 'NUMERIC(1)', 'NO'),
            ('FI_QTR_START_DATE', 'DATE', 'NO'),
            ('FI_REVENUE', 'NUMERIC(15,2)', 'NO'),
            ('FI_NET_EARN', 'NUMERIC(15,2)', 'NO'),
            ('FI_BASIC_EPS', 'NUMERIC(10,2)', 'NO'),
            ('FI_DILUT_EPS', 'NUMERIC(10,2)', 'NO'),
            ('FI_MARGIN', 'NUMERIC(10,2)', 'NO'),
            ('FI_INVENTORY', 'NUMERIC(15,2)', 'NO'),
            ('FI_ASSETS', 'NUMERIC(15,2)', 'NO'),
            ('FI_LIABILITY', 'NUMERIC(15,2)', 'NO'),
            ('FI_OUT_BASIC', 'NUMERIC(12)', 'NO'),
            ('FI_OUT_DILUT', 'NUMERIC(12)', 'NO'),
        ],
        'Industry': [
            ('IN_ID', 'CHAR(2)', 'NO'),
            ('IN_NAME', 'CHAR(50)', 'NO'),
            ('IN_SC_ID', 'CHAR(4)', 'NO'),
        ],
        'StatusType': [
            ('ST_ID', 'CHAR(4)', 'NO'),
            ('ST_NAME', 'CHAR(10)', 'NO'),
        ],
        'TaxRate': [
            ('TX_ID', 'CHAR(4)', 'NO'),
            ('TX_NAME', 'CHAR(50)', 'NO'),
            ('TX_RATE', 'NUMERIC(6,5)', 'NO'),
        ],
        'TradeType': [
            ('TT_ID', 'CHAR(3)', 'NO'),
            ('TT_NAME', 'CHAR(12)', 'NO'),
            ('TT_IS_SELL', 'NUMERIC(1)', 'NO'),
            ('TT_IS_MRKT', 'NUMERIC(1)', 'NO'),
        ],
    }

    path = out_dir / 'TPCDI_Columns.csv'
    total = 0
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'TABLE_SCHEMA', 'TABLE_NAME', 'COLUMN_NAME', 'DATA_TYPE',
            'CHARACTER_MAXIMUM_LENGTH', 'IS_NULLABLE'
        ])
        writer.writeheader()
        for table_name, cols in columns.items():
            for col_name, data_type, nullable in cols:
                writer.writerow({
                    'TABLE_SCHEMA': 'dbo',
                    'TABLE_NAME': table_name,
                    'COLUMN_NAME': col_name,
                    'DATA_TYPE': data_type,
                    'CHARACTER_MAXIMUM_LENGTH': '',
                    'IS_NULLABLE': nullable,
                })
                total += 1
    print(f"  Written {total} columns across {len(columns)} tables to {path}")
    return columns


def build_lineage_edges(out_dir: Path):
    """Build ETL lineage edges from the TPC-DI specification.

    TPC-DI has explicit source → destination mappings documented in the spec.
    Source files represent OLTP extracts; destination = warehouse tables.
    """
    # From TPC-DI v1.1 specification Section 4 (Transformations)
    lineage = [
        # CustomerMgmt.xml → DimCustomer (customer data with denormalized address/phone/tax)
        {'source': 'SRC.CustomerMgmt', 'target': 'DimCustomer',
         'evidence': 'TPC-DI Spec: CustomerMgmt.xml contains customer demographics, addresses, phones'},
        # CustomerMgmt.xml → DimAccount (account open/update/close actions)
        {'source': 'SRC.CustomerMgmt', 'target': 'DimAccount',
         'evidence': 'TPC-DI Spec: CustomerMgmt.xml contains account creation and updates'},
        # HR.csv → DimBroker
        {'source': 'SRC.HR', 'target': 'DimBroker',
         'evidence': 'TPC-DI Spec: HR.csv is the sole source for broker data'},
        # FINWIRE (CMP records) → DimCompany
        {'source': 'SRC.FINWIRE_CMP', 'target': 'DimCompany',
         'evidence': 'TPC-DI Spec: FINWIRE CMP records contain company information'},
        # FINWIRE (SEC records) → DimSecurity
        {'source': 'SRC.FINWIRE_SEC', 'target': 'DimSecurity',
         'evidence': 'TPC-DI Spec: FINWIRE SEC records contain security/stock information'},
        # FINWIRE (FIN records) → Financial
        {'source': 'SRC.FINWIRE_FIN', 'target': 'Financial',
         'evidence': 'TPC-DI Spec: FINWIRE FIN records contain quarterly financial data'},
        # Date.txt → DimDate
        {'source': 'SRC.Date', 'target': 'DimDate',
         'evidence': 'TPC-DI Spec: Date.txt provides pre-generated date dimension data'},
        # Time.txt → DimTime
        {'source': 'SRC.Time', 'target': 'DimTime',
         'evidence': 'TPC-DI Spec: Time.txt provides pre-generated time dimension data'},
        # Trade.txt + TradeHistory.txt → DimTrade
        {'source': 'SRC.Trade', 'target': 'DimTrade',
         'evidence': 'TPC-DI Spec: Trade.txt and TradeHistory.txt contain trade lifecycle events'},
        {'source': 'SRC.TradeHistory', 'target': 'DimTrade',
         'evidence': 'TPC-DI Spec: TradeHistory.txt contains historical trade status changes'},
        # Trade.txt → FactHoldings (via trade execution)
        {'source': 'SRC.Trade', 'target': 'FactHoldings',
         'evidence': 'TPC-DI Spec: Trade executions create/update holdings positions'},
        # HoldingHistory.txt → FactHoldings
        {'source': 'SRC.HoldingHistory', 'target': 'FactHoldings',
         'evidence': 'TPC-DI Spec: HoldingHistory.txt tracks position changes'},
        # CashTransaction.txt → FactCashBalances
        {'source': 'SRC.CashTransaction', 'target': 'FactCashBalances',
         'evidence': 'TPC-DI Spec: CashTransaction.txt contains cash deposits/withdrawals'},
        # DailyMarket.txt → FactMarketHistory
        {'source': 'SRC.DailyMarket', 'target': 'FactMarketHistory',
         'evidence': 'TPC-DI Spec: DailyMarket.txt contains daily stock prices and volumes'},
        # WatchHistory.txt → FactWatches
        {'source': 'SRC.WatchHistory', 'target': 'FactWatches',
         'evidence': 'TPC-DI Spec: WatchHistory.txt tracks customer watch list events'},
        # Prospect.csv → Prospect table
        {'source': 'SRC.Prospect', 'target': 'Prospect',
         'evidence': 'TPC-DI Spec: Prospect.csv contains prospective customer demographics'},
        # StatusType.txt → StatusType (reference)
        {'source': 'SRC.StatusType', 'target': 'StatusType',
         'evidence': 'TPC-DI Spec: StatusType.txt provides status code reference data'},
        # TaxRate.txt → TaxRate (reference)
        {'source': 'SRC.TaxRate', 'target': 'TaxRate',
         'evidence': 'TPC-DI Spec: TaxRate.txt provides tax rate reference data'},
        # TradeType.txt → TradeType (reference)
        {'source': 'SRC.TradeType', 'target': 'TradeType',
         'evidence': 'TPC-DI Spec: TradeType.txt provides trade type reference data'},
        # Industry.txt → Industry (reference)
        {'source': 'SRC.Industry', 'target': 'Industry',
         'evidence': 'TPC-DI Spec: Industry.txt provides industry classification reference data'},
        # Cross-table derivation: DimCustomer → Prospect.IsCustomer flag
        {'source': 'DimCustomer', 'target': 'Prospect',
         'evidence': 'TPC-DI Spec: Prospect.IsCustomer is derived by matching against DimCustomer'},
        # Cross-table: DimCompany → DimSecurity (company lookup for security)
        {'source': 'DimCompany', 'target': 'DimSecurity',
         'evidence': 'TPC-DI Spec: DimSecurity.SK_CompanyID is looked up from DimCompany'},
        # Cross-table: TaxRate → DimCustomer (tax rate lookup)
        {'source': 'TaxRate', 'target': 'DimCustomer',
         'evidence': 'TPC-DI Spec: DimCustomer.NationalTaxRate/LocalTaxRate looked up from TaxRate'},
    ]

    path = out_dir / 'lineage_edges.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(lineage, f, indent=2, ensure_ascii=False)
    print(f"  Written {len(lineage)} lineage edges to {path}")
    return lineage


def main():
    out_dir = Path(__file__).parent.parent / 'datasets' / 'tpc-di'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building TPC-DI dataset from official specification...")
    print()

    print("1. Tables:")
    tables = build_tables_csv(out_dir)

    print("\n2. FK Relationships:")
    fks = build_relationships_csv(out_dir)

    print("\n3. Columns:")
    columns = build_columns_csv(out_dir)

    print("\n4. ETL Lineage:")
    lineage = build_lineage_edges(out_dir)

    # Summary
    print(f"\n{'=' * 60}")
    print("TPC-DI Dataset Summary")
    print(f"{'=' * 60}")
    print(f"  Tables:     {len(tables)}")
    print(f"  FK edges:   {len(fks)}")
    print(f"  Columns:    {sum(len(c) for c in columns.values())}")
    print(f"  Lineage:    {len(lineage)} ETL edges")
    print(f"  Domain:     Retail brokerage firm")
    print(f"  Source:     TPC-DI v1.1.0 specification")


if __name__ == '__main__':
    main()
