"""Generate AdventureWorks Tier-2 value-level data.

Creates synthetic row-level CSV data for AdventureWorks across 3 layers:
  OLTP source tables  → Staging views  → DW Dim/Fact tables

Also builds a row-level lineage_map.json linking DW rows → Staging rows → OLTP rows.

All data is deterministic (seeded RNG) for reproducibility.

Usage:
    python scripts/generate_aw_value_data.py
    python scripts/generate_aw_value_data.py --validate
"""
import argparse
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

REPO_ROOT = Path(__file__).parent.parent
DS_DIR = REPO_ROOT / 'datasets' / 'adventureworks'
OUT_DIR = DS_DIR / 'value_data'


# ──────────────────────────────────────────────────────────────────
# Schema definition: 3 layers
# ──────────────────────────────────────────────────────────────────

# OLTP source tables (key tables that feed into DW, with their schemas)
OLTP_TABLES = {
    # Person schema
    'Person.Person': {
        'cols': ['BusinessEntityID', 'PersonType', 'FirstName', 'LastName',
                 'EmailPromotion', 'ModifiedDate'],
        'rows': 2000,
    },
    'Person.EmailAddress': {
        'cols': ['BusinessEntityID', 'EmailAddressID', 'EmailAddress',
                 'ModifiedDate'],
        'rows': 2000,
    },
    'Person.Address': {
        'cols': ['AddressID', 'AddressLine1', 'City', 'StateProvinceID',
                 'PostalCode', 'ModifiedDate'],
        'rows': 1500,
    },
    'Person.BusinessEntityAddress': {
        'cols': ['BusinessEntityID', 'AddressID', 'AddressTypeID',
                 'ModifiedDate'],
        'rows': 1500,
    },
    'Person.StateProvince': {
        'cols': ['StateProvinceID', 'StateProvinceCode', 'CountryRegionCode',
                 'Name', 'TerritoryID'],
        'rows': 181,
    },
    'Person.CountryRegion': {
        'cols': ['CountryRegionCode', 'Name', 'ModifiedDate'],
        'rows': 238,
    },
    # Sales schema
    'Sales.Customer': {
        'cols': ['CustomerID', 'PersonID', 'StoreID', 'TerritoryID',
                 'AccountNumber', 'ModifiedDate'],
        'rows': 1500,
    },
    'Sales.Store': {
        'cols': ['BusinessEntityID', 'Name', 'SalesPersonID',
                 'Demographics', 'ModifiedDate'],
        'rows': 700,
    },
    'Sales.SalesOrderHeader': {
        'cols': ['SalesOrderID', 'OrderDate', 'DueDate', 'ShipDate',
                 'Status', 'CustomerID', 'SalesPersonID', 'TerritoryID',
                 'SubTotal', 'TaxAmt', 'Freight', 'TotalDue'],
        'rows': 3000,
    },
    'Sales.SalesOrderDetail': {
        'cols': ['SalesOrderID', 'SalesOrderDetailID', 'ProductID',
                 'OrderQty', 'UnitPrice', 'LineTotal'],
        'rows': 6000,
    },
    'Sales.SalesTerritory': {
        'cols': ['TerritoryID', 'Name', 'CountryRegionCode', 'Group',
                 'SalesYTD', 'SalesLastYear'],
        'rows': 10,
    },
    'Sales.SalesPerson': {
        'cols': ['BusinessEntityID', 'TerritoryID', 'SalesQuota',
                 'Bonus', 'CommissionPct', 'SalesYTD'],
        'rows': 17,
    },
    'Sales.SalesPersonQuotaHistory': {
        'cols': ['BusinessEntityID', 'QuotaDate', 'SalesQuota',
                 'ModifiedDate'],
        'rows': 200,
    },
    'Sales.Currency': {
        'cols': ['CurrencyCode', 'Name', 'ModifiedDate'],
        'rows': 105,
    },
    'Sales.CurrencyRate': {
        'cols': ['CurrencyRateID', 'CurrencyRateDate', 'FromCurrencyCode',
                 'ToCurrencyCode', 'AverageRate', 'EndOfDayRate'],
        'rows': 500,
    },
    'Sales.SpecialOffer': {
        'cols': ['SpecialOfferID', 'Description', 'DiscountPct',
                 'Type', 'Category', 'StartDate', 'EndDate'],
        'rows': 16,
    },
    'Sales.SalesReason': {
        'cols': ['SalesReasonID', 'Name', 'ReasonType', 'ModifiedDate'],
        'rows': 10,
    },
    'Sales.SalesOrderHeaderSalesReason': {
        'cols': ['SalesOrderID', 'SalesReasonID', 'ModifiedDate'],
        'rows': 4000,
    },
    # Production schema
    'Production.Product': {
        'cols': ['ProductID', 'Name', 'ProductNumber', 'Color',
                 'StandardCost', 'ListPrice', 'ProductSubcategoryID',
                 'ProductModelID'],
        'rows': 504,
    },
    'Production.ProductSubcategory': {
        'cols': ['ProductSubcategoryID', 'ProductCategoryID', 'Name',
                 'ModifiedDate'],
        'rows': 37,
    },
    'Production.ProductCategory': {
        'cols': ['ProductCategoryID', 'Name', 'ModifiedDate'],
        'rows': 4,
    },
    'Production.ProductModel': {
        'cols': ['ProductModelID', 'Name', 'CatalogDescription',
                 'Instructions', 'ModifiedDate'],
        'rows': 128,
    },
    # Purchasing schema
    'Purchasing.PurchaseOrderHeader': {
        'cols': ['PurchaseOrderID', 'EmployeeID', 'VendorID',
                 'ShipMethodID', 'OrderDate', 'SubTotal', 'TaxAmt',
                 'Freight', 'TotalDue'],
        'rows': 1000,
    },
    'Purchasing.PurchaseOrderDetail': {
        'cols': ['PurchaseOrderID', 'PurchaseOrderDetailID', 'ProductID',
                 'OrderQty', 'UnitPrice', 'LineTotal', 'ReceivedQty',
                 'StockedQty'],
        'rows': 3000,
    },
    'Purchasing.Vendor': {
        'cols': ['BusinessEntityID', 'AccountNumber', 'Name',
                 'CreditRating', 'ActiveFlag'],
        'rows': 104,
    },
    # HumanResources schema
    'HumanResources.Employee': {
        'cols': ['BusinessEntityID', 'NationalIDNumber', 'LoginID',
                 'JobTitle', 'BirthDate', 'Gender', 'HireDate',
                 'SalariedFlag', 'VacationHours'],
        'rows': 290,
    },
    'HumanResources.Department': {
        'cols': ['DepartmentID', 'Name', 'GroupName', 'ModifiedDate'],
        'rows': 16,
    },
}

# Staging views (intermediate layer — extracted from lineage evidence)
STAGING_VIEWS = {
    'stg_FactResellerSales': {
        'oltp_sources': ['Sales.SalesOrderHeader', 'Sales.SalesOrderDetail',
                         'Sales.Customer', 'Production.Product'],
        'cols': ['staging_id', 'SalesOrderID', 'SalesOrderDetailID',
                 'CustomerID', 'ProductID', 'OrderDate', 'UnitPrice',
                 'OrderQty', 'LineTotal', 'TerritoryID', 'etl_timestamp'],
        'rows': 3000,
    },
    'stg_FactOnlineSales': {
        'oltp_sources': ['Sales.SalesOrderHeader', 'Sales.SalesOrderDetail',
                         'Sales.Customer', 'Production.Product'],
        'cols': ['staging_id', 'SalesOrderID', 'SalesOrderDetailID',
                 'CustomerID', 'ProductID', 'OrderDate', 'UnitPrice',
                 'OrderQty', 'LineTotal', 'TerritoryID', 'etl_timestamp'],
        'rows': 3000,
    },
    'stg_FactPurchaseOrder': {
        'oltp_sources': ['Purchasing.PurchaseOrderDetail',
                         'Purchasing.PurchaseOrderHeader', 'Purchasing.Vendor'],
        'cols': ['staging_id', 'PurchaseOrderID', 'ProductID', 'VendorID',
                 'OrderQty', 'UnitPrice', 'LineTotal', 'OrderDate',
                 'etl_timestamp'],
        'rows': 2000,
    },
    'stg_DimReseller': {
        'oltp_sources': ['Sales.Customer', 'Sales.Store'],
        'cols': ['staging_id', 'CustomerID', 'StoreID', 'StoreName',
                 'SalesPersonID', 'etl_timestamp'],
        'rows': 700,
    },
    'stg_DimCustomer': {
        'oltp_sources': ['Person.Person', 'Sales.Customer',
                         'Person.EmailAddress', 'Person.BusinessEntityAddress',
                         'Person.Address'],
        'cols': ['staging_id', 'CustomerID', 'PersonID', 'FirstName',
                 'LastName', 'EmailAddress', 'AddressLine1', 'City',
                 'etl_timestamp'],
        'rows': 1500,
    },
    'stg_DimProduct': {
        'oltp_sources': ['Production.Product', 'Production.ProductSubcategory',
                         'Production.ProductCategory', 'Production.ProductModel'],
        'cols': ['staging_id', 'ProductID', 'ProductName', 'SubcategoryName',
                 'CategoryName', 'ModelName', 'ListPrice', 'etl_timestamp'],
        'rows': 504,
    },
    'stg_DimEmployee': {
        'oltp_sources': ['HumanResources.Employee', 'Person.Person',
                         'Sales.SalesPerson'],
        'cols': ['staging_id', 'EmployeeID', 'FirstName', 'LastName',
                 'JobTitle', 'HireDate', 'SalesQuota', 'etl_timestamp'],
        'rows': 290,
    },
    'stg_DimGeography': {
        'oltp_sources': ['Person.Address', 'Person.StateProvince',
                         'Person.CountryRegion'],
        'cols': ['staging_id', 'AddressID', 'City', 'StateProvinceName',
                 'CountryRegionName', 'PostalCode', 'etl_timestamp'],
        'rows': 500,
    },
    'stg_DimSalesTerritory': {
        'oltp_sources': ['Sales.SalesTerritory'],
        'cols': ['staging_id', 'TerritoryID', 'TerritoryName',
                 'CountryRegionCode', 'Group', 'etl_timestamp'],
        'rows': 10,
    },
    'stg_DimCurrency': {
        'oltp_sources': ['Sales.Currency'],
        'cols': ['staging_id', 'CurrencyCode', 'CurrencyName',
                 'etl_timestamp'],
        'rows': 105,
    },
    'stg_DimPromotion': {
        'oltp_sources': ['Sales.SpecialOffer'],
        'cols': ['staging_id', 'SpecialOfferID', 'Description',
                 'DiscountPct', 'Type', 'Category', 'etl_timestamp'],
        'rows': 16,
    },
    'stg_DimSalesReason': {
        'oltp_sources': ['Sales.SalesReason'],
        'cols': ['staging_id', 'SalesReasonID', 'Name', 'ReasonType',
                 'etl_timestamp'],
        'rows': 10,
    },
    'stg_FactCurrencyRate': {
        'oltp_sources': ['Sales.CurrencyRate'],
        'cols': ['staging_id', 'CurrencyRateID', 'CurrencyRateDate',
                 'FromCurrencyCode', 'ToCurrencyCode', 'AverageRate',
                 'etl_timestamp'],
        'rows': 500,
    },
    'stg_FactSalesQuota': {
        'oltp_sources': ['Sales.SalesPersonQuotaHistory'],
        'cols': ['staging_id', 'BusinessEntityID', 'QuotaDate',
                 'SalesQuota', 'etl_timestamp'],
        'rows': 200,
    },
    'stg_DimDepartmentGroup': {
        'oltp_sources': ['HumanResources.Department'],
        'cols': ['staging_id', 'DepartmentID', 'DepartmentName',
                 'GroupName', 'etl_timestamp'],
        'rows': 16,
    },
    'stg_DimProductCategory': {
        'oltp_sources': ['Production.ProductCategory'],
        'cols': ['staging_id', 'ProductCategoryID', 'CategoryName',
                 'etl_timestamp'],
        'rows': 4,
    },
    'stg_DimProductSubcategory': {
        'oltp_sources': ['Production.ProductSubcategory'],
        'cols': ['staging_id', 'ProductSubcategoryID', 'SubcategoryName',
                 'ProductCategoryID', 'etl_timestamp'],
        'rows': 37,
    },
    'stg_FactInternetSalesReason': {
        'oltp_sources': ['Sales.SalesOrderHeaderSalesReason',
                         'Sales.SalesReason'],
        'cols': ['staging_id', 'SalesOrderID', 'SalesReasonID',
                 'ReasonName', 'etl_timestamp'],
        'rows': 2000,
    },
}

# DW Dim/Fact tables and their staging sources
DW_TABLES = {
    'DimCustomer': {
        'staging_source': 'stg_DimCustomer',
        'cols': ['CustomerKey', 'CustomerAlternateKey', 'FirstName',
                 'LastName', 'EmailAddress', 'AddressLine1', 'City',
                 'GeographyKey'],
        'rows': 1000,
    },
    'DimProduct': {
        'staging_source': 'stg_DimProduct',
        'cols': ['ProductKey', 'ProductAlternateKey', 'EnglishProductName',
                 'ProductSubcategoryKey', 'ListPrice', 'Color'],
        'rows': 504,
    },
    'DimEmployee': {
        'staging_source': 'stg_DimEmployee',
        'cols': ['EmployeeKey', 'EmployeeNationalIDAlternateKey',
                 'FirstName', 'LastName', 'Title', 'HireDate',
                 'SalesPersonFlag'],
        'rows': 290,
    },
    'DimGeography': {
        'staging_source': 'stg_DimGeography',
        'cols': ['GeographyKey', 'City', 'StateProvinceName',
                 'EnglishCountryRegionName', 'PostalCode',
                 'SalesTerritoryKey'],
        'rows': 500,
    },
    'DimReseller': {
        'staging_source': 'stg_DimReseller',
        'cols': ['ResellerKey', 'ResellerAlternateKey', 'ResellerName',
                 'GeographyKey', 'BusinessType'],
        'rows': 700,
    },
    'DimSalesTerritory': {
        'staging_source': 'stg_DimSalesTerritory',
        'cols': ['SalesTerritoryKey', 'SalesTerritoryRegion',
                 'SalesTerritoryCountry', 'SalesTerritoryGroup'],
        'rows': 10,
    },
    'DimCurrency': {
        'staging_source': 'stg_DimCurrency',
        'cols': ['CurrencyKey', 'CurrencyAlternateKey', 'CurrencyName'],
        'rows': 105,
    },
    'DimPromotion': {
        'staging_source': 'stg_DimPromotion',
        'cols': ['PromotionKey', 'PromotionAlternateKey',
                 'EnglishPromotionName', 'DiscountPct',
                 'EnglishPromotionType', 'EnglishPromotionCategory'],
        'rows': 16,
    },
    'DimSalesReason': {
        'staging_source': 'stg_DimSalesReason',
        'cols': ['SalesReasonKey', 'SalesReasonAlternateKey',
                 'SalesReasonName', 'SalesReasonReasonType'],
        'rows': 10,
    },
    'DimDepartmentGroup': {
        'staging_source': 'stg_DimDepartmentGroup',
        'cols': ['DepartmentGroupKey', 'ParentDepartmentGroupKey',
                 'DepartmentGroupName'],
        'rows': 16,
    },
    'DimProductCategory': {
        'staging_source': 'stg_DimProductCategory',
        'cols': ['ProductCategoryKey', 'ProductCategoryAlternateKey',
                 'EnglishProductCategoryName'],
        'rows': 4,
    },
    'DimProductSubcategory': {
        'staging_source': 'stg_DimProductSubcategory',
        'cols': ['ProductSubcategoryKey', 'ProductSubcategoryAlternateKey',
                 'EnglishProductSubcategoryName', 'ProductCategoryKey'],
        'rows': 37,
    },
    'FactResellerSales': {
        'staging_source': 'stg_FactResellerSales',
        'cols': ['FactResellerSalesKey', 'ProductKey', 'ResellerKey',
                 'EmployeeKey', 'PromotionKey', 'CurrencyKey',
                 'SalesTerritoryKey', 'OrderDateKey', 'OrderQuantity',
                 'UnitPrice', 'SalesAmount'],
        'rows': 3000,
    },
    'FactInternetSales': {
        'staging_source': 'stg_FactOnlineSales',
        'cols': ['FactInternetSalesKey', 'ProductKey', 'CustomerKey',
                 'PromotionKey', 'CurrencyKey', 'SalesTerritoryKey',
                 'OrderDateKey', 'OrderQuantity', 'UnitPrice',
                 'SalesAmount'],
        'rows': 3000,
    },
    'FactProductInventory': {
        'staging_source': 'stg_FactPurchaseOrder',
        'cols': ['ProductKey', 'DateKey', 'UnitCost', 'UnitsIn',
                 'UnitsOut', 'UnitsBalance'],
        'rows': 2000,
    },
    'FactCurrencyRate': {
        'staging_source': 'stg_FactCurrencyRate',
        'cols': ['CurrencyKey', 'DateKey', 'AverageRate', 'EndOfDayRate'],
        'rows': 500,
    },
    'FactSalesQuota': {
        'staging_source': 'stg_FactSalesQuota',
        'cols': ['EmployeeKey', 'DateKey', 'CalendarYear',
                 'CalendarQuarter', 'SalesAmountQuota'],
        'rows': 200,
    },
    'FactInternetSalesReason': {
        'staging_source': 'stg_FactInternetSalesReason',
        'cols': ['SalesOrderNumber', 'SalesReasonKey'],
        'rows': 2000,
    },
}


# ──────────────────────────────────────────────────────────────────
# Vocabulary pools
# ──────────────────────────────────────────────────────────────────
FIRST_NAMES = [
    'James', 'Mary', 'Robert', 'Patricia', 'John', 'Jennifer', 'Michael',
    'Linda', 'David', 'Elizabeth', 'William', 'Barbara', 'Richard', 'Susan',
    'Joseph', 'Jessica', 'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher',
    'Lisa', 'Daniel', 'Nancy', 'Matthew', 'Betty', 'Anthony', 'Margaret',
    'Mark', 'Sandra', 'Donald', 'Ashley', 'Steven', 'Kimberly', 'Paul',
    'Emily', 'Andrew', 'Donna', 'Joshua', 'Michelle',
]
LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
    'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
    'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
    'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark',
    'Ramirez', 'Lewis', 'Robinson',
]
CITIES = [
    'Seattle', 'Los Angeles', 'New York', 'Chicago', 'Houston', 'Phoenix',
    'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'London', 'Paris',
    'Berlin', 'Tokyo', 'Sydney', 'Toronto', 'Melbourne', 'Munich', 'Bordeaux',
]
STATES = ['WA', 'CA', 'NY', 'IL', 'TX', 'AZ', 'PA', 'FL', 'OH', 'GA']
COUNTRIES = ['US', 'CA', 'GB', 'DE', 'FR', 'AU', 'JP']
COLORS = ['Black', 'Red', 'Silver', 'White', 'Blue', 'Yellow', 'Multi', None]
PRODUCT_NAMES = [
    'Mountain-100', 'Mountain-200', 'Road-150', 'Road-250', 'Road-350',
    'Touring-1000', 'Touring-2000', 'Sport-100', 'HL Mountain Frame',
    'LL Mountain Frame', 'ML Mountain Frame', 'HL Road Frame',
    'LL Road Frame', 'ML Road Frame', 'Front Wheel', 'Rear Wheel',
    'Chain', 'Pedal', 'Headset', 'Seat', 'Fork', 'Water Bottle',
    'Patch Kit', 'Helmet', 'Jersey', 'Shorts', 'Gloves', 'Socks',
    'Tire Tube', 'Fender', 'Bike Wash', 'Cable Lock', 'Pump',
]
CATEGORIES = ['Bikes', 'Components', 'Clothing', 'Accessories']
SUBCATEGORIES = [
    'Mountain Bikes', 'Road Bikes', 'Touring Bikes', 'Handlebars',
    'Bottom Brackets', 'Brakes', 'Chains', 'Cranksets', 'Derailleurs',
    'Forks', 'Headsets', 'Mountain Frames', 'Pedals', 'Road Frames',
    'Saddles', 'Touring Frames', 'Wheels', 'Jerseys', 'Shorts',
    'Caps', 'Gloves', 'Vests', 'Socks', 'Cleaners', 'Fenders',
    'Helmets', 'Hydration Packs', 'Lights', 'Locks', 'Panniers',
    'Pumps', 'Tires and Tubes', 'Bottles and Cages',
]
STORE_NAMES = [
    'A Bike Store', 'Progressive Sports', 'Advanced Bicycles',
    'Modular Cycle Systems', 'Metropolitan Sports Supply',
    'Aerobic Exercise Company', 'Associated Bikes', 'Bike World',
    'Central Bicycle Supply', 'Channel Outlet', 'Classic Cycle',
    'Comfort Road Bicycles', 'Country Parts Shop', 'Cross-Country Riding',
    'Cycles Mart', 'Discount Cycles', 'Excellent Riding Supplies',
    'Fun Factory Outlet', 'Good Bikes And Gear', 'Great Bikes',
]
JOB_TITLES = [
    'Chief Executive Officer', 'Vice President of Engineering',
    'Engineering Manager', 'Senior Tool Designer', 'Design Engineer',
    'Marketing Manager', 'Sales Representative', 'Production Technician',
    'Research and Development Manager', 'Quality Assurance Technician',
]
DEPARTMENTS = [
    'Engineering', 'Tool Design', 'Marketing', 'Sales', 'Production',
    'Purchasing', 'Research and Development', 'Quality Assurance',
    'Human Resources', 'Finance', 'Information Services', 'Executive',
    'Shipping and Receiving', 'Document Control', 'Facilities and Maintenance',
    'Manufacturing',
]
CURRENCIES = ['USD', 'CAD', 'EUR', 'GBP', 'AUD', 'JPY', 'CHF', 'BRL']
PROMOTIONS = [
    'No Discount', 'Volume Discount 11-14', 'Volume Discount 15-24',
    'Volume Discount 25+', 'Mountain-100 Clearance', 'Sport Helmet Promo',
    'Road-250 Clearance', 'Touring-2000 Promo', 'Half-Price Pedal',
    'Seasonal Discount', 'New Product', 'Reseller Discount',
]
SALES_REASONS = [
    'Price', 'On Promotion', 'Magazine Advertisement', 'Television Ad',
    'Manufacturer', 'Review', 'Quality', 'Recommendation',
    'Demo Event', 'Sponsorship',
]
TERRITORY_NAMES = [
    'Northwest', 'Northeast', 'Central', 'Southwest', 'Southeast',
    'Canada', 'France', 'Germany', 'Australia', 'United Kingdom',
]
TERRITORY_GROUPS = ['North America', 'Europe', 'Pacific']
VENDOR_NAMES = [
    'International', 'American Bicycles', 'National Bike Association',
    'Australia Bike Retailer', 'Trikes Inc.', 'Morgan Bike Accessories',
    'Cycling Master', 'Chicago Rent-All', 'Greenwood Athletic Company',
    'Compete Enterprises Inc.',
]

BASE_DATE = datetime(2022, 1, 1)


def rand_date(rng, days=730):
    return (BASE_DATE + timedelta(days=rng.randint(0, days))).strftime('%Y-%m-%d')


def rand_ts(rng, days=730):
    dt = BASE_DATE + timedelta(
        days=rng.randint(0, days), hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))
    return dt.strftime('%Y-%m-%d %H:%M:%S')


# ──────────────────────────────────────────────────────────────────
# OLTP row generators
# ──────────────────────────────────────────────────────────────────
def gen_oltp(table_name, n, rng):
    """Generate OLTP rows with domain-appropriate data."""
    if table_name == 'Person.Person':
        h = ['BusinessEntityID', 'PersonType', 'FirstName', 'LastName',
             'EmailPromotion', 'ModifiedDate']
        rows = [[i, rng.choice(['SC', 'IN', 'SP', 'EM', 'VC', 'GC']),
                 rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES),
                 rng.randint(0, 2), rand_date(rng)] for i in range(n)]
    elif table_name == 'Person.EmailAddress':
        h = ['BusinessEntityID', 'EmailAddressID', 'EmailAddress',
             'ModifiedDate']
        rows = [[rng.randint(0, 1999), i,
                 f'{rng.choice(FIRST_NAMES).lower()}.{rng.choice(LAST_NAMES).lower()}@adventure-works.com',
                 rand_date(rng)] for i in range(n)]
    elif table_name == 'Person.Address':
        h = ['AddressID', 'AddressLine1', 'City', 'StateProvinceID',
             'PostalCode', 'ModifiedDate']
        rows = [[i, f'{rng.randint(100, 9999)} {rng.choice(["Main", "Oak", "Pine", "Elm", "Maple"])} St',
                 rng.choice(CITIES), rng.randint(0, 180),
                 f'{rng.randint(10000, 99999)}', rand_date(rng)] for i in range(n)]
    elif table_name == 'Person.BusinessEntityAddress':
        h = ['BusinessEntityID', 'AddressID', 'AddressTypeID', 'ModifiedDate']
        rows = [[rng.randint(0, 1999), rng.randint(0, 1499),
                 rng.randint(1, 3), rand_date(rng)] for i in range(n)]
    elif table_name == 'Person.StateProvince':
        h = ['StateProvinceID', 'StateProvinceCode', 'CountryRegionCode',
             'Name', 'TerritoryID']
        rows = [[i, STATES[i % len(STATES)], COUNTRIES[i % len(COUNTRIES)],
                 f'State_{i}', rng.randint(0, 9)] for i in range(n)]
    elif table_name == 'Person.CountryRegion':
        h = ['CountryRegionCode', 'Name', 'ModifiedDate']
        rows = [[COUNTRIES[i % len(COUNTRIES)], f'Country_{i}',
                 rand_date(rng)] for i in range(n)]
    elif table_name == 'Sales.Customer':
        h = ['CustomerID', 'PersonID', 'StoreID', 'TerritoryID',
             'AccountNumber', 'ModifiedDate']
        rows = [[i, rng.randint(0, 1999) if rng.random() > 0.3 else None,
                 rng.randint(0, 699) if rng.random() > 0.5 else None,
                 rng.randint(0, 9), f'AW{i:08d}', rand_date(rng)]
                for i in range(n)]
    elif table_name == 'Sales.Store':
        h = ['BusinessEntityID', 'Name', 'SalesPersonID',
             'Demographics', 'ModifiedDate']
        rows = [[i, STORE_NAMES[i % len(STORE_NAMES)],
                 rng.randint(0, 16), 'XML_DATA', rand_date(rng)]
                for i in range(n)]
    elif table_name == 'Sales.SalesOrderHeader':
        h = ['SalesOrderID', 'OrderDate', 'DueDate', 'ShipDate',
             'Status', 'CustomerID', 'SalesPersonID', 'TerritoryID',
             'SubTotal', 'TaxAmt', 'Freight', 'TotalDue']
        rows = []
        for i in range(n):
            sub = round(rng.uniform(10, 50000), 4)
            tax = round(sub * 0.08, 4)
            freight = round(sub * 0.02, 4)
            rows.append([i, rand_date(rng), rand_date(rng), rand_date(rng),
                         5, rng.randint(0, 1499),
                         rng.randint(0, 16) if rng.random() > 0.3 else None,
                         rng.randint(0, 9), sub, tax, freight,
                         round(sub + tax + freight, 4)])
    elif table_name == 'Sales.SalesOrderDetail':
        h = ['SalesOrderID', 'SalesOrderDetailID', 'ProductID',
             'OrderQty', 'UnitPrice', 'LineTotal']
        rows = []
        for i in range(n):
            qty = rng.randint(1, 20)
            price = round(rng.uniform(5, 3500), 4)
            rows.append([rng.randint(0, 2999), i, rng.randint(0, 503),
                         qty, price, round(qty * price, 4)])
    elif table_name == 'Sales.SalesTerritory':
        h = ['TerritoryID', 'Name', 'CountryRegionCode', 'Group',
             'SalesYTD', 'SalesLastYear']
        rows = [[i, TERRITORY_NAMES[i % len(TERRITORY_NAMES)],
                 COUNTRIES[i % len(COUNTRIES)],
                 TERRITORY_GROUPS[i % len(TERRITORY_GROUPS)],
                 round(rng.uniform(100000, 10000000), 4),
                 round(rng.uniform(100000, 10000000), 4)] for i in range(n)]
    elif table_name == 'Sales.SalesPerson':
        h = ['BusinessEntityID', 'TerritoryID', 'SalesQuota',
             'Bonus', 'CommissionPct', 'SalesYTD']
        rows = [[i, rng.randint(0, 9), round(rng.uniform(100000, 500000), 4),
                 round(rng.uniform(0, 10000), 4),
                 round(rng.uniform(0, 0.02), 4),
                 round(rng.uniform(500000, 5000000), 4)] for i in range(n)]
    elif table_name == 'Sales.SalesPersonQuotaHistory':
        h = ['BusinessEntityID', 'QuotaDate', 'SalesQuota', 'ModifiedDate']
        rows = [[rng.randint(0, 16), rand_date(rng),
                 round(rng.uniform(100000, 500000), 4),
                 rand_date(rng)] for i in range(n)]
    elif table_name == 'Sales.Currency':
        h = ['CurrencyCode', 'Name', 'ModifiedDate']
        rows = [[CURRENCIES[i % len(CURRENCIES)],
                 f'Currency_{i}', rand_date(rng)] for i in range(n)]
    elif table_name == 'Sales.CurrencyRate':
        h = ['CurrencyRateID', 'CurrencyRateDate', 'FromCurrencyCode',
             'ToCurrencyCode', 'AverageRate', 'EndOfDayRate']
        rows = [[i, rand_date(rng), 'USD', rng.choice(CURRENCIES),
                 round(rng.uniform(0.5, 150), 4),
                 round(rng.uniform(0.5, 150), 4)] for i in range(n)]
    elif table_name == 'Sales.SpecialOffer':
        h = ['SpecialOfferID', 'Description', 'DiscountPct',
             'Type', 'Category', 'StartDate', 'EndDate']
        rows = [[i, PROMOTIONS[i % len(PROMOTIONS)],
                 round(rng.uniform(0, 0.4), 4),
                 rng.choice(['Volume', 'Customer', 'Seasonal', 'Promo']),
                 rng.choice(['Reseller', 'Customer']),
                 rand_date(rng), rand_date(rng)] for i in range(n)]
    elif table_name == 'Sales.SalesReason':
        h = ['SalesReasonID', 'Name', 'ReasonType', 'ModifiedDate']
        rows = [[i, SALES_REASONS[i % len(SALES_REASONS)],
                 rng.choice(['Marketing', 'Promotion', 'Other']),
                 rand_date(rng)] for i in range(n)]
    elif table_name == 'Sales.SalesOrderHeaderSalesReason':
        h = ['SalesOrderID', 'SalesReasonID', 'ModifiedDate']
        rows = [[rng.randint(0, 2999), rng.randint(0, 9),
                 rand_date(rng)] for i in range(n)]
    elif table_name == 'Production.Product':
        h = ['ProductID', 'Name', 'ProductNumber', 'Color',
             'StandardCost', 'ListPrice', 'ProductSubcategoryID',
             'ProductModelID']
        rows = [[i, PRODUCT_NAMES[i % len(PRODUCT_NAMES)],
                 f'PN-{i:04d}', rng.choice(COLORS),
                 round(rng.uniform(5, 2000), 4),
                 round(rng.uniform(10, 3500), 4),
                 rng.randint(0, 36) if rng.random() > 0.2 else None,
                 rng.randint(0, 127) if rng.random() > 0.3 else None]
                for i in range(n)]
    elif table_name == 'Production.ProductSubcategory':
        h = ['ProductSubcategoryID', 'ProductCategoryID', 'Name', 'ModifiedDate']
        rows = [[i, rng.randint(0, 3), SUBCATEGORIES[i % len(SUBCATEGORIES)],
                 rand_date(rng)] for i in range(n)]
    elif table_name == 'Production.ProductCategory':
        h = ['ProductCategoryID', 'Name', 'ModifiedDate']
        rows = [[i, CATEGORIES[i % len(CATEGORIES)],
                 rand_date(rng)] for i in range(n)]
    elif table_name == 'Production.ProductModel':
        h = ['ProductModelID', 'Name', 'CatalogDescription',
             'Instructions', 'ModifiedDate']
        rows = [[i, f'Model-{i}', f'Desc for model {i}',
                 f'Instructions for model {i}', rand_date(rng)]
                for i in range(n)]
    elif table_name == 'Purchasing.PurchaseOrderHeader':
        h = ['PurchaseOrderID', 'EmployeeID', 'VendorID',
             'ShipMethodID', 'OrderDate', 'SubTotal', 'TaxAmt',
             'Freight', 'TotalDue']
        rows = []
        for i in range(n):
            sub = round(rng.uniform(100, 100000), 4)
            tax = round(sub * 0.08, 4)
            freight = round(sub * 0.015, 4)
            rows.append([i, rng.randint(0, 289), rng.randint(0, 103),
                         rng.randint(1, 5), rand_date(rng),
                         sub, tax, freight, round(sub + tax + freight, 4)])
    elif table_name == 'Purchasing.PurchaseOrderDetail':
        h = ['PurchaseOrderID', 'PurchaseOrderDetailID', 'ProductID',
             'OrderQty', 'UnitPrice', 'LineTotal', 'ReceivedQty', 'StockedQty']
        rows = []
        for i in range(n):
            qty = rng.randint(1, 500)
            price = round(rng.uniform(1, 2000), 4)
            rows.append([rng.randint(0, 999), i, rng.randint(0, 503),
                         qty, price, round(qty * price, 4), qty,
                         rng.randint(0, qty)])
    elif table_name == 'Purchasing.Vendor':
        h = ['BusinessEntityID', 'AccountNumber', 'Name',
             'CreditRating', 'ActiveFlag']
        rows = [[i, f'VEND{i:04d}', VENDOR_NAMES[i % len(VENDOR_NAMES)],
                 rng.randint(1, 5), rng.choice([1, 1, 1, 0])]
                for i in range(n)]
    elif table_name == 'HumanResources.Employee':
        h = ['BusinessEntityID', 'NationalIDNumber', 'LoginID',
             'JobTitle', 'BirthDate', 'Gender', 'HireDate',
             'SalariedFlag', 'VacationHours']
        rows = [[i, f'{rng.randint(100000000, 999999999)}',
                 f'adventure-works\\{rng.choice(FIRST_NAMES).lower()}{i}',
                 JOB_TITLES[i % len(JOB_TITLES)],
                 (BASE_DATE - timedelta(days=rng.randint(7300, 21900))).strftime('%Y-%m-%d'),
                 rng.choice(['M', 'F']), rand_date(rng),
                 rng.choice([0, 1]), rng.randint(0, 99)] for i in range(n)]
    elif table_name == 'HumanResources.Department':
        h = ['DepartmentID', 'Name', 'GroupName', 'ModifiedDate']
        rows = [[i, DEPARTMENTS[i % len(DEPARTMENTS)],
                 rng.choice(['Research and Development', 'Manufacturing',
                             'Sales and Marketing', 'Executive General',
                             'Inventory Management', 'Quality Assurance']),
                 rand_date(rng)] for i in range(n)]
    else:
        # Generic fallback
        h = ['id', 'value1', 'value2', 'modified_date']
        rows = [[i, f'val_{rng.randint(1, 10000)}',
                 round(rng.uniform(0, 1000), 2), rand_date(rng)]
                for i in range(n)]
    return h, rows


def gen_staging(table_name, n, rng):
    """Generate staging view rows."""
    info = STAGING_VIEWS[table_name]
    return info['cols'], [[i] + [
        f'stg_val_{rng.randint(1, 10000)}' if isinstance(c, str) and c != 'staging_id'
        else rng.randint(0, 9999)
        for c in info['cols'][1:]
    ] for i in range(n)]


def gen_dw(table_name, n, rng):
    """Generate DW dim/fact rows."""
    info = DW_TABLES[table_name]
    return info['cols'], [[i] + [
        round(rng.uniform(1, 5000), 2) if 'Price' in c or 'Amount' in c or 'Cost' in c or 'Rate' in c
        else rng.randint(0, 999) if 'Key' in c and c != info['cols'][0]
        else rng.randint(1, 100) if 'Qty' in c or 'Quantity' in c or 'Units' in c
        else rand_date(rng) if 'Date' in c
        else rng.choice(FIRST_NAMES) if 'FirstName' in c
        else rng.choice(LAST_NAMES) if 'LastName' in c
        else rng.choice(CITIES) if 'City' in c
        else rng.choice(PRODUCT_NAMES) if 'ProductName' in c or 'EnglishProduct' in c
        else rng.choice(CATEGORIES) if 'Category' in c and 'Key' not in c
        else rng.choice(TERRITORY_NAMES) if 'Territory' in c and 'Key' not in c
        else rng.choice(CURRENCIES) if 'Currency' in c and 'Key' not in c
        else f'dw_val_{rng.randint(1, 10000)}'
        for c in info['cols'][1:]
    ] for i in range(n)]


# ──────────────────────────────────────────────────────────────────
# Lineage map builder
# ──────────────────────────────────────────────────────────────────
def build_lineage_map(manifest):
    """Build row-level lineage: DW→Staging→OLTP.

    Returns dict: {target_table: {row_key: {sources: [{table, rows}]}}}
    """
    lineage_map = {}
    rng = random.Random(12345)  # Separate seed for lineage

    # Layer 1: Staging → OLTP
    for stg_name, stg_info in STAGING_VIEWS.items():
        stg_rows = stg_info['rows']
        oltp_sources = stg_info['oltp_sources']
        lineage_map[stg_name] = {}

        for row_idx in range(stg_rows):
            # Each staging row derives from 1-3 source rows per OLTP table
            sources = []
            for oltp_table in oltp_sources:
                oltp_n = OLTP_TABLES[oltp_table]['rows']
                n_src = rng.randint(1, min(3, oltp_n))
                src_rows = rng.sample(range(oltp_n), n_src)
                sources.append({
                    'table': oltp_table,
                    'rows': sorted(src_rows),
                })
            lineage_map[stg_name][f'row_{row_idx}'] = {'sources': sources}

    # Layer 2: DW → Staging
    for dw_name, dw_info in DW_TABLES.items():
        dw_rows = dw_info['rows']
        stg_source = dw_info['staging_source']
        stg_n = STAGING_VIEWS[stg_source]['rows']
        lineage_map[dw_name] = {}

        for row_idx in range(dw_rows):
            n_src = rng.randint(1, min(3, stg_n))
            src_rows = rng.sample(range(stg_n), n_src)
            lineage_map[dw_name][f'row_{row_idx}'] = {
                'sources': [{
                    'table': stg_source,
                    'rows': sorted(src_rows),
                }]
            }

    return lineage_map


# ──────────────────────────────────────────────────────────────────
# Main generation
# ──────────────────────────────────────────────────────────────────
def generate_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    total_rows = 0

    # Generate OLTP tables
    print('  OLTP tables:')
    for table_name, info in sorted(OLTP_TABLES.items()):
        n = info['rows']
        rng = random.Random(hash(table_name) & 0xFFFFFFFF)
        headers, rows = gen_oltp(table_name, n, rng)
        safe_name = table_name.replace('.', '_')
        csv_path = OUT_DIR / f'{safe_name}.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        manifest[table_name] = {
            'rows': len(rows), 'columns': len(headers),
            'layer': 'oltp', 'file': f'{safe_name}.csv'}
        total_rows += len(rows)
        print(f'    {table_name:45s}: {len(rows):>6,} rows × {len(headers)} cols')

    # Generate Staging views
    print('\n  Staging views:')
    for table_name, info in sorted(STAGING_VIEWS.items()):
        n = info['rows']
        rng = random.Random(hash(table_name) & 0xFFFFFFFF)
        headers, rows = gen_staging(table_name, n, rng)
        csv_path = OUT_DIR / f'{table_name}.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        manifest[table_name] = {
            'rows': len(rows), 'columns': len(headers),
            'layer': 'staging', 'file': f'{table_name}.csv'}
        total_rows += len(rows)
        print(f'    {table_name:45s}: {len(rows):>6,} rows × {len(headers)} cols')

    # Generate DW tables
    print('\n  DW tables:')
    for table_name, info in sorted(DW_TABLES.items()):
        n = info['rows']
        rng = random.Random(hash(table_name) & 0xFFFFFFFF)
        headers, rows = gen_dw(table_name, n, rng)
        csv_path = OUT_DIR / f'{table_name}.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        manifest[table_name] = {
            'rows': len(rows), 'columns': len(headers),
            'layer': 'dw', 'file': f'{table_name}.csv'}
        total_rows += len(rows)
        print(f'    {table_name:45s}: {len(rows):>6,} rows × {len(headers)} cols')

    # Save manifest
    with open(OUT_DIR / '_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    # Build and save lineage map
    print('\n  Building lineage map...')
    lineage_map = build_lineage_map(manifest)
    lineage_path = DS_DIR / 'lineage_map.json'
    with open(lineage_path, 'w', encoding='utf-8') as f:
        json.dump(lineage_map, f, indent=2)

    # Summary
    n_oltp = len([t for t in manifest if manifest[t]['layer'] == 'oltp'])
    n_stg = len([t for t in manifest if manifest[t]['layer'] == 'staging'])
    n_dw = len([t for t in manifest if manifest[t]['layer'] == 'dw'])
    print(f'\n  Total: {len(manifest)} tables ({n_oltp} OLTP + {n_stg} staging + {n_dw} DW)')
    print(f'  Total rows: {total_rows:,}')
    print(f'  Lineage map: {len(lineage_map)} tables with row-level lineage')
    print(f'  Saved to: {OUT_DIR}')
    return manifest


def validate(manifest=None):
    if manifest is None:
        mp = OUT_DIR / '_manifest.json'
        if not mp.exists():
            print('❌ No manifest found.')
            return False
        with open(mp, 'r', encoding='utf-8') as f:
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
            errors.append(f'{table_name}: cols {len(header)} != {info["columns"]}')
        if row_count != info['rows']:
            errors.append(f'{table_name}: rows {row_count} != {info["rows"]}')

    if errors:
        print(f'❌ {len(errors)} validation errors:')
        for e in errors:
            print(f'  {e}')
        return False

    # Validate lineage map
    lm_path = DS_DIR / 'lineage_map.json'
    if lm_path.exists():
        with open(lm_path, 'r', encoding='utf-8') as f:
            lm = json.load(f)
        print(f'✅ Lineage map: {len(lm)} tables')
    else:
        print('⚠️  No lineage_map.json found')

    total = sum(m['rows'] for m in manifest.values())
    print(f'✅ All {len(manifest)} CSVs validated ({total:,} total rows)')
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate AdventureWorks value data')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    if args.validate:
        exit(0 if validate() else 1)

    print('Generating AdventureWorks value data (3-layer pipeline)...')
    manifest = generate_all()
    print('\nValidating...')
    validate(manifest)
    print('\n✅ Done!')


if __name__ == '__main__':
    main()
