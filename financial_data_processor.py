import os
import json
import pandas as pd

def load_financial_data(company_symbol):
    """加载指定公司的财报数据"""
    data_path = os.path.join('financial_data', company_symbol.lower())
    annual_financials_path = os.path.join(data_path, f'{company_symbol.lower()}_finnhub_annual_financials.json')
    quarterly_financials_path = os.path.join(data_path, f'{company_symbol.lower()}_finnhub_quarterly_financials.json')

    annual_financials = []
    quarterly_financials = []

    if os.path.exists(annual_financials_path):
        with open(annual_financials_path, 'r') as f:
            # 修正：Finnhub API 返回的 JSON 结构中，财务数据存储在 'data' 字段中，而不是 'financials' 字段
            annual_financials = json.load(f).get('data', [])
            
    if os.path.exists(quarterly_financials_path):
        with open(quarterly_financials_path, 'r') as f:
            # 修正：Finnhub API 返回的 JSON 结构中，财务数据存储在 'data' 字段中，而不是 'financials' 字段
            quarterly_financials = json.load(f).get('data', [])
            
    return annual_financials, quarterly_financials

def extract_financial_metrics(annual_financials, quarterly_financials):
    """从财报数据中提取关键财务指标"""
    all_metrics = []
    
    # 提取年度报告中的指标
    for statement in annual_financials:
        report = statement.get('report', {})
        # 修改：从 statement 对象获取 year 字段，而不是从 report 对象
        year = statement.get('year')
        period = 'FY'
        
        if year:
            # 提取所有需要的指标
            # 优先从 'ic' (Income Statement) 和 'bs' (Balance Sheet) 部分提取
            # 修改：增加更多可能的概念名称来匹配不同公司的财务报表格式
            revenue = next((item['value'] for item in report.get('ic', []) 
                          if item['concept'] in ['us-gaap_Revenues', 'us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax', 
                                                'us-gaap_SalesRevenueNet', 'us-gaap_SalesRevenueGoodsNet']), None)
            net_income = next((item['value'] for item in report.get('ic', []) 
                             if item['concept'] in ['us-gaap_NetIncomeLoss', 'us-gaap_ProfitLoss', 
                                                   'us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic']), None)
            eps = next((item['value'] for item in report.get('ic', []) 
                       if item['concept'] in ['us-gaap_EarningsPerShareBasic', 'us-gaap_EarningsPerShareBasicAndDiluted', 
                                             'us-gaap_IncomeLossPerShareBasic']), None)
            total_assets = next((item['value'] for item in report.get('bs', []) 
                               if item['concept'] in ['us-gaap_Assets', 'us-gaap_AssetsCurrent', 'us-gaap_AssetsTotal']), None)
            total_liabilities = next((item['value'] for item in report.get('bs', []) 
                                   if item['concept'] in ['us-gaap_Liabilities', 'us-gaap_LiabilitiesCurrent', 'us-gaap_LiabilitiesTotal']), None)
            equity = next((item['value'] for item in report.get('bs', []) 
                         if item['concept'] in ['us-gaap_StockholdersEquity', 'us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 
                                               'us-gaap_StockholdersEquityParentCompany']), None)

            # 如果在 'ic' 或 'bs' 中未找到，尝试从 report 顶层获取（针对某些 Finnhub 报告结构）
            # 修正：Finnhub 的数据通常在 'ic' 或 'bs' 中，顶层字段可能不常用或不准确
            # 保持原有的备用逻辑，但强调优先从 'ic'/'bs' 提取
            if revenue is None: revenue = report.get('revenue')
            if net_income is None: net_income = report.get('netIncome')
            if eps is None: eps = report.get('eps')
            if total_assets is None: total_assets = report.get('totalAssets')
            if total_liabilities is None: total_liabilities = report.get('totalLiabilities')
            if equity is None: equity = report.get('equity')
            
            all_metrics.append({
                'year': year,
                'period': period,
                'revenue': revenue,
                'netIncome': net_income,
                'eps': eps,
                'totalAssets': total_assets,
                'totalLiabilities': total_liabilities,
                'equity': equity
            })

    # 提取季度报告中的指标
    for statement in quarterly_financials:
        report = statement.get('report', {})
        # 修改：从 statement 对象获取 year 和 quarter 字段，而不是从 report 对象
        year = statement.get('year')
        period = statement.get('quarter') # Finnhub 季度报告有 'quarter' 字段
        
        if year and period:
            # 提取所有需要的指标
            # 优先从 'ic' (Income Statement) 和 'bs' (Balance Sheet) 部分提取
            # 修改：增加更多可能的概念名称来匹配不同公司的财务报表格式
            revenue = next((item['value'] for item in report.get('ic', []) 
                          if item['concept'] in ['us-gaap_Revenues', 'us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax', 
                                                'us-gaap_SalesRevenueNet', 'us-gaap_SalesRevenueGoodsNet']), None)
            net_income = next((item['value'] for item in report.get('ic', []) 
                             if item['concept'] in ['us-gaap_NetIncomeLoss', 'us-gaap_ProfitLoss', 
                                                   'us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic']), None)
            eps = next((item['value'] for item in report.get('ic', []) 
                       if item['concept'] in ['us-gaap_EarningsPerShareBasic', 'us-gaap_EarningsPerShareBasicAndDiluted', 
                                             'us-gaap_IncomeLossPerShareBasic']), None)
            total_assets = next((item['value'] for item in report.get('bs', []) 
                               if item['concept'] in ['us-gaap_Assets', 'us-gaap_AssetsCurrent', 'us-gaap_AssetsTotal']), None)
            total_liabilities = next((item['value'] for item in report.get('bs', []) 
                                   if item['concept'] in ['us-gaap_Liabilities', 'us-gaap_LiabilitiesCurrent', 'us-gaap_LiabilitiesTotal']), None)
            equity = next((item['value'] for item in report.get('bs', []) 
                         if item['concept'] in ['us-gaap_StockholdersEquity', 'us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 
                                               'us-gaap_StockholdersEquityParentCompany']), None)

            # 如果在 'ic' 或 'bs' 中未找到，尝试从 report 顶层获取（针对某些 Finnhub 报告结构）
            # 修正：Finnhub 的数据通常在 'ic' 或 'bs' 中，顶层字段可能不常用或不准确
            # 保持原有的备用逻辑，但强调优先从 'ic'/'bs' 提取
            if revenue is None: revenue = report.get('revenue')
            if net_income is None: net_income = report.get('netIncome')
            if eps is None: eps = report.get('eps')
            if total_assets is None: total_assets = report.get('totalAssets')
            if total_liabilities is None: total_liabilities = report.get('totalLiabilities')
            if equity is None: equity = report.get('equity')
            
            all_metrics.append({
                'year': year,
                'period': f'Q{period}', # 格式化为 Q1, Q2 等
                'revenue': revenue,
                'netIncome': net_income,
                'eps': eps,
                'totalAssets': total_assets,
                'totalLiabilities': total_liabilities,
                'equity': equity
            })
                
    return pd.DataFrame(all_metrics)


if __name__ == '__main__':
    company_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD', 'META']
    
    for symbol in company_symbols:
        print(f"Processing financial data for {symbol}...")
        annual_financials, quarterly_financials = load_financial_data(symbol)
        financial_df = extract_financial_metrics(annual_financials, quarterly_financials)
        
        if not financial_df.empty:
            print(f"Financial metrics for {symbol}:\n{financial_df.head()}\n")
            # 可以选择将处理后的数据保存到CSV或数据库
            output_dir = os.path.join('processed_financial_data', symbol.lower())
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{symbol.lower()}_financial_metrics.csv')
            financial_df.to_csv(output_path, index=False)
            print(f"Processed financial data saved to {output_path}")
        else:
            print(f"No financial data found or extracted for {symbol}.")
        print("-" * 30)