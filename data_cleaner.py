#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def clean_financial_data(data_directory_root):
    """
    Cleans financial data for all companies by removing entries before 2018.

    Args:
        data_directory_root (str): The root directory containing company financial data.
                                     It's expected to have subdirectories for each company,
                                     and each company directory should contain a file named
                                     '*_finnhub_annual_financials.json'.
    """
    print(f"Starting data cleaning process in: {data_directory_root}")

    for company_dir_name in os.listdir(data_directory_root):
        company_path = os.path.join(data_directory_root, company_dir_name)
        if os.path.isdir(company_path):
            print(f"Processing company: {company_dir_name}")
            annual_financial_file_name = f"{company_dir_name.lower()}_finnhub_annual_financials.json"
            annual_financial_file_path = os.path.join(company_path, annual_financial_file_name)

            if os.path.exists(annual_financial_file_path):
                print(f"  Found annual financial file: {annual_financial_file_path}")
                try:
                    with open(annual_financial_file_path, 'r', encoding='utf-8') as f:
                        financial_data = json.load(f)

                    if not isinstance(financial_data, dict) or 'data' not in financial_data or not isinstance(financial_data['data'], list):
                        print(f"    Skipping {annual_financial_file_path}: 'data' key not found or not a list.")
                        continue

                    original_reports_count = len(financial_data['data'])
                    # Filter out reports where the year is before 2018
                    # Assuming each item in 'data' list has a 'year' field
                    cleaned_reports = [report for report in financial_data['data'] if isinstance(report, dict) and report.get('year') is not None and int(report.get('year')) >= 2020]
                    
                    if len(cleaned_reports) < original_reports_count:
                        financial_data['data'] = cleaned_reports
                        with open(annual_financial_file_path, 'w', encoding='utf-8') as f:
                            json.dump(financial_data, f, indent=4)
                        print(f"    Cleaned data for {company_dir_name}. Removed {original_reports_count - len(cleaned_reports)} entries before 2020.")
                    else:
                        print(f"    No data before 2020 found for {company_dir_name}. No changes made.")

                except json.JSONDecodeError:
                    print(f"    Error decoding JSON from {annual_financial_file_path}. Skipping.")
                except Exception as e:
                    print(f"    An error occurred while processing {annual_financial_file_path}: {e}. Skipping.")
            else:
                # Try to find a file that matches the pattern if the exact name isn't found
                found_file = False
                for file_name in os.listdir(company_path):
                    if file_name.endswith('_finnhub_annual_financials.json'):
                        annual_financial_file_path = os.path.join(company_path, file_name)
                        print(f"  Found annual financial file (alternative match): {annual_financial_file_path}")
                        try:
                            with open(annual_financial_file_path, 'r', encoding='utf-8') as f:
                                financial_data = json.load(f)

                            if not isinstance(financial_data, dict) or 'data' not in financial_data or not isinstance(financial_data['data'], list):
                                print(f"    Skipping {annual_financial_file_path}: 'data' key not found or not a list.")
                                continue

                            original_reports_count = len(financial_data['data'])
                            cleaned_reports = [report for report in financial_data['data'] if isinstance(report, dict) and report.get('year') is not None and int(report.get('year')) >= 2020]
                            
                            if len(cleaned_reports) < original_reports_count:
                                financial_data['data'] = cleaned_reports
                                with open(annual_financial_file_path, 'w', encoding='utf-8') as f:
                                    json.dump(financial_data, f, indent=4)
                                print(f"    Cleaned data for {company_dir_name}. Removed {original_reports_count - len(cleaned_reports)} entries before 2020.")
                            else:
                                print(f"    No data before 2020 found for {company_dir_name}. No changes made.")
                            found_file = True
                            break # Found and processed the file
                        except json.JSONDecodeError:
                            print(f"    Error decoding JSON from {annual_financial_file_path}. Skipping.")
                        except Exception as e:
                            print(f"    An error occurred while processing {annual_financial_file_path}: {e}. Skipping.")
                if not found_file:
                     print(f"  Annual financial file not found for {company_dir_name} (expected: {annual_financial_file_name} or similar). Skipping.")

    print("Data cleaning process finished.")

if __name__ == '__main__':
    # The script assumes it's run from the root of the 'Stock' project directory
    # or that the path to 'financial_data' is correctly specified.
    financial_data_path = os.path.join(os.path.dirname(__file__), 'financial_data')
    
    # Check if the default path exists, otherwise prompt user or use a predefined alternative
    if not os.path.isdir(financial_data_path):
        print(f"Default financial_data directory not found at: {financial_data_path}")
        # Attempt to use the path from the workspace context if available and different
        # This is a placeholder for more robust path handling if needed
        workspace_financial_data_path = "/Users/apple/PycharmProjects/Stock/financial_data/"
        if os.path.isdir(workspace_financial_data_path):
            print(f"Using workspace financial_data directory: {workspace_financial_data_path}")
            financial_data_path = workspace_financial_data_path
        else:
            print("Please ensure the 'financial_data' directory is correctly located or update the path in the script.")
            exit(1)
            
    clean_financial_data(financial_data_path)