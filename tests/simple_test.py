#!/usr/bin/env python3
"""
Simple test to verify RAG system works
"""

import requests
import json

# Test document
test_content = """Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data. The algorithms use computational methods to learn information directly from data. Machine learning algorithms build a mathematical model based on sample data."""

# Upload test
files = {'file': ('test.txt', test_content, 'text/plain')}
response = requests.post('http://localhost:8000/api/upload', files=files)

print("Upload Response:")
print(response.status_code)
print(response.json())

# Check stats
stats = requests.get('http://localhost:8000/api/stats').json()
print("\nStats:")
print(stats)

# Test search
if stats['vectors_count'] > 0:
    search_data = {'message': 'What is machine learning?'}
    search_response = requests.post('http://localhost:8000/api/chat', json=search_data)
    print("\nSearch Response:")
    print(search_response.status_code)
    print(search_response.json())
else:
    print("\nNo vectors to search")
