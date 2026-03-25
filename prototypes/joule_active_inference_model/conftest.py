"""
conftest.py
-----------
Pytest configuration. Makes the dmbd_joule package importable
without installation or sys.path manipulation in test files.
"""
import sys
import os

# Add repo root to sys.path so 'import dmbd_joule' works from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
