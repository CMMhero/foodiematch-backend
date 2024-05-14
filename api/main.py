import numpy as np
import pandas as pd
from flask import Flask
from supabase import Client, create_client

# Supabase config
supabase_url = 'https://knuqcixcjyymqbzekkqc.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtudXFjaXhjanl5bXFiemVra3FjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTM5NDIzNTYsImV4cCI6MjAyOTUxODM1Nn0.hr5kkLMIM0GLIC_Dp3m2Pq_qdCgXYYxRWwt333Xc9xs'
supabase = create_client(supabase_url, supabase_key)

app = Flask(__name__)

@app.route('/')
def home():
    return "hello world"

@app.route('/about')
def about():
    return "nice"