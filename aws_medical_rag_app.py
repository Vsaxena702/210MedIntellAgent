#!/usr/bin/env python3
"""
AWS Production Deployment - Medical RAG SQL Query System
Optimized for EC2 + RDS PostgreSQL deployment with Knowledge Graph Visualization
"""

import os
import re
import logging
import psycopg2
import psycopg2.extras
import psycopg2.pool
from datetime import datetime
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv
import gradio as gr
import pandas as pd
from enhanced_gradio_ui_rag_cag import create_enhanced_gradio_interface

# Load environment variables
load_dotenv()

# Configure logging FIRST (before any imports that use logger)
log_level = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Knowledge Graph imports
try:
    from knowledge_graph import MedicalKGExtractor, MedicalGraphBuilder, MedicalGraphVisualizer
    KG_AVAILABLE = True
except ImportError:
    logger.warning("⚠️  Knowledge Graph modules not available - KG visualization disabled")
    KG_AVAILABLE = False

# Deep Thinking RAG - lazy import to avoid circular dependency
DEEP_THINKING_AVAILABLE = False
IntegratedMedicalRAG = None

def _load_deep_thinking_rag():
    """Lazy load Deep Thinking RAG to avoid circular imports"""
    global DEEP_THINKING_AVAILABLE, IntegratedMedicalRAG
    if IntegratedMedicalRAG is None:
        try:
            from integrated_medical_rag import IntegratedMedicalRAG as _IntegratedMedicalRAG
            IntegratedMedicalRAG = _IntegratedMedicalRAG
            DEEP_THINKING_AVAILABLE = True
            logger.info("✅ Deep Thinking RAG loaded successfully")
        except ImportError as e:
            DEEP_THINKING_AVAILABLE = False
            logger.warning(f"⚠️  Deep Thinking RAG not available: {e}")
    return DEEP_THINKING_AVAILABLE

# ==============================================================================
# AWS SECRETS MANAGER INTEGRATION
# ==============================================================================

def load_rds_credentials_from_secrets_manager():
    """
    Load RDS credentials from AWS Secrets Manager
    Returns dict with host, port, username, password, dbname
    """
    try:
        import json
        import boto3
        from botocore.exceptions import ClientError
        
        secret_name = os.getenv("AWS_SECRET_NAME", "rds!db-e1e88d94-089d-4f4f-96e1-36df00099e6b")
        region_name = os.getenv("AWS_REGION", "us-east-1")
        
        logger.info(f"🔐 Loading credentials from AWS Secrets Manager: {secret_name}")
        
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=region_name)
        
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret_string = get_secret_value_response['SecretString']
        secret_dict = json.loads(secret_string)
        
        logger.info("✅ Credentials loaded from AWS Secrets Manager")
        return secret_dict
        
    except ClientError as e:
        logger.error(f"❌ Failed to retrieve secret: {e}")
        return None
    except ImportError:
        logger.warning("⚠️  boto3 not installed - using environment variables instead")
        return None

# ==============================================================================
# CONFIGURATION - AWS RDS & Application Settings
# ==============================================================================

class Config:
    """Application configuration from environment variables or AWS Secrets Manager"""
    
    # Try to load from AWS Secrets Manager first
    USE_SECRETS_MANAGER = os.getenv("USE_AWS_SECRETS_MANAGER", "true").lower() == "true"
    _secrets = None
    
    if USE_SECRETS_MANAGER:
        _secrets = load_rds_credentials_from_secrets_manager()
    
    # AWS RDS PostgreSQL - prioritize Secrets Manager, fallback to env vars
    if _secrets:
        DB_HOST = _secrets.get('host', os.getenv("DB_HOST", "localhost"))
        DB_PORT = int(_secrets.get('port', os.getenv("DB_PORT", "5432")))
        DB_NAME = _secrets.get('dbname', os.getenv("DB_NAME", "postgres"))
        DB_USER = _secrets.get('username', os.getenv("DB_USER", "postgres"))
        DB_PASSWORD = _secrets.get('password', os.getenv("DB_PASSWORD", ""))
    else:
        DB_HOST = os.getenv("DB_HOST", "localhost")
        DB_PORT = int(os.getenv("DB_PORT", "5432"))
        DB_NAME = os.getenv("DB_NAME", "postgres")
        DB_USER = os.getenv("DB_USER", "postgres")
        DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # Match FAISS build
    
    # Groq (for ultra-fast SQL generation)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_SQL_MODEL = os.getenv("GROQ_SQL_MODEL", "llama-3.1-8b-instant")
    USE_GROQ_FOR_SQL = os.getenv("USE_GROQ_FOR_SQL", "true").lower() == "true"
    
    # Knowledge Graph (disabled for deep thinking to provide faster responses)
    ENABLE_KNOWLEDGE_GRAPH = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "false").lower() == "true"
    
    # Use absolute paths for FAISS and examples
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    FAISS_PATH = os.getenv("FAISS_PATH", os.path.join(_base_dir, "sql-vectorstore_210", "medintellagent_faiss_v1_1"))
    EXAMPLES_JSON = os.getenv("EXAMPLES_JSON", os.path.join(_base_dir, "medintellagent_examples_all_intents_01.json"))
    
    # Gradio
    SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        # Log the actual paths being checked
        logger.info(f"   Validating configuration...")
        logger.info(f"   Base directory: {cls._base_dir}")
        logger.info(f"   FAISS_PATH: {cls.FAISS_PATH}")
        logger.info(f"   FAISS_PATH exists: {os.path.exists(cls.FAISS_PATH)}")
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY not set")
        if cls.USE_GROQ_FOR_SQL and not cls.GROQ_API_KEY:
            logger.warning("⚠️  GROQ_API_KEY not set - will use GPT-4o-mini for SQL (slower)")
            cls.USE_GROQ_FOR_SQL = False
        if not cls.DB_PASSWORD:
            errors.append("DB_PASSWORD not set (check AWS Secrets Manager or .env file)")
        if not os.path.exists(cls.FAISS_PATH):
            errors.append(f"FAISS vector store not found at {cls.FAISS_PATH}")
            # List what's actually there
            parent_dir = os.path.dirname(cls.FAISS_PATH)
            if os.path.exists(parent_dir):
                logger.error(f"   Parent directory exists: {parent_dir}")
                logger.error(f"   Contents: {os.listdir(parent_dir)}")
            else:
                logger.error(f"   Parent directory doesn't exist: {parent_dir}")
            
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        logger.info("✅ Configuration validated successfully")
        logger.info(f"   DB Host: {cls.DB_HOST}")
        logger.info(f"   DB Name: {cls.DB_NAME}")
        logger.info(f"   DB User: {cls.DB_USER}")
        logger.info(f"   Credentials from: {'AWS Secrets Manager' if cls._secrets else 'Environment Variables'}")
        logger.info(f"   FAISS Path: {cls.FAISS_PATH}")
        logger.info(f"   SQL Generation: {'Groq (ultra-fast)' if cls.USE_GROQ_FOR_SQL else 'GPT-4o-mini'}")

# ==============================================================================
# DATABASE CONNECTION POOL 
# ==============================================================================

# Global connection pool - initialized once, reused for all queries
_connection_pool = None

def initialize_connection_pool():
    """Initialize database connection pool for fast query execution"""
    global _connection_pool
    
    if _connection_pool is not None:
        logger.info("Connection pool already initialized")
        return _connection_pool
    
    try:
        logger.info(" Initializing database connection pool...")
        _connection_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=2,  # Minimum connections
            maxconn=10,  # Maximum connections
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            connect_timeout=10,
            options="-c statement_timeout=30000"
        )
        logger.info("✅ Connection pool initialized (2-10 connections)")
        return _connection_pool
    except psycopg2.Error as e:
        logger.error(f"❌ Failed to initialize connection pool: {e}")
        raise

def get_pooled_connection():
    """Get connection from pool (fast!)"""
    global _connection_pool
    
    if _connection_pool is None:
        initialize_connection_pool()
    
    try:
        conn = _connection_pool.getconn()
        return conn
    except psycopg2.pool.PoolError as e:
        logger.error(f"Failed to get connection from pool: {e}")
        raise

def return_pooled_connection(conn):
    """Return connection to pool for reuse"""
    global _connection_pool
    
    if _connection_pool is not None and conn is not None:
        _connection_pool.putconn(conn)

# ==============================================================================
# DATABASE CONNECTION & SQL EXECUTION
# ==============================================================================

_SELECT_ONLY = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)
_PARAM = re.compile(r":(\w+)")

def get_database_connection():
    """Get database connection to AWS RDS"""
    try:
        conn = psycopg2.connect(
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            connect_timeout=10,
            options="-c statement_timeout=30000"
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise

def validate_sql_safety(sql: str) -> bool:
    """Validate SQL is safe to execute"""
    sql_lower = sql.lower().strip()
    
    # Must be SELECT only
    if not _SELECT_ONLY.match(sql):
        logger.warning(f"Rejected non-SELECT query: {sql[:100]}")
        return False
    
    # Block dangerous keywords
    dangerous = ["drop", "delete", "update", "insert", "alter", "create", "truncate", "grant", "revoke"]
    for keyword in dangerous:
        if re.search(rf"\b{keyword}\b", sql_lower):
            logger.warning(f"Rejected SQL with dangerous keyword '{keyword}'")
            return False
    
    # Require patient_id parameter for patient safety
    if "patient_id" not in sql_lower:
        logger.warning("SQL query missing patient_id parameter")
    
    return True

def convert_to_psycopg2_format(sql: str) -> str:
    """
    Convert :parameter to %(parameter)s format
    Handles multiple formats to ensure consistency
    Also escapes % in LIKE patterns
    """
    import re
    
    # First, escape % in LIKE patterns (e.g., LIKE '%text%' becomes LIKE '%%text%%')
    # Match LIKE followed by a string with %
    def escape_like_pattern(match):
        like_part = match.group(0)
        # Replace % with %% in the string literal
        return like_part.replace("'%", "'%%").replace("%'", "%%'")
    
    sql = re.sub(r"LIKE\s+'[^']*%[^']*'", escape_like_pattern, sql, flags=re.IGNORECASE)
    
    # Convert any existing %(param)s back to :param to normalize
    sql = re.sub(r'%\((\w+)\)s', r':\1', sql)
    
    # Now convert all :param to %(param)s format
    def replace_param(match):
        param_name = match.group(1)
        return f"%({param_name})s"
    
    converted = _PARAM.sub(replace_param, sql)
    return converted

def execute_sql_query(sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Execute SQL query against AWS RDS PostgreSQL database using connection pool
    
    Args:
        sql: SQL query with :parameter placeholders
        params: Dictionary of parameter values
        
    Returns:
        DataFrame with query results
    """
    # Validate SQL safety
    if not validate_sql_safety(sql):
        raise ValueError("SQL query failed safety validation")
    
    # Log original SQL for debugging (COMPLETE SQL)
    logger.info(f"📝 Original SQL query:\n{sql}")
    
    # Convert parameter format
    sql_converted = convert_to_psycopg2_format(sql)
    
    # Log converted SQL for debugging  
    logger.info(f"🔄 Converted SQL:\n{sql_converted}")
    logger.info(f"🔑 Parameters: {params}")
    
    # Set defaults
    params = params or {}
    if "patient_id" not in params:
        params["patient_id"] = None
    
    logger.info(f"Executing query with patient_id={params.get('patient_id')}")
    logger.debug(f"SQL: {sql_converted[:200]}...")
    
    conn = None
    try:
        # Get connection from pool 
        conn = get_pooled_connection()
        df = pd.read_sql_query(sql_converted, conn, params=params)
        
        logger.info(f"Query returned {len(df)} rows")
        
        # Post-process medication results to extract dose and route from med_name
        if 'med_name' in df.columns:
            import re
            
            # Add parsed dose column
            def extract_dose(med_name):
                if pd.isna(med_name):
                    return None
                dose_match = re.search(r'(\d+\.?\d*)\s*(MG|ML|MCG|G|UNITS?(?:/ML)?|%)', str(med_name), re.IGNORECASE)
                if dose_match:
                    return f"{dose_match.group(1)} {dose_match.group(2).upper()}"
                return None
            
            # Add parsed route column
            def extract_route(med_name):
                if pd.isna(med_name):
                    return None
                med_lower = str(med_name).lower()
                if 'oral' in med_lower:
                    return 'Oral'
                elif 'topical' in med_lower:
                    return 'Topical'
                elif 'inject' in med_lower:
                    return 'Injectable'
                elif 'inhal' in med_lower:
                    return 'Inhalation'
                elif 'nasal' in med_lower:
                    return 'Nasal'
                return None
            
            # Override dose and route columns with parsed values
            df['dose'] = df['med_name'].apply(extract_dose)
            df['route'] = df['med_name'].apply(extract_route)
            
            logger.info(f"✅ Parsed dose and route from med_name for {len(df)} medications")
        
        # Filter out social determinants from conditions queries
        if 'display' in df.columns and 'code' in df.columns:
            # Social determinant keywords to exclude (case-insensitive)
            social_keywords = [
                'employment', 'employed', 'unemployed', 'job',
                'education', 'school', 'college', 'degree',
                'criminal', 'housing', 'homelessness', 'homeless',
                'stress', 'social isolation', 'victim of', 'refugee',
                'body mass index', 'bmi', 'obesity', 'overweight',  # BMI findings
                'reports of violence', 'violence in the environment',
                'lack of access', 'transportation', 'food insecurity'
            ]
            
            # Filter function
            def is_medical_condition(display_text):
                if pd.isna(display_text):
                    return True
                display_lower = str(display_text).lower()
                # Exclude if it contains 'finding' AND a social keyword
                if '(finding)' in display_lower:
                    return not any(keyword in display_lower for keyword in social_keywords)
                # Keep all disorders
                if '(disorder)' in display_lower:
                    return True
                # For other types, check against social keywords
                return not any(keyword in display_lower for keyword in social_keywords)
            
            original_len = len(df)
            df = df[df['display'].apply(is_medical_condition)].copy()
            filtered_count = original_len - len(df)
            
            if filtered_count > 0:
                logger.info(f"🔍 Filtered out {filtered_count} social determinant records, kept {len(df)} medical conditions")
        
        return df
        
    except psycopg2.Error as e:
        logger.error(f"Database query error: {e}")
        raise
    finally:
        if conn:
            # Return connection to pool (don't close it!)
            return_pooled_connection(conn)

# ==============================================================================
# VECTOR STORE & FEW-SHOT RETRIEVAL
# ==============================================================================

def load_vector_store():
    """Load FAISS vector store with medical examples"""
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        
        logger.info(f"Loading FAISS vector store from {Config.FAISS_PATH}")
        
        embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        vectorstore = FAISS.load_local(
            Config.FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Get example count
        example_count = len(vectorstore.docstore._dict)
        logger.info(f"✅ Loaded {example_count} examples from FAISS")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Failed to load FAISS vector store: {e}")
        raise

def get_few_shot_examples(vectorstore, question: str, k: int = 3) -> List[Dict]:
    """Retrieve similar examples from vector store"""
    try:
        # Guard against None inputs
        if vectorstore is None:
            logger.warning("Vector store is None, returning empty examples")
            return []
        
        if question is None or not isinstance(question, str):
            logger.warning(f"Invalid question type: {type(question)}, returning empty examples")
            return []
        
        logger.debug(f"Searching for similar examples: {question[:100]}...")
        
        # Try to retrieve examples - catch FAISS dimension mismatch errors
        try:
            docs_and_scores = vectorstore.similarity_search_with_score(question, k=k)
        except (AssertionError, ValueError) as faiss_error:
            logger.warning(f"⚠️  Vector store dimension mismatch: {faiss_error}")
            logger.info("📝 Continuing without few-shot examples (will use schema-based SQL generation)")
            return []
        
        logger.debug(f"Raw results: {len(docs_and_scores)} documents returned")
        
        examples = []
        for i, (doc, score) in enumerate(docs_and_scores):
            logger.debug(f"Document {i+1}: score={score}, metadata keys={list(doc.metadata.keys())}")
            logger.debug(f"Document {i+1} content preview: {doc.page_content[:100]}...")
            
            # Extract question and SQL from document
            question_text = doc.metadata.get("question", "")
            sql_text = doc.metadata.get("sql", "")
            
            if not question_text or not sql_text:
                # Try parsing from page_content if metadata is missing
                logger.warning(f"Document {i+1} missing metadata, trying to parse from content")
                if "Question:" in doc.page_content and "SQL:" in doc.page_content:
                    parts = doc.page_content.split("SQL:", 1)
                    question_text = parts[0].replace("Question:", "").strip()
                    sql_text = parts[1].strip() if len(parts) > 1 else ""
            
            examples.append({
                "question": question_text or doc.page_content[:200],
                "sql": sql_text,
                "score": score
            })
        
        logger.info(f"Retrieved {len(examples)} similar examples for question")
        return examples
        
    except Exception as e:
        import traceback
        logger.error(f"Few-shot retrieval failed: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

# ==============================================================================
# DATABASE SCHEMA INFORMATION
# ==============================================================================

def get_db_schema() -> str:
    """
    Get database schema information for Deep Thinking RAG planning
    Returns a concise description of available tables and key columns
    """
    return """
Available Tables:
- patients (patient_id, birth_date, sex, race, ethnicity)
- encounters (encounter_id, patient_id, start_datetime, end_datetime, class, reason_text)
- conditions (condition_id, patient_id, encounter_id, onset_datetime, abatement_datetime, code, display)
- observations (observation_id, patient_id, encounter_id, effective_datetime, loinc_code, display, value_num, value_unit)
- medication_requests (med_request_id, patient_id, encounter_id, start_datetime, end_datetime, med_name, dose, route, refills)
- procedures (procedure_id, patient_id, encounter_id, performed_datetime, code, display)
- immunizations (date, patient_id, encounter_id, code, display, base_cost)
""".strip()

# ==============================================================================
# SQL GENERATION WITH GPT-4
# ==============================================================================

def build_sql_prompt(question: str, examples: List[Dict]) -> str:
    """Build prompt for SQL generation"""
    
    prompt = """You are a medical SQL expert. Generate a PostgreSQL query to answer the patient's question.

DATABASE SCHEMA (use these EXACT column names):
- patients (patient_id, birth_date, sex, race, ethnicity)
  * WARNING: NO 'city' or 'state' columns exist!
- encounters (encounter_id, patient_id, start_datetime, end_datetime, class, reason_text)
  * Note: Use 'start_datetime' NOT 'encounter_datetime'
  * Note: Use 'class' NOT 'encounter_class' 
  * Note: There is NO 'encounter_type' column - use 'class' instead
- conditions (condition_id, patient_id, encounter_id, onset_datetime, abatement_datetime, code, display)
- observations (observation_id, patient_id, encounter_id, effective_datetime, loinc_code, display, value_num, value_unit)
- medication_requests (med_request_id, patient_id, encounter_id, start_datetime, end_datetime, med_name, dose, route, refills)
  * Note: Primary key is 'med_request_id' NOT 'medication_id'
- procedures (procedure_id, patient_id, encounter_id, performed_datetime, code, display)
- immunizations (date, patient_id, encounter_id, code, display, base_cost)
  * WARNING: NO 'immunization_id' column exists! Use (patient_id, code, date) for uniqueness
  * Note: Primary columns are 'date', 'code', 'display' for vaccine information

CRITICAL RULES:
1. ONLY query tables relevant to the question - do NOT join all tables!
2. For patients: NEVER use 'city' or 'state' - they don't exist!
3. For medications: IMPORTANT - dose/route are in med_name, dates come from encounters!
   - ALWAYS JOIN medication_requests with encounters to get prescription dates
   - Use e.start_datetime as prescription_date (medication_requests has NO start_datetime!)
   - Use DISTINCT ON (m.med_name) to avoid duplicates
   - med_name format: "Medication DOSE ROUTE" (e.g., "Lisinopril 10 MG Oral Tablet")
4. For immunizations: use 'display' for vaccine name, 'date' for vaccination date
   - NEVER use 'immunization_id' - it does NOT exist!
5. For conditions/observations/procedures: use both 'code' and 'display'
6. For encounters: ALWAYS use 'start_datetime' and 'class'
7. IMPORTANT: If question is about procedures, ONLY join patients and procedures tables!
8. IMPORTANT: If question is about medications, MUST join medication_requests with encounters for dates!
9. IMPORTANT: If question is about vaccines, ONLY join patients and immunizations!
10. MEDICATION QUERY TEMPLATE (MUST FOLLOW THIS):
    SELECT DISTINCT ON (m.med_name) m.med_name, e.start_datetime as prescription_date
    FROM medication_requests m
    JOIN encounters e ON m.encounter_id = e.encounter_id
    WHERE m.patient_id = :patient_id
    ORDER BY m.med_name, e.start_datetime DESC

EXAMPLES:
"""
    
    for i, ex in enumerate(examples, 1):
        prompt += f"\nExample {i}:\n"
        prompt += f"Question: {ex['question']}\n"
        prompt += f"SQL: {ex['sql']}\n"
    
    prompt += f"""

NOW ANSWER THIS QUESTION:
Question: {question}

REQUIREMENTS:
1. Use :patient_id as the parameter placeholder (it will be converted automatically)
2. NEVER mix :patient_id with %(patient_id)s - use ONLY :patient_id format
3. Return only SELECT statements
4. When using DISTINCT ON, the ORDER BY clause MUST start with the exact same expressions
   Example: DISTINCT ON (a, b) requires ORDER BY a, b, ... (in that exact order)
5. Follow the example queries EXACTLY - copy their structure, especially DISTINCT ON and ORDER BY
6. Include relevant columns only
7. CRITICAL: Use the EXACT column names from the examples above (e.g., med_name, not code)
8. Match the table structure shown in the examples exactly

Generate the SQL query following the examples above EXACTLY:

SQL:"""
    
    return prompt

def generate_sql_with_llm(question: str, examples: List[Dict]) -> str:
    """
    Generate SQL using Groq (ultra-fast) or GPT-4 (fallback)
    
    Groq is 18x faster (0.3s vs 5.4s) for SQL generation!
    """
    try:
        # Try Groq first if enabled (18x faster!)
        if Config.USE_GROQ_FOR_SQL and Config.GROQ_API_KEY:
            try:
                return _generate_sql_with_groq(question, examples)
            except Exception as groq_error:
                logger.warning(f"Groq SQL generation failed, falling back to GPT-4o-mini: {groq_error}")
                # Fall through to GPT-4o-mini
        
        # Fallback to GPT-4o-mini
        return _generate_sql_with_openai(question, examples)
        
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        raise


def _generate_sql_with_groq(question: str, examples: List[Dict]) -> str:
    """Generate SQL using Groq (llama-3.1-8b-instant) - 18x faster than GPT-4o-mini!"""
    from groq import Groq
    
    client = Groq(api_key=Config.GROQ_API_KEY)
    
    prompt = build_sql_prompt(question, examples)
    
    response = client.chat.completions.create(
        model=Config.GROQ_MODEL,
        messages=[
            {
                "role": "system", 
                "content": "You are a PostgreSQL expert for medical databases. You MUST follow the provided examples exactly, especially the DISTINCT ON and ORDER BY syntax. PostgreSQL requires DISTINCT ON expressions to match the initial ORDER BY expressions exactly. Generate ONLY the SQL query, no explanations."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # Maximum consistency with examples
        max_tokens=500
    )
    
    sql = response.choices[0].message.content.strip()
    
    # Clean up SQL - remove markdown, "SQL:" prefix, etc.
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    # Remove common prefixes the LLM might add
    if sql.upper().startswith("SQL:"):
        sql = sql[4:].strip()
    
    logger.info(f"✅ Groq SQL generation (0.2-0.5s): {sql[:150]}...")
    return sql


def _generate_sql_with_openai(question: str, examples: List[Dict]) -> str:
    """Generate SQL using GPT-4o-mini (slower fallback)"""
    from openai import OpenAI
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    prompt = build_sql_prompt(question, examples)
    
    response = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a PostgreSQL expert for medical databases. You MUST follow the provided examples exactly, especially the DISTINCT ON and ORDER BY syntax. PostgreSQL requires DISTINCT ON expressions to match the initial ORDER BY expressions exactly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # Use 0 for maximum consistency with examples
        max_tokens=500
    )
    
    sql = response.choices[0].message.content.strip()
    
    # Clean up SQL - remove markdown, "SQL:" prefix, etc.
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    # Remove common prefixes the LLM might add
    if sql.upper().startswith("SQL:"):
        sql = sql[4:].strip()
    
    logger.info(f"Generated SQL (GPT-4o-mini): {sql[:200]}...")
    return sql

# ==============================================================================
# RESULT SUMMARIZATION
# ==============================================================================

def summarize_results(question: str, df: pd.DataFrame) -> str:
    """Create natural language summary of results"""
    
    if df.empty:
        return "No data found for this query."
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Convert DataFrame to readable format
        data_summary = df.head(10).to_string(index=False)
        
        prompt = f"""Summarize these medical records in a patient-friendly way.

Question: {question}

Data:
{data_summary}

Provide a clear, concise summary in 2-3 sentences."""
        
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a medical assistant explaining data to patients."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return f"Found {len(df)} records. Please review the data table below."

# ==============================================================================
# KNOWLEDGE GRAPH GENERATION
# ==============================================================================

def should_show_knowledge_graph(question: str, df: pd.DataFrame) -> bool:
    """
    Determine if knowledge graph should be generated for this query
    
    KG is shown only for comprehensive queries that benefit from relationship visualization:
    - Patient overview/summary queries
    - Treatment history/all medications
    - Medical history/all conditions
    - Care plan/comprehensive queries
    
    Args:
        question: User's question
        df: Query results dataframe
        
    Returns:
        True if KG should be generated, False otherwise
    """
    if len(df) == 0:
        return False
    
    question_lower = question.lower()
    
    # Keywords for queries that should NOT show KG (too specific/temporal)
    skip_keywords = [
        'encounter', 'appointment', 'visit', 'last', 'recent', 'latest',
        'when', 'date', 'time', 'today', 'yesterday', 'this week'
    ]
    
    # Check if query is too specific (skip KG)
    for keyword in skip_keywords:
        if keyword in question_lower:
            logger.info(f"⏭️  KG skipped: Query contains specific term '{keyword}'")
            return False
    
    # Keywords indicating comprehensive queries that benefit from KG visualization
    kg_keywords = [
        'overview', 'summary', 'history', 'all', 'complete', 'comprehensive',
        'patient health', 'medical record', 'treatment summary', 'care plan',
        'health status', 'medical background', 'patient summary', 'full record'
    ]
    
    # Check if query contains KG-worthy keywords
    for keyword in kg_keywords:
        if keyword in question_lower:
            logger.info(f"🕸️ KG enabled: Question contains '{keyword}'")
            return True
    
    # Also enable for queries returning multiple types of data (5+ records)
    # BUT not if it's just a list of encounters/appointments
    if len(df) >= 5:
        # Check if it's likely an encounter list by column names
        columns_lower = [col.lower() for col in df.columns]
        if 'encounter' in ' '.join(columns_lower) and len(columns_lower) < 6:
            logger.info(f"⏭️  KG skipped: Encounter list ({len(df)} records)")
            return False
        
        logger.info(f"🕸️ KG enabled: Query returned {len(df)} records (comprehensive data)")
        return True
    
    logger.info(f"⏭️  KG skipped: Query too specific ('{question[:50]}...', {len(df)} records)")
    return False


def generate_knowledge_graph(df: pd.DataFrame, patient_id: str) -> str:
    """
    Generate interactive knowledge graph visualization from query results
    
    Args:
        df: Query results dataframe
        patient_id: Patient ID for context
        
    Returns:
        HTML string with interactive Plotly visualization
    """
    try:
        from openai import OpenAI
        
        # Convert DataFrame to structured medical records format
        # The extractor expects a dict with keys like 'medications', 'conditions', etc.
        records_dict = {}
        
        # Detect data type based on columns and organize
        if 'med_name' in df.columns:
            # Medications - parse dose/route from med_name and get dates from encounters
            records_dict['medications'] = []
            seen_meds = set()
            
            for _, row in df.iterrows():
                med_name = row.get('med_name', 'Unknown medication')
                
                # Skip duplicates
                if med_name in seen_meds:
                    continue
                seen_meds.add(med_name)
                
                # Parse dose and route from med_name (format: "Medication DOSE ROUTE")
                # Example: "Lisinopril 10 MG Oral Tablet" -> dose="10 MG", route="Oral"
                import re
                dose = None
                route = None
                
                # Extract dose (e.g., "10 MG", "325 MG", "500 MG")
                dose_match = re.search(r'(\d+\.?\d*)\s*(MG|ML|MCG|G|%)', med_name, re.IGNORECASE)
                if dose_match:
                    dose = f"{dose_match.group(1)} {dose_match.group(2).upper()}"
                
                # Extract route (Oral, Topical, Injectable, etc.)
                if 'oral' in med_name.lower():
                    route = 'Oral'
                elif 'topical' in med_name.lower():
                    route = 'Topical'
                elif 'inject' in med_name.lower():
                    route = 'Injectable'
                elif 'inhal' in med_name.lower():
                    route = 'Inhalation'
                elif 'nasal' in med_name.lower():
                    route = 'Nasal'
                
                # Get prescription date from encounter (if joined)
                prescription_date = None
                if 'prescription_date' in row:
                    prescription_date = str(row.get('prescription_date', ''))[:10]
                elif 'start_datetime' in row:
                    prescription_date = str(row.get('start_datetime', ''))[:10]
                
                med = {
                    'description': med_name,
                    'dose': dose,
                    'route': route,
                    'start_date': prescription_date,
                    'end_date': None  # Database doesn't have end dates
                }
                records_dict['medications'].append(med)
        
        if 'condition_name' in df.columns or 'condition' in df.columns:
            # Conditions
            records_dict['conditions'] = []
            for _, row in df.iterrows():
                cond = {
                    'description': row.get('condition_name') or row.get('condition', 'Unknown condition'),
                    'start_date': str(row.get('start_datetime', '')) if pd.notna(row.get('start_datetime')) else None,
                    'clinical_status': row.get('clinical_status')
                }
                records_dict['conditions'].append(cond)
        
        if 'procedure_name' in df.columns or 'procedure' in df.columns:
            # Procedures
            records_dict['procedures'] = []
            for _, row in df.iterrows():
                proc = {
                    'description': row.get('procedure_name') or row.get('procedure', 'Unknown procedure'),
                    'date': str(row.get('procedure_datetime', '')) if pd.notna(row.get('procedure_datetime')) else None,
                    'status': row.get('status')
                }
                records_dict['procedures'].append(proc)
        
        if 'observation_name' in df.columns or 'observation' in df.columns:
            # Observations/Lab Results
            records_dict['observations'] = []
            for _, row in df.iterrows():
                obs = {
                    'description': row.get('observation_name') or row.get('observation', 'Unknown observation'),
                    'value': row.get('value'),
                    'units': row.get('units'),
                    'date': str(row.get('observation_datetime', '')) if pd.notna(row.get('observation_datetime')) else None
                }
                records_dict['observations'].append(obs)
        
        if 'immunization_name' in df.columns or 'vaccine_name' in df.columns:
            # Immunizations
            records_dict['immunizations'] = []
            for _, row in df.iterrows():
                imm = {
                    'display': row.get('immunization_name') or row.get('vaccine_name', 'Unknown vaccine'),
                    'date': str(row.get('immunization_datetime', '')) if pd.notna(row.get('immunization_datetime')) else None,
                    'status': row.get('status')
                }
                records_dict['immunizations'].append(imm)
        
        # If no specific structure detected, create a generic format from all data
        if not records_dict:
            # Fallback: convert all rows to a simple text representation
            records_dict['general_records'] = []
            for _, row in df.iterrows():
                # Create a text summary of the row
                text_parts = []
                for col, val in row.items():
                    if pd.notna(val) and val is not None and str(val).strip():
                        text_parts.append(f"{col}: {val}")
                if text_parts:
                    records_dict['general_records'].append({
                        'description': ', '.join(text_parts)
                    })
        
        # Initialize KG components
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        builder = MedicalGraphBuilder()
        visualizer = MedicalGraphVisualizer()
        
        # SIMPLIFIED: Extract triples directly via LLM from text
        # Build text summary from records_dict
        text_parts = []
        for category, items in records_dict.items():
            if items:
                text_parts.append(f"\n{category.upper()}:")
                for item in items[:10]:  # Limit for tokens
                    desc = item.get('description', str(item))
                    text_parts.append(f"  - {desc}")
        
        medical_text = "\n".join(text_parts)
        logger.info(f"� Medical context: {len(medical_text)} chars")
        
        # Call LLM to extract triples
        extraction_prompt = f"""Extract medical knowledge graph triples from this data:

{medical_text}

Return triples in format: (subject, predicate, object)
Examples:
- (Patient, has_condition, Diabetes)
- (Patient, takes_medication, Metformin)
- (Metformin, treats, Diabetes)

Return ONLY the triples, one per line."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract (subject, predicate, object) triples from medical data. Return ONLY triples."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.0,
            max_tokens=800
        )
        
        raw_output = response.choices[0].message.content.strip()
        logger.info(f"🤖 LLM output: {len(raw_output)} chars")
        
        # Parse triples
        triples = []
        for line in raw_output.split('\n'):
            line = line.strip()
            if '(' in line and ')' in line:
                try:
                    start = line.index('(')
                    end = line.rindex(')')
                    content = line[start+1:end]
                    parts = [p.strip().strip('"').strip("'") for p in content.split(',')]
                    if len(parts) == 3 and all(parts):
                        triples.append({'subject': parts[0], 'predicate': parts[1], 'object': parts[2]})
                except:
                    pass
        
        if not triples:
            logger.warning(f"⚠️ No triples extracted")
            return "<div style='padding: 20px; text-align: center; color: gray;'>📊 No relationships found in the data</div>"
        
        logger.info(f"✅ Extracted {len(triples)} triples")
        
        # Build graph
        graph = builder.build_graph(triples)
        
        # Generate visualization
        kg_html = visualizer.visualize(
            graph,
            title=f"Patient {patient_id[:8]}... - Medical Knowledge Graph"
        )
        
        # Wrap in iframe for better Gradio compatibility
        if kg_html and len(kg_html) > 100:  # Valid HTML should be longer
            logger.info(f"✅ Generated HTML visualization ({len(kg_html)} chars)")
            
            # Escape the HTML properly for iframe srcdoc
            import html as html_module
            escaped_html = html_module.escape(kg_html)
            
            # Use iframe with srcdoc for better isolation and rendering
            wrapped_html = f"""
            <div style="width: 100%; height: 850px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
                <iframe 
                    srcdoc="{escaped_html}" 
                    style="width: 100%; height: 100%; border: none;"
                    frameborder="0"
                    scrolling="yes"
                ></iframe>
            </div>
            """
            return wrapped_html
        else:
            logger.warning(f"⚠️ Generated HTML seems incomplete: {len(kg_html) if kg_html else 0} chars")
            return "<div style='padding: 20px; text-align: center; color: orange;'>⚠️ Visualization generation incomplete</div>"
        
    except Exception as e:
        logger.error(f"❌ Knowledge graph generation error: {e}")
        raise

# ==============================================================================
# MAIN QUERY PROCESSING PIPELINE
# ==============================================================================

def process_patient_query(patient_id: str, question: str, vectorstore) -> tuple:
    """
    Main pipeline: Question → SQL → Execute → Summarize
    
    Returns:
        (summary_text, dataframe, sql_query, status)
    """
    try:
        # Step 1: Retrieve few-shot examples (will use schema if unavailable)
        logger.info(f"Processing query for patient {patient_id}: {question}")
        examples = get_few_shot_examples(vectorstore, question, k=3)
        
        if not examples:
            logger.info("No few-shot examples available - will use schema-based SQL generation")
            examples = []  # Continue with empty list - LLM will use schema instead
        
        # Step 2: Generate SQL
        sql = generate_sql_with_llm(question, examples)
        
        if not sql:
            return ("Error: Could not generate SQL query", 
                    pd.DataFrame(), "", "", "error")  # 5 values
        
        # Step 3: Execute SQL
        params = {"patient_id": patient_id}
        df = execute_sql_query(sql, params)
        
        # Step 4: Summarize results
        summary = summarize_results(question, df)
        
        # Step 5: Generate Knowledge Graph (only for comprehensive queries)
        kg_html = None
        if KG_AVAILABLE and should_show_knowledge_graph(question, df):
            try:
                logger.info("🕸️ Generating knowledge graph visualization...")
                kg_html = generate_knowledge_graph(df, patient_id)
                logger.info("✅ Knowledge graph generated successfully")
            except Exception as kg_error:
                logger.warning(f"⚠️  Knowledge graph generation failed: {kg_error}")
                kg_html = f"<div style='padding: 20px; text-align: center; color: orange;'>Knowledge graph visualization unavailable: {str(kg_error)}</div>"
        else:
            if not KG_AVAILABLE:
                kg_html = "<div style='padding: 20px; text-align: center; color: gray;'>Knowledge graph module not installed</div>"
            elif len(df) == 0:
                kg_html = "<div style='padding: 20px; text-align: center; color: gray;'>No data to visualize</div>"
            else:
                # Query was successful but not comprehensive enough for KG
                kg_html = """<div style='padding: 20px; text-align: center; color: #666;'>
                    <p style='font-size: 16px; margin-bottom: 10px;'>💡 Knowledge Graph</p>
                    <p style='font-size: 14px;'>Shown for comprehensive queries like:</p>
                    <ul style='list-style: none; padding: 0; margin-top: 10px;'>
                        <li>• "Patient health overview"</li>
                        <li>• "Summary of all treatments"</li>
                        <li>• "Complete medical history"</li>
                        <li>• "All medications and conditions"</li>
                    </ul>
                </div>"""
        
        logger.info(f"✅ Query successful: {len(df)} records returned")
        
        return (summary, df, sql, kg_html, "success")
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        empty_kg = "<div style='padding: 20px; text-align: center; color: red;'>Error occurred</div>"
        return (error_msg, pd.DataFrame(), "", empty_kg, "error")

# ==============================================================================
# GRADIO WEB INTERFACE
# ==============================================================================

#def create_gradio_interface(vectorstore):
# OLD FUNCTION - COMMENTED OUT TO USE ENHANCED VERSION FROM enhanced_gradio_ui_rag_cag.py
# def create_enhanced_gradio_interface(vectorstore,engine):
#     """Create Gradio web interface with Deep Thinking RAG toggle"""
# 
#     # Lazy load Deep Thinking RAG to avoid circular import warning
#     _load_deep_thinking_rag()    # Initialize Deep Thinking RAG
    deep_thinking_rag = None
    if DEEP_THINKING_AVAILABLE:
        try:
            # Get database schema for Deep Thinking RAG
            db_schema = get_db_schema()
            
            deep_thinking_rag = IntegratedMedicalRAG(
                vector_store=vectorstore,
                sql_engine=None,  # We use execute_sql_query function directly
                db_schema=db_schema
            )
            logger.info("✅ Deep Thinking RAG initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️  Deep Thinking RAG initialization failed: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
    
    def query_handler(patient_id: str, question: str, use_deep_thinking: bool = False):
        """Handle query from Gradio interface with mode selection"""
        
        if not patient_id or not question:
            empty_kg = "<div style='padding: 20px; text-align: center; color: gray;'>Please provide Patient ID and Question</div>"
            return "Please provide both Patient ID and Question", pd.DataFrame(), "", empty_kg, ""
        
        # Choose processing mode
        if use_deep_thinking and deep_thinking_rag:
            # Use Deep Thinking RAG
            try:
                import time
                start_time = time.time()
                
                logger.info(f"🧠 Using Deep Thinking RAG mode for: {question}")
                result = deep_thinking_rag.run(question, patient_id)
                
                execution_time = time.time() - start_time
                
                # Format comprehensive answer
                summary = result["final_answer"]
                
                # Create execution details
                execution_info = f"""
### 🧠 Deep Thinking RAG Analysis

**Execution Time**: {execution_time:.2f} seconds  
**Planning Steps**: {len(result['plan'].steps)}  
**Documents Retrieved**: {sum(len(r['documents']) for r in result['step_results'])}  
**Sources Used**: EHR + Knowledge Base + Clinical Guidelines

**Reasoning**: {result['plan'].reasoning}

**Steps Executed**:
"""
                for i, step in enumerate(result['plan'].steps, 1):
                    execution_info += f"\n{i}. {step.sub_question} [`{step.tool}`]"
                
                # Extract data from step results (if any)
                df = pd.DataFrame()
                sql_query = "-- Deep Thinking RAG (Multi-step approach)\n"
                for i, step_result in enumerate(result['step_results'], 1):
                    sql_query += f"\n-- Step {i}: {step_result['sub_question']}\n"
                    sql_query += f"-- Tool: {step_result['tool']}\n"
                    sql_query += f"-- Retrieved: {len(step_result['documents'])} documents\n"
                
                # Create knowledge graph (placeholder for now)
                kg_html = "<div style='padding: 20px; text-align: center; color: #666;'><h3>🧠 Deep Thinking Mode</h3><p>Multi-source comprehensive analysis complete. Knowledge Graph visualization coming soon for Deep Thinking mode!</p></div>"
                
                return summary, df, sql_query, kg_html, execution_info
                
            except Exception as e:
                logger.error(f"❌ Deep Thinking RAG error: {e}")
                return f"Error in Deep Thinking mode: {str(e)}", pd.DataFrame(), "", "", ""
        
        else:
            # Use standard fast SQL RAG
            logger.info(f"⚡ Using Fast SQL RAG mode for: {question}")
            summary, df, sql, kg_html, status = process_patient_query(patient_id, question, vectorstore)
            execution_info = f"""
### ⚡ Fast SQL RAG

**Mode**: Direct SQL query  
**Response Time**: ~1-3 seconds  
**Data Source**: EHR Database only  
**Status**: {status}
"""
            return summary, df, sql, kg_html, execution_info
    
    # Define interface with custom CSS
    custom_css = """
    #kg_visualization {
        min-height: 850px !important;
        overflow: visible !important;
    }
    #kg_visualization iframe {
        min-height: 850px !important;
    }
    .deep-thinking-toggle {
        font-size: 16px !important;
        font-weight: bold !important;
    }
    """
    
    with gr.Blocks(title="Medical RAG Query System", theme=gr.themes.Soft(), css=custom_css) as app:
        gr.Markdown("# 🏥 Medical Intelligence RAG System")
        gr.Markdown("### Natural Language Patient Data Queries with AI-Powered Analysis")
        gr.Markdown("Ask questions about patient medical records in plain English")
        
        with gr.Row():
            with gr.Column(scale=1):
                patient_id_input = gr.Textbox(
                    label="Patient ID",
                    placeholder="8c8e1c9a-b310-43c6-33a7-ad11bad21c40",
                    value="8c8e1c9a-b310-43c6-33a7-ad11bad21c40"
                )
                
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What vaccines have I received?",
                    lines=3
                )
                
                # Deep Thinking Toggle (NEW!)
                deep_thinking_toggle = gr.Checkbox(
                    label="🧠 Deep Thinking Mode",
                    value=False,
                    info="Enable AI planning for complex multi-source analysis (slower, more comprehensive)",
                    elem_classes=["deep-thinking-toggle"],
                    interactive=DEEP_THINKING_AVAILABLE
                )
                
                # Mode explanation
                with gr.Accordion("ℹ️ Mode Comparison", open=False):
                    gr.Markdown("""
**⚡ Fast SQL RAG** (Default):
- Response Time: 1-3 seconds
- Data Source: EHR Database
- Best For: Simple queries (medications, lab results, conditions)
- Answer Length: Concise, focused

**🧠 Deep Thinking RAG**:
- Response Time: 20-40 seconds
- Data Sources: EHR + Medical Knowledge + Clinical Guidelines
- Best For: Complex questions requiring analysis, drug interactions, treatment recommendations
- Answer Length: Comprehensive, detailed (300+ words)
- Features: Multi-step reasoning, evidence synthesis, clinical context

**Examples**:
- Simple: "What medications am I taking?" → Use Fast Mode ⚡
- Complex: "Given my diabetes and recent HbA1c, what should I discuss with my doctor?" → Use Deep Thinking 🧠
                    """)
                
                submit_btn = gr.Button("🔍 Search Medical Records", variant="primary", size="lg")
                
                # Quick question buttons
                gr.Markdown("### 💨 Quick Questions (Fast Mode)")
                with gr.Row():
                    q1 = gr.Button("💉 Vaccines", size="sm")
                    q2 = gr.Button("💊 Medications", size="sm")
                    q3 = gr.Button("🩺 Conditions", size="sm")
                with gr.Row():
                    q4 = gr.Button("🔬 Lab Results", size="sm")
                    q5 = gr.Button("⚕️ Procedures", size="sm")
                    q6 = gr.Button("📋 Encounters", size="sm")
                
                # Complex question examples (NEW!)
                if DEEP_THINKING_AVAILABLE:
                    gr.Markdown("### 🧠 Complex Analysis (Deep Thinking)")
                    with gr.Row():
                        c1 = gr.Button("💊 Drug Interactions", size="sm")
                        c2 = gr.Button("📊 Trend Analysis", size="sm")
                    with gr.Row():
                        c3 = gr.Button("⚕️ Treatment Review", size="sm")
                        c4 = gr.Button("🎯 Risk Assessment", size="sm")
        
        with gr.Column(scale=2):
            summary_output = gr.Textbox(
                label="📊 Answer",
                lines=8,
                interactive=False
            )
            
            # Execution info (NEW!)
            with gr.Accordion("⚙️ Execution Details", open=False):
                execution_output = gr.Markdown(
                    value="Run a query to see execution details"
                )
            
            # Knowledge Graph Visualization
            with gr.Accordion("🕸️ Knowledge Graph Visualization", open=True):
                kg_output = gr.HTML(
                    label="Interactive Medical Knowledge Graph",
                    value="<div style='padding: 20px; text-align: center; color: gray;'>Run a query to see the knowledge graph</div>",
                    elem_id="kg_visualization",
                    show_label=False
                )
            
            data_output = gr.Dataframe(
                label="📋 Detailed Results",
                wrap=True
            )
        
            with gr.Accordion("💻 Generated SQL Query / Execution Plan", open=False):
                sql_output = gr.Code(
                    label="SQL",
                    language="sql",
                    interactive=False
                )
        
        # Event handlers
        submit_btn.click(
            fn=query_handler,
            inputs=[patient_id_input, question_input, deep_thinking_toggle],
            outputs=[summary_output, data_output, sql_output, kg_output, execution_output]
        )
        
        # Quick question handlers (Fast mode - turn off deep thinking)
        q1.click(
            lambda: ("What vaccines have I received?", False),
            outputs=[question_input, deep_thinking_toggle]
        )
        q2.click(
            lambda: ("What medications am I currently taking?", False),
            outputs=[question_input, deep_thinking_toggle]
        )
        q3.click(
            lambda: ("What are my active medical conditions?", False),
            outputs=[question_input, deep_thinking_toggle]
        )
        q4.click(
            lambda: ("Show me my latest lab results", False),
            outputs=[question_input, deep_thinking_toggle]
        )
        q5.click(
            lambda: ("What procedures have I had?", False),
            outputs=[question_input, deep_thinking_toggle]
        )
        q6.click(
            lambda: ("Show me my recent medical appointments", False),
            outputs=[question_input, deep_thinking_toggle]
        )
        
        # Complex question handlers (Deep thinking mode - turn on!)
        if DEEP_THINKING_AVAILABLE:
            c1.click(
                lambda: ("Are there any concerning drug interactions between my current medications? Include cardiac risk considerations.", True),
                outputs=[question_input, deep_thinking_toggle]
            )
            c2.click(
                lambda: ("How has my HbA1c trended over the past year, and what does this mean for my diabetes management?", True),
                outputs=[question_input, deep_thinking_toggle]
            )
            c3.click(
                lambda: ("Given my current medications and recent lab results, what concerns should I discuss with my doctor?", True),
                outputs=[question_input, deep_thinking_toggle]
            )
            c4.click(
                lambda: ("Based on my medical history and current conditions, what is my cardiovascular risk profile?", True),
                outputs=[question_input, deep_thinking_toggle]
            )
        
        # Add footer with mode indicator
        gr.Markdown("""
---
**System Status**: 
- ⚡ Fast SQL RAG: ✅ Always Available
- 🧠 Deep Thinking RAG: {}
- 🕸️ Knowledge Graph: {}

*Tip: Use Fast mode for simple data retrieval, Deep Thinking for complex analysis and clinical insights.*
        """.format(
            "✅ Available" if DEEP_THINKING_AVAILABLE and deep_thinking_rag else "⚠️ Unavailable",
            "✅ Available" if KG_AVAILABLE else "⚠️ Unavailable"
        ))
    
    return app

# ==============================================================================
# SYSTEM HEALTH CHECKS
# ==============================================================================

def test_db_connection():
    """Test database connectivity"""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        conn.close()
        logger.info(f"✅ Database connection successful: {version[:50]}")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def test_vectorstore():
    """Test FAISS vector store"""
    try:
        vs = load_vector_store()
        logger.info("✅ Vector store loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        return False

def test_openai():
    """Test OpenAI API"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        logger.info("✅ OpenAI API working")
        return True
    except Exception as e:
        logger.error(f"❌ OpenAI API test failed: {e}")
        return False

def run_health_checks():
    """Run all system health checks"""
    logger.info("=" * 60)
    logger.info(" Running System Health Checks")
    logger.info("=" * 60)
    
    # Initialize connection pool first
    try:
        initialize_connection_pool()
    except Exception as e:
        logger.error(f"❌ Failed to initialize connection pool: {e}")
        raise
    
    checks = {
        "Configuration": lambda: Config.validate() or True,
        "Database Connection": test_db_connection,
        "FAISS Vector Store": test_vectorstore,
        "OpenAI API": test_openai
    }
    
    all_passed = True
    for name, check_fn in checks.items():
        try:
            result = check_fn()
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status} - {name}") 
        except Exception as e:
            logger.error(f"❌ FAIL - {name}: {e}")
            all_passed = False
    
    logger.info("=" * 60)
    
    if not all_passed:
        raise RuntimeError("Health checks failed - cannot start application")
    
    logger.info(" All health checks passed!")
    return True

# ==============================================================================
# MAIN APPLICATION ENTRY POINT
# ==============================================================================

def main():
    """Main application entry point"""
    
    logger.info("=" * 60)
    logger.info("  Starting Medical RAG System - AWS Production")
    logger.info("=" * 60)
    
    try:
        # Run health checks
        run_health_checks()
        
        # Load vector store
        logger.info(" Loading FAISS vector store...")
        vectorstore = load_vector_store()
        
        # Create Gradio interface
        logger.info(" Creating web interface...")
        #app = create_gradio_interface(vectorstore)
        # Note: This system uses psycopg2 connections, not SQLAlchemy engines
        # Pass None as engine since it's not used in this implementation
        # IMPORTANT: Using enhanced interface from enhanced_gradio_ui_rag_cag.py (not local function)
        from enhanced_gradio_ui_rag_cag import create_enhanced_gradio_interface as enhanced_interface
        app = enhanced_interface(vectorstore, engine=None)
        
        # Launch application
        logger.info("=" * 60)
        logger.info(" Application Ready!")
        logger.info(f"cd .. Starting Gradio server on {Config.SERVER_NAME}:{Config.SERVER_PORT}")
        logger.info(f" Access at: http://your-ec2-ip:{Config.SERVER_PORT}")
        logger.info("=" * 60)
        
        app.launch(
            server_name=Config.SERVER_NAME,
            server_port=Config.SERVER_PORT,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"❌ Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()

